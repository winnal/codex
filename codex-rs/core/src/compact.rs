use std::sync::Arc;
use std::time::Instant;

use crate::Prompt;
use crate::client::ModelClientSession;
use crate::client_common::ResponseEvent;
use crate::compact_exact_tail::CompactionHistoryPolicy;
use crate::compact_exact_tail::EXACT_TAIL_LOCAL_RETAINED_COLD_USER_MESSAGE_BUDGET_TOKENS;
use crate::compact_exact_tail::ExactTailImplementation;
use crate::compact_exact_tail::ExactTailPrepareInput;
use crate::compact_exact_tail::build_exact_tail_replacement;
use crate::compact_exact_tail::check_cold_input_fits;
use crate::compact_exact_tail::emit_exact_tail_compaction_diagnostic;
use crate::compact_exact_tail::emit_exact_tail_prepare_failure_diagnostic;
use crate::compact_exact_tail::ensure_non_empty_local_summary;
use crate::compact_exact_tail::exact_tail_backend_context_exceeded_error;
use crate::compact_exact_tail::exact_tail_cold_input_too_large_error;
use crate::compact_exact_tail::local_summary_scaffold_overhead_tokens;
use crate::compact_exact_tail::normalize_tool_outputs_for_exact_tail_policy;
use crate::compact_exact_tail::pending_exact_tail_tool_surface_hint;
use crate::compact_exact_tail::prepare_exact_tail_plan;
pub(crate) use crate::compact_route::CompactRoute;
pub(crate) use crate::compact_route::compact_route;
use crate::config::Config;
use crate::context_manager::estimate_response_items_token_count;
use crate::hook_runtime::PostCompactHookOutcome;
use crate::hook_runtime::PreCompactHookOutcome;
use crate::hook_runtime::run_post_compact_hooks;
use crate::hook_runtime::run_pre_compact_hooks;
use crate::responses_metadata::CodexResponsesMetadata;
use crate::responses_metadata::CodexResponsesRequestKind;
use crate::responses_metadata::CompactionTurnMetadata;
#[cfg(test)]
use crate::session::PreviousTurnSettings;
use crate::session::session::Session;
use crate::session::turn::get_last_assistant_message_from_turn;
use crate::session::turn_context::TurnContext;
use crate::util::backoff;
use codex_analytics::CodexCompactionEvent;
use codex_analytics::CompactionImplementation;
use codex_analytics::CompactionPhase;
use codex_analytics::CompactionReason;
use codex_analytics::CompactionStatus;
use codex_analytics::CompactionStrategy;
use codex_analytics::CompactionTrigger;
use codex_analytics::now_unix_seconds;
use codex_app_server_protocol::ConfigLayerSource;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::items::ContextCompactionItem;
use codex_protocol::items::TurnItem;
use codex_protocol::models::ContentItem;
use codex_protocol::models::InternalChatMessageMetadataPassthrough;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::CompactedItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::TurnStartedEvent;
use codex_protocol::protocol::WarningEvent;
use codex_protocol::user_input::UserInput;
use codex_rollout_trace::InferenceTraceContext;
use codex_utils_output_truncation::TruncationPolicy;
use codex_utils_output_truncation::approx_token_count;
use codex_utils_output_truncation::truncate_text;
use futures::prelude::*;
use tracing::error;

use codex_model_provider_info::ModelProviderInfo;

pub use codex_prompts::SUMMARIZATION_PROMPT;
pub use codex_prompts::SUMMARY_PREFIX;
pub(crate) const COMPACT_USER_MESSAGE_MAX_TOKENS: usize = 20_000;

/// Controls whether compaction replacement history must carry initial context.
///
/// Pre-turn/manual compaction clears the reference context item and lets the next turn reinject.
/// Mid-turn compaction must inject context above the last real user so the summary stays last.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InitialContextInjection {
    BeforeLastUserMessage,
    DoNotInject,
}

pub(crate) fn should_use_remote_compact_task(provider: &ModelProviderInfo) -> bool {
    provider.supports_remote_compaction()
}

#[cfg(test)]
pub(crate) fn should_use_remote_compact_task_v2(turn_context: &TurnContext) -> bool {
    should_use_remote_compact_task_v2_for_config(&turn_context.config)
}

pub(crate) fn should_use_remote_compact_task_v2_for_config(config: &Config) -> bool {
    if !config
        .features
        .enabled(codex_features::Feature::RemoteCompactionV2)
    {
        return false;
    }

    if matches!(
        CompactionHistoryPolicy::from_config(config),
        CompactionHistoryPolicy::PreserveRecentExact { .. }
    ) && !remote_compaction_v2_explicitly_enabled(config)
    {
        tracing::info!(
            exact_tail_enabled = true,
            implementation = "remote",
            bypassed_implementation = "remote_v2",
            "exact-tail compaction bypassed default remote v2 route"
        );
        return false;
    }

    true
}

fn remote_compaction_v2_explicitly_enabled(config: &Config) -> bool {
    config
        .config_layer_stack
        .layers_high_to_low()
        .into_iter()
        .filter(|layer| {
            matches!(
                layer.name,
                ConfigLayerSource::User { .. } | ConfigLayerSource::Project { .. }
            )
        })
        .find_map(remote_compaction_v2_layer_value)
        .unwrap_or(false)
}

fn remote_compaction_v2_layer_value(layer: &codex_config::ConfigLayerEntry) -> Option<bool> {
    layer
        .config
        .get("features")?
        .get(codex_features::Feature::RemoteCompactionV2.key())?
        .as_bool()
}

pub(crate) async fn run_inline_auto_compact_task(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
    initial_context_injection: InitialContextInjection,
    reason: CompactionReason,
    phase: CompactionPhase,
) -> CodexResult<()> {
    let prompt = turn_context
        .config
        .compact_prompt
        .as_deref()
        .unwrap_or(SUMMARIZATION_PROMPT)
        .to_string();
    let input = vec![UserInput::Text {
        text: prompt,
        // Compaction prompt is synthesized; no UI element ranges to preserve.
        text_elements: Vec::new(),
    }];

    run_compact_task_inner(
        sess,
        turn_context,
        input,
        initial_context_injection,
        CompactionTrigger::Auto,
        reason,
        phase,
    )
    .await?;
    Ok(())
}

pub(crate) async fn run_compact_task(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
    input: Vec<UserInput>,
) -> CodexResult<()> {
    let start_event = EventMsg::TurnStarted(TurnStartedEvent {
        turn_id: turn_context.sub_id.clone(),
        trace_id: turn_context.trace_id.clone(),
        started_at: turn_context.turn_timing_state.started_at_unix_secs().await,
        model_context_window: turn_context.model_context_window(),
        collaboration_mode_kind: turn_context.collaboration_mode.mode,
    });
    sess.send_event(&turn_context, start_event).await;
    run_compact_task_inner(
        sess.clone(),
        turn_context,
        input,
        InitialContextInjection::DoNotInject,
        CompactionTrigger::Manual,
        CompactionReason::UserRequested,
        CompactionPhase::StandaloneTurn,
    )
    .await?;
    Ok(())
}

async fn run_compact_task_inner(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
    input: Vec<UserInput>,
    initial_context_injection: InitialContextInjection,
    trigger: CompactionTrigger,
    reason: CompactionReason,
    phase: CompactionPhase,
) -> CodexResult<()> {
    let compaction_metadata =
        CompactionTurnMetadata::new(trigger, reason, CompactionImplementation::Responses, phase);
    let attempt = CompactionAnalyticsAttempt::begin(
        sess.as_ref(),
        turn_context.as_ref(),
        trigger,
        reason,
        CompactionImplementation::Responses,
        phase,
    )
    .await;
    let pre_compact_outcome = run_pre_compact_hooks(&sess, &turn_context, trigger).await;
    match pre_compact_outcome {
        PreCompactHookOutcome::Continue => {}
        PreCompactHookOutcome::Stopped => {
            let error = CodexErr::TurnAborted;
            attempt
                .track(
                    sess.as_ref(),
                    CompactionStatus::Interrupted,
                    Some(&error),
                    CompactionAnalyticsDetails::default(),
                )
                .await;
            return Err(error);
        }
    }
    let result = run_compact_task_inner_impl(
        Arc::clone(&sess),
        Arc::clone(&turn_context),
        input,
        initial_context_injection,
        trigger,
        compaction_metadata,
    )
    .await;
    let status = compaction_status_from_result(&result);
    let codex_error = result.as_ref().err();
    if result.is_ok() {
        let post_compact_outcome = run_post_compact_hooks(&sess, &turn_context, trigger).await;
        if let PostCompactHookOutcome::Stopped = post_compact_outcome {
            attempt
                .track(
                    sess.as_ref(),
                    status,
                    codex_error,
                    CompactionAnalyticsDetails::default(),
                )
                .await;
            return Err(CodexErr::TurnAborted);
        }
    }
    attempt
        .track(
            sess.as_ref(),
            status,
            codex_error,
            CompactionAnalyticsDetails::default(),
        )
        .await;
    result.map(|_| ())
}

async fn run_compact_task_inner_impl(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
    input: Vec<UserInput>,
    initial_context_injection: InitialContextInjection,
    trigger: CompactionTrigger,
    compaction_metadata: CompactionTurnMetadata,
) -> CodexResult<String> {
    let context_compaction_item = ContextCompactionItem::new();
    let compaction_id = context_compaction_item.id.clone();
    let compaction_item = TurnItem::ContextCompaction(context_compaction_item);
    sess.emit_turn_item_started(&turn_context, &compaction_item)
        .await;
    let initial_input_for_turn: ResponseInputItem = ResponseInputItem::from(input);

    let mut source_history = sess.clone_history().await;
    let base_instructions = sess.get_base_instructions().await;
    let policy = CompactionHistoryPolicy::from_config(&turn_context.config);
    let normalized_tool_output_count = normalize_tool_outputs_for_exact_tail_policy(
        &mut source_history,
        policy,
        turn_context.model_info.truncation_policy.into(),
    );
    let source_history_items = source_history.raw_items().to_vec();
    let exact_tail_plan = match prepare_exact_tail_plan(ExactTailPrepareInput {
        sess: sess.as_ref(),
        turn_context: turn_context.as_ref(),
        history_items: &source_history_items,
        base_instructions: &base_instructions,
        policy,
        trigger,
        initial_context_injection,
        estimated_summary_scaffold_overhead_tokens: local_summary_scaffold_overhead_tokens(),
        retained_cold_user_message_budget_tokens:
            EXACT_TAIL_LOCAL_RETAINED_COLD_USER_MESSAGE_BUDGET_TOKENS,
        normalized_tool_output_count,
        implementation: ExactTailImplementation::Local,
    })
    .await
    {
        Ok(plan) => plan,
        Err(error) => {
            emit_exact_tail_prepare_failure_diagnostic(
                sess.as_ref(),
                turn_context.as_ref(),
                &compaction_id,
                trigger,
                ExactTailImplementation::Local,
                &error,
            )
            .await;
            let error = error.into_codex_err();
            send_local_compaction_error(&sess, turn_context.as_ref(), &error).await;
            return Err(error);
        }
    };

    let mut history = source_history;
    if let Some(prepared) = &exact_tail_plan {
        history.replace(prepared.plan.cold_history.clone());
    }
    history.record_items(
        &[initial_input_for_turn.into()],
        turn_context.model_info.truncation_policy.into(),
    );
    if let Some(prepared) = &exact_tail_plan {
        if let Err(error) = check_cold_input_fits(
            &prepared.plan,
            history.raw_items(),
            turn_context.model_context_window(),
        ) {
            let failure_reason = error.reason;
            emit_exact_tail_compaction_diagnostic(
                sess.as_ref(),
                turn_context.as_ref(),
                &compaction_id,
                trigger,
                &prepared.plan,
                None,
                Some(failure_reason),
            )
            .await;
            let error = error.into_codex_err();
            send_local_compaction_error(&sess, turn_context.as_ref(), &error).await;
            return Err(error);
        }
        if let Some(context_window) = turn_context.model_context_window()
            && let Some(request_tokens) =
                history.estimate_token_count_with_base_instructions(&base_instructions)
            && request_tokens > context_window
        {
            let error = exact_tail_cold_input_too_large_error(request_tokens, context_window);
            emit_exact_tail_compaction_diagnostic(
                sess.as_ref(),
                turn_context.as_ref(),
                &compaction_id,
                trigger,
                &prepared.plan,
                None,
                Some(super::compact_exact_tail::ExactTailFailReason::ColdInputTooLarge),
            )
            .await;
            send_local_compaction_error(&sess, turn_context.as_ref(), &error).await;
            return Err(error);
        }
    }

    let max_retries = turn_context.provider.info().stream_max_retries();
    let mut retries = 0;
    let mut client_session = sess.services.model_client.new_session();
    // Reuse one client session so turn-scoped state survives retries within this compact turn.
    let window_id = sess.current_window_id().await;
    let responses_metadata = turn_context.turn_metadata_state.to_responses_metadata(
        sess.installation_id.clone(),
        window_id,
        CodexResponsesRequestKind::Compaction(compaction_metadata),
    );

    let compaction_output = loop {
        // Clone is required because of the loop
        let turn_input = history
            .clone()
            .for_prompt(&turn_context.model_info.input_modalities);
        let turn_input_len = turn_input.len();
        let prompt = Prompt {
            input: turn_input,
            base_instructions: base_instructions.clone(),
            ..Default::default()
        };
        let drain_mode = if exact_tail_plan.is_some() {
            LocalCompactionDrainMode::BufferOnly
        } else {
            LocalCompactionDrainMode::RecordToSession
        };
        let attempt_result = drain_to_completed(
            &sess,
            turn_context.as_ref(),
            &mut client_session,
            &responses_metadata,
            &prompt,
            drain_mode,
        )
        .await;

        match attempt_result {
            Ok(output) => {
                break output;
            }
            Err(err @ (CodexErr::Interrupted | CodexErr::TurnAborted)) => {
                return Err(err);
            }
            Err(e @ CodexErr::ContextWindowExceeded) => {
                if exact_tail_plan.is_some() {
                    let error = exact_tail_backend_context_exceeded_error();
                    error!(
                        "Exact-tail local compaction request exceeded backend context window without a safe cold-pruning repair. Original error: {e}"
                    );
                    if let Some(prepared) = &exact_tail_plan {
                        emit_exact_tail_compaction_diagnostic(
                            sess.as_ref(),
                            turn_context.as_ref(),
                            &compaction_id,
                            trigger,
                            &prepared.plan,
                            None,
                            Some(
                                super::compact_exact_tail::ExactTailFailReason::BackendContextExceeded,
                            ),
                        )
                        .await;
                    }
                    send_local_compaction_error(&sess, turn_context.as_ref(), &error).await;
                    return Err(error);
                }
                if turn_input_len > 1 {
                    // Trim from the beginning to preserve cache (prefix-based) and keep recent messages intact.
                    error!(
                        "Context window exceeded while compacting; removing oldest history item. Error: {e}"
                    );
                    history.remove_first_item();
                    retries = 0;
                    continue;
                }
                sess.set_total_tokens_full(turn_context.as_ref()).await;
                sess.track_turn_codex_error(turn_context.as_ref(), &e);
                let event = EventMsg::Error(e.to_error_event(/*message_prefix*/ None));
                sess.send_event(&turn_context, event).await;
                return Err(e);
            }
            Err(e) => {
                if retries < max_retries {
                    retries += 1;
                    let delay = backoff(retries);
                    sess.notify_stream_error(
                        turn_context.as_ref(),
                        format!("Reconnecting... {retries}/{max_retries}"),
                        e,
                    )
                    .await;
                    tokio::time::sleep(delay).await;
                    continue;
                } else {
                    sess.track_turn_codex_error(turn_context.as_ref(), &e);
                    let event = EventMsg::Error(e.to_error_event(/*message_prefix*/ None));
                    sess.send_event(&turn_context, event).await;
                    return Err(e);
                }
            }
        }
    };

    let (summary_suffix, user_messages) = match &exact_tail_plan {
        Some(prepared) => {
            let summary_suffix =
                get_last_assistant_message_from_turn(&compaction_output.completed_items)
                    .unwrap_or_default();
            if let Err(error) = ensure_non_empty_local_summary(&prepared.plan, &summary_suffix) {
                let failure_reason = error.reason;
                emit_exact_tail_compaction_diagnostic(
                    sess.as_ref(),
                    turn_context.as_ref(),
                    &compaction_id,
                    trigger,
                    &prepared.plan,
                    None,
                    Some(failure_reason),
                )
                .await;
                let error = error.into_codex_err();
                send_local_compaction_error(&sess, turn_context.as_ref(), &error).await;
                return Err(error);
            }
            (summary_suffix, prepared.plan.cold_user_messages.clone())
        }
        None => {
            let history_snapshot = sess.clone_history().await;
            let history_items = history_snapshot.raw_items();
            (
                get_last_assistant_message_from_turn(history_items).unwrap_or_default(),
                collect_user_messages(history_items),
            )
        }
    };
    let summary_text = format!("{SUMMARY_PREFIX}\n{summary_suffix}");

    let mut new_history = build_compacted_history(Vec::new(), &user_messages, &summary_text);
    if let Some(summary_item) = new_history.last_mut() {
        // This replacement history skips `record_conversation_items`; only the appended summary
        // belongs to this compaction turn.
        summary_item.set_turn_id_if_missing(&turn_context.sub_id);
    }
    let mut exact_tail_tool_surface_hint = None;
    if let Some(prepared) = &exact_tail_plan {
        match build_exact_tail_replacement(
            prepared,
            new_history,
            initial_context_injection,
            estimate_response_items_token_count(&compaction_output.completed_items),
        ) {
            Ok(replacement) => {
                emit_exact_tail_compaction_diagnostic(
                    sess.as_ref(),
                    turn_context.as_ref(),
                    &compaction_id,
                    trigger,
                    &prepared.plan,
                    Some(&replacement.diagnostics),
                    None,
                )
                .await;
                exact_tail_tool_surface_hint = Some(pending_exact_tail_tool_surface_hint(
                    &compaction_id,
                    &replacement,
                    prepared.plan.diagnostics.implementation,
                ));
                new_history = replacement.replacement_history;
            }
            Err(error) => {
                emit_exact_tail_compaction_diagnostic(
                    sess.as_ref(),
                    turn_context.as_ref(),
                    &compaction_id,
                    trigger,
                    &prepared.plan,
                    None,
                    Some(error.reason),
                )
                .await;
                let error = error.into_codex_err();
                send_local_compaction_error(&sess, turn_context.as_ref(), &error).await;
                return Err(error);
            }
        }
    }
    let (window_number, window_ids) = sess.advance_auto_compact_window().await;

    if exact_tail_plan.is_none()
        && matches!(
            initial_context_injection,
            InitialContextInjection::BeforeLastUserMessage
        )
    {
        let initial_context = sess.build_initial_context(turn_context.as_ref()).await;
        new_history =
            insert_initial_context_before_last_real_user_or_summary(new_history, initial_context);
    }
    let reference_context_item = match initial_context_injection {
        InitialContextInjection::DoNotInject => None,
        InitialContextInjection::BeforeLastUserMessage => Some(turn_context.to_turn_context_item()),
    };
    let compacted_item = CompactedItem {
        message: summary_text.clone(),
        replacement_history: Some(new_history.clone()),
        window_number: Some(window_number),
        first_window_id: Some(window_ids.first_window_id.to_string()),
        previous_window_id: window_ids.previous_window_id.map(|id| id.to_string()),
        window_id: Some(window_ids.window_id.to_string()),
    };
    sess.replace_compacted_history(
        turn_context.as_ref(),
        new_history,
        reference_context_item,
        compacted_item,
    )
    .await;
    if let Some(hint) = exact_tail_tool_surface_hint {
        sess.set_active_exact_tail_tool_surface_hint(hint).await;
    }
    sess.recompute_token_usage(&turn_context).await;

    sess.emit_turn_item_completed(&turn_context, compaction_item)
        .await;
    let warning = EventMsg::Warning(WarningEvent {
        message: "Heads up: Long threads and multiple compactions can cause the model to be less accurate. Start a new thread when possible to keep threads small and targeted.".to_string(),
    });
    sess.send_event(&turn_context, warning).await;
    Ok(summary_suffix)
}

async fn send_local_compaction_error(sess: &Session, turn_context: &TurnContext, error: &CodexErr) {
    sess.track_turn_codex_error(turn_context, error);
    let event = EventMsg::Error(error.to_error_event(/*message_prefix*/ None));
    sess.send_event(turn_context, event).await;
}

pub(crate) struct CompactionAnalyticsAttempt {
    thread_id: String,
    turn_id: String,
    trigger: CompactionTrigger,
    reason: CompactionReason,
    implementation: CompactionImplementation,
    phase: CompactionPhase,
    active_context_tokens_before: i64,
    started_at: u64,
    start_instant: Instant,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct CompactionAnalyticsDetails {
    pub(crate) active_context_tokens_before: Option<i64>,
    pub(crate) retained_image_count: Option<usize>,
    pub(crate) compaction_summary_tokens: Option<i64>,
    pub(crate) cached_input_tokens: Option<i64>,
}

impl CompactionAnalyticsAttempt {
    pub(crate) async fn begin(
        sess: &Session,
        turn_context: &TurnContext,
        trigger: CompactionTrigger,
        reason: CompactionReason,
        implementation: CompactionImplementation,
        phase: CompactionPhase,
    ) -> Self {
        let active_context_tokens_before = sess.get_total_token_usage().await;
        Self {
            thread_id: sess.thread_id.to_string(),
            turn_id: turn_context.sub_id.clone(),
            trigger,
            reason,
            implementation,
            phase,
            active_context_tokens_before,
            started_at: now_unix_seconds(),
            start_instant: Instant::now(),
        }
    }

    pub(crate) async fn track(
        self,
        sess: &Session,
        status: CompactionStatus,
        codex_error: Option<&CodexErr>,
        details: CompactionAnalyticsDetails,
    ) {
        let CompactionAnalyticsDetails {
            active_context_tokens_before,
            retained_image_count,
            compaction_summary_tokens,
            cached_input_tokens,
        } = details;
        let active_context_tokens_before =
            active_context_tokens_before.unwrap_or(self.active_context_tokens_before);
        let active_context_tokens_after = sess.get_total_token_usage().await;
        sess.services
            .analytics_events_client
            .track_compaction(CodexCompactionEvent {
                thread_id: self.thread_id,
                turn_id: self.turn_id,
                trigger: self.trigger,
                reason: self.reason,
                implementation: self.implementation,
                phase: self.phase,
                strategy: CompactionStrategy::Memento,
                status,
                codex_error_kind: codex_error.map(Into::into),
                codex_error_http_status_code: codex_error
                    .and_then(CodexErr::http_status_code_value),
                active_context_tokens_before,
                active_context_tokens_after,
                retained_image_count,
                compaction_summary_tokens,
                cached_input_tokens,
                started_at: self.started_at,
                completed_at: now_unix_seconds(),
                duration_ms: Some(
                    u64::try_from(self.start_instant.elapsed().as_millis()).unwrap_or(u64::MAX),
                ),
            });
    }
}

pub(crate) fn compaction_status_from_result<T>(result: &CodexResult<T>) -> CompactionStatus {
    match result {
        Ok(_) => CompactionStatus::Completed,
        Err(CodexErr::Interrupted | CodexErr::TurnAborted) => CompactionStatus::Interrupted,
        Err(_) => CompactionStatus::Failed,
    }
}

pub fn content_items_to_text(content: &[ContentItem]) -> Option<String> {
    let mut pieces = Vec::new();
    for item in content {
        match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                if !text.is_empty() {
                    pieces.push(text.as_str());
                }
            }
            ContentItem::InputImage { .. } => {}
        }
    }
    if pieces.is_empty() {
        None
    } else {
        Some(pieces.join("\n"))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompactedUserMessage {
    message: String,
    internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
}

pub(crate) fn collect_user_messages(items: &[ResponseItem]) -> Vec<CompactedUserMessage> {
    items
        .iter()
        .filter_map(|item| match crate::event_mapping::parse_turn_item(item) {
            Some(TurnItem::UserMessage(user)) => {
                if is_summary_message(&user.message()) {
                    None
                } else {
                    Some(CompactedUserMessage {
                        message: user.message(),
                        internal_chat_message_metadata_passthrough: match item {
                            ResponseItem::Message {
                                internal_chat_message_metadata_passthrough,
                                ..
                            } => internal_chat_message_metadata_passthrough.clone(),
                            _ => None,
                        },
                    })
                }
            }
            _ => None,
        })
        .collect()
}

pub(crate) fn is_summary_message(message: &str) -> bool {
    message.starts_with(format!("{SUMMARY_PREFIX}\n").as_str())
}

/// Inserts canonical initial context into compacted replacement history at the
/// model-expected boundary.
///
/// Placement rules:
/// - Prefer immediately before the last real user message.
/// - If no real user messages remain, insert before the compaction summary so
///   the summary stays last.
/// - If there are no user messages, insert before the last compaction item so
///   that item remains last (remote compaction may return only compaction items).
/// - If there are no user messages or compaction items, append the context.
pub(crate) fn insert_initial_context_before_last_real_user_or_summary(
    mut compacted_history: Vec<ResponseItem>,
    initial_context: Vec<ResponseItem>,
) -> Vec<ResponseItem> {
    let mut last_user_or_summary_index = None;
    let mut last_real_user_index = None;
    for (i, item) in compacted_history.iter().enumerate().rev() {
        let Some(TurnItem::UserMessage(user)) = crate::event_mapping::parse_turn_item(item) else {
            continue;
        };
        // Compaction summaries are encoded as user messages, so track both:
        // the last real user message (preferred insertion point) and the last
        // user-message-like item (fallback summary insertion point).
        last_user_or_summary_index.get_or_insert(i);
        if !is_summary_message(&user.message()) {
            last_real_user_index = Some(i);
            break;
        }
    }
    let last_compaction_index = compacted_history
        .iter()
        .enumerate()
        .rev()
        .find_map(|(i, item)| {
            matches!(
                item,
                ResponseItem::Compaction { .. } | ResponseItem::ContextCompaction { .. }
            )
            .then_some(i)
        });
    let insertion_index = last_real_user_index
        .or(last_user_or_summary_index)
        .or(last_compaction_index);

    // Re-inject canonical context from the current session since we stripped it
    // from the pre-compaction history. Prefer placing it before the last real
    // user message; if there is no real user message left, place it before the
    // summary or compaction item so the compaction item remains last.
    if let Some(insertion_index) = insertion_index {
        compacted_history.splice(insertion_index..insertion_index, initial_context);
    } else {
        compacted_history.extend(initial_context);
    }

    compacted_history
}

pub(crate) fn build_compacted_history(
    initial_context: Vec<ResponseItem>,
    user_messages: &[CompactedUserMessage],
    summary_text: &str,
) -> Vec<ResponseItem> {
    build_compacted_history_with_limit(
        initial_context,
        user_messages,
        summary_text,
        COMPACT_USER_MESSAGE_MAX_TOKENS,
    )
}

fn build_compacted_history_with_limit(
    mut history: Vec<ResponseItem>,
    user_messages: &[CompactedUserMessage],
    summary_text: &str,
    max_tokens: usize,
) -> Vec<ResponseItem> {
    let mut selected_messages: Vec<CompactedUserMessage> = Vec::new();
    if max_tokens > 0 {
        let mut remaining = max_tokens;
        for message in user_messages.iter().rev() {
            if remaining == 0 {
                break;
            }
            let tokens = approx_token_count(&message.message);
            if tokens <= remaining {
                selected_messages.push(message.clone());
                remaining = remaining.saturating_sub(tokens);
            } else {
                let truncated =
                    truncate_text(&message.message, TruncationPolicy::Tokens(remaining));
                selected_messages.push(CompactedUserMessage {
                    message: truncated,
                    internal_chat_message_metadata_passthrough: message
                        .internal_chat_message_metadata_passthrough
                        .clone(),
                });
                break;
            }
        }
        selected_messages.reverse();
    }

    for message in &selected_messages {
        history.push(ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: message.message.clone(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: message
                .internal_chat_message_metadata_passthrough
                .clone(),
        });
    }

    let summary_text = if summary_text.is_empty() {
        "(no summary available)".to_string()
    } else {
        summary_text.to_string()
    };

    history.push(ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText { text: summary_text }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    });

    history
}

struct LocalCompactionOutput {
    completed_items: Vec<ResponseItem>,
}

#[derive(Clone, Copy)]
enum LocalCompactionDrainMode {
    RecordToSession,
    BufferOnly,
}

async fn drain_to_completed(
    sess: &Session,
    turn_context: &TurnContext,
    client_session: &mut ModelClientSession,
    responses_metadata: &CodexResponsesMetadata,
    prompt: &Prompt,
    mode: LocalCompactionDrainMode,
) -> CodexResult<LocalCompactionOutput> {
    let mut stream = client_session
        .stream(
            prompt,
            &turn_context.model_info,
            &turn_context.session_telemetry,
            turn_context.reasoning_effort.clone(),
            turn_context.reasoning_summary,
            turn_context.config.service_tier.clone(),
            responses_metadata,
            // Rollout tracing currently models remote compaction only; local compaction streams
            // are left untraced until the reducer has a first-class local compaction lifecycle.
            &InferenceTraceContext::disabled(),
        )
        .await?;
    let mut completed_items = Vec::new();
    loop {
        let maybe_event = stream.next().await;
        let Some(event) = maybe_event else {
            return Err(CodexErr::Stream(
                "stream closed before response.completed".into(),
                None,
            ));
        };
        match event {
            Ok(ResponseEvent::OutputItemDone(item)) => match mode {
                LocalCompactionDrainMode::RecordToSession => {
                    sess.record_conversation_items(turn_context, std::slice::from_ref(&item))
                        .await;
                }
                LocalCompactionDrainMode::BufferOnly => {
                    completed_items.push(item);
                }
            },
            Ok(ResponseEvent::ServerReasoningIncluded(included)) => {
                sess.set_server_reasoning_included(included).await;
            }
            Ok(ResponseEvent::RateLimits(snapshot)) => {
                sess.update_rate_limits(turn_context, snapshot).await;
            }
            Ok(ResponseEvent::Completed { token_usage, .. }) => {
                sess.update_token_usage_info(turn_context, token_usage.as_ref())
                    .await?;
                return Ok(LocalCompactionOutput { completed_items });
            }
            Ok(_) => continue,
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
#[path = "compact_tests.rs"]
mod tests;
