use std::sync::Arc;

use crate::Prompt;
use crate::ResponseStream;
use crate::client::ModelClientSession;
use crate::client_common::ResponseEvent;
use crate::compact::CompactionAnalyticsAttempt;
use crate::compact::CompactionAnalyticsDetails;
use crate::compact::InitialContextInjection;
use crate::compact::compaction_status_from_result;
use crate::compact_exact_tail::CompactionHistoryPolicy;
use crate::compact_exact_tail::ExactTailImplementation;
use crate::compact_exact_tail::ExactTailPrepareInput;
use crate::compact_exact_tail::build_exact_tail_replacement;
use crate::compact_exact_tail::check_cold_input_fits;
use crate::compact_exact_tail::emit_exact_tail_compaction_diagnostic;
use crate::compact_exact_tail::emit_exact_tail_prepare_failure_diagnostic;
use crate::compact_exact_tail::ensure_replacement_has_cold_summary;
use crate::compact_exact_tail::exact_tail_backend_context_exceeded_error;
use crate::compact_exact_tail::exact_tail_cold_input_too_large_error;
use crate::compact_exact_tail::normalize_tool_outputs_for_exact_tail_policy;
use crate::compact_exact_tail::pending_exact_tail_tool_surface_hint;
use crate::compact_exact_tail::prepare_exact_tail_plan;
use crate::compact_remote::process_compacted_history;
use crate::compact_remote::trim_function_call_history_to_fit_context_window;
use crate::compact_remote_v2_retention::REMOTE_COMPACTION_V2_RETAINED_MESSAGE_TOKEN_BUDGET;
use crate::compact_remote_v2_retention::build_v2_compacted_history;
use crate::context_manager::estimate_response_items_token_count;
use crate::hook_runtime::PostCompactHookOutcome;
use crate::hook_runtime::PreCompactHookOutcome;
use crate::hook_runtime::run_post_compact_hooks;
use crate::hook_runtime::run_pre_compact_hooks;
use crate::responses_metadata::CodexResponsesMetadata;
use crate::responses_metadata::CodexResponsesRequestKind;
use crate::responses_metadata::CompactionTurnMetadata;
use crate::responses_retry::ResponsesStreamRequest;
use crate::responses_retry::handle_retryable_response_stream_error;
use crate::session::session::Session;
use crate::session::turn::built_tools_without_exact_tail_tool_surface_hint;
use crate::session::turn_context::TurnContext;
use codex_analytics::CompactionImplementation;
use codex_analytics::CompactionPhase;
use codex_analytics::CompactionReason;
use codex_analytics::CompactionTrigger;
use codex_features::Feature;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::items::ContextCompactionItem;
use codex_protocol::items::TurnItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::CompactedItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::TokenUsage;
use codex_protocol::protocol::TurnStartedEvent;
use codex_rollout_trace::CompactionCheckpointTracePayload;
use codex_rollout_trace::InferenceTraceContext;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;
use tracing::info;

// Compact attempts can run much longer than normal turns, so keep the per-transport
// retry budget smaller than the general Responses stream retry budget.
const MAX_REMOTE_COMPACTION_V2_STREAM_RETRIES: u64 = 2;

pub(crate) async fn run_inline_remote_auto_compact_task(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
    client_session: &mut ModelClientSession,
    initial_context_injection: InitialContextInjection,
    reason: CompactionReason,
    phase: CompactionPhase,
) -> CodexResult<()> {
    run_remote_compact_task_inner(
        &sess,
        &turn_context,
        Some(client_session),
        initial_context_injection,
        CompactionTrigger::Auto,
        reason,
        phase,
    )
    .await
}

pub(crate) async fn run_remote_compact_task(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
) -> CodexResult<()> {
    let start_event = EventMsg::TurnStarted(TurnStartedEvent {
        turn_id: turn_context.sub_id.clone(),
        trace_id: turn_context.trace_id.clone(),
        started_at: turn_context.turn_timing_state.started_at_unix_secs().await,
        model_context_window: turn_context.model_context_window(),
        collaboration_mode_kind: turn_context.collaboration_mode.mode,
    });
    sess.send_event(&turn_context, start_event).await;

    run_remote_compact_task_inner(
        &sess,
        &turn_context,
        /*client_session*/ None,
        InitialContextInjection::DoNotInject,
        CompactionTrigger::Manual,
        CompactionReason::UserRequested,
        CompactionPhase::StandaloneTurn,
    )
    .await
}

async fn run_remote_compact_task_inner(
    sess: &Arc<Session>,
    turn_context: &Arc<TurnContext>,
    client_session: Option<&mut ModelClientSession>,
    initial_context_injection: InitialContextInjection,
    trigger: CompactionTrigger,
    reason: CompactionReason,
    phase: CompactionPhase,
) -> CodexResult<()> {
    let compaction_metadata = CompactionTurnMetadata::new(
        trigger,
        reason,
        CompactionImplementation::ResponsesCompactionV2,
        phase,
    );
    let mut analytics_details = CompactionAnalyticsDetails {
        active_context_tokens_before: Some(sess.get_total_token_usage().await),
        ..Default::default()
    };
    let attempt = CompactionAnalyticsAttempt::begin(
        sess.as_ref(),
        turn_context.as_ref(),
        trigger,
        reason,
        CompactionImplementation::ResponsesCompactionV2,
        phase,
    )
    .await;
    let pre_compact_outcome = run_pre_compact_hooks(sess, turn_context, trigger).await;
    match pre_compact_outcome {
        PreCompactHookOutcome::Continue => {}
        PreCompactHookOutcome::Stopped => {
            let error = CodexErr::TurnAborted;
            attempt
                .track(
                    sess.as_ref(),
                    codex_analytics::CompactionStatus::Interrupted,
                    Some(&error),
                    analytics_details,
                )
                .await;
            return Err(error);
        }
    }
    let result = run_remote_compact_task_inner_impl(
        sess,
        turn_context,
        client_session,
        initial_context_injection,
        trigger,
        compaction_metadata,
        &mut analytics_details,
    )
    .await;
    let status = compaction_status_from_result(&result);
    let codex_error = result.as_ref().err();
    if result.is_ok() {
        let post_compact_outcome = run_post_compact_hooks(sess, turn_context, trigger).await;
        if let PostCompactHookOutcome::Stopped = post_compact_outcome {
            attempt
                .track(sess.as_ref(), status, codex_error, analytics_details)
                .await;
            return Err(CodexErr::TurnAborted);
        }
    }
    attempt
        .track(sess.as_ref(), status, codex_error, analytics_details)
        .await;
    if matches!(&result, Err(CodexErr::TurnAborted)) {
        return result;
    }
    if let Err(err) = result {
        sess.track_turn_codex_error(turn_context, &err);
        let event = EventMsg::Error(
            err.to_error_event(Some("Error running remote compact task".to_string())),
        );
        sess.send_event(turn_context, event).await;
        return Err(err);
    }
    Ok(())
}

async fn run_remote_compact_task_inner_impl(
    sess: &Arc<Session>,
    turn_context: &Arc<TurnContext>,
    client_session: Option<&mut ModelClientSession>,
    initial_context_injection: InitialContextInjection,
    trigger: CompactionTrigger,
    compaction_metadata: CompactionTurnMetadata,
    analytics_details: &mut CompactionAnalyticsDetails,
) -> CodexResult<()> {
    let context_compaction_item = ContextCompactionItem::new();
    let compaction_id = context_compaction_item.id.clone();
    let compaction_trace = sess.services.rollout_thread_trace.compaction_trace_context(
        turn_context.sub_id.as_str(),
        context_compaction_item.id.as_str(),
        turn_context.model_info.slug.as_str(),
        turn_context.provider.info().name.as_str(),
    );
    let compaction_item = TurnItem::ContextCompaction(context_compaction_item);
    sess.emit_turn_item_started(turn_context, &compaction_item)
        .await;

    let mut source_history = sess.clone_history().await;
    let base_instructions = sess.get_base_instructions().await;
    let policy = CompactionHistoryPolicy::from_config(&turn_context.config);
    let normalized_tool_output_count = normalize_tool_outputs_for_exact_tail_policy(
        &mut source_history,
        policy,
        turn_context.model_info.truncation_policy.into(),
    );
    let source_history_items = source_history.raw_items().to_vec();
    let exact_tail_implementation = ExactTailImplementation::RemoteV2;
    let exact_tail_plan = match prepare_exact_tail_plan(ExactTailPrepareInput {
        sess: sess.as_ref(),
        turn_context: turn_context.as_ref(),
        history_items: &source_history_items,
        base_instructions: &base_instructions,
        policy,
        trigger,
        initial_context_injection,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: REMOTE_COMPACTION_V2_RETAINED_MESSAGE_TOKEN_BUDGET
            as i64,
        normalized_tool_output_count,
        implementation: exact_tail_implementation,
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
                exact_tail_implementation,
                &error,
            )
            .await;
            return Err(error.into_codex_err());
        }
    };

    let mut history = source_history;
    if let Some(prepared) = &exact_tail_plan {
        history.replace(prepared.plan.cold_history.clone());
    }

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
            return Err(error.into_codex_err());
        }
        if let Some(context_window) = turn_context.model_context_window()
            && let Some(request_tokens) =
                history.estimate_token_count_with_base_instructions(&base_instructions)
            && request_tokens > context_window
        {
            emit_exact_tail_compaction_diagnostic(
                sess.as_ref(),
                turn_context.as_ref(),
                &compaction_id,
                trigger,
                &prepared.plan,
                None,
                Some(crate::compact_exact_tail::ExactTailFailReason::ColdInputTooLarge),
            )
            .await;
            return Err(exact_tail_cold_input_too_large_error(
                request_tokens,
                context_window,
            ));
        }
    } else {
        let (rewritten_outputs, estimated_deleted_tokens) =
            trim_function_call_history_to_fit_context_window(
                &mut history,
                turn_context.as_ref(),
                &base_instructions,
            );
        if rewritten_outputs > 0 {
            info!(
                turn_id = %turn_context.sub_id,
                rewritten_outputs,
                "rewrote history outputs before remote compaction v2"
            );
        }
        if estimated_deleted_tokens > 0 {
            let max_local_deleted_tokens = sess
                .estimated_tokens_after_last_model_generated_item()
                .await;
            analytics_details.active_context_tokens_before = analytics_details
                .active_context_tokens_before
                .map(|active_context_tokens_before| {
                    active_context_tokens_before
                        .saturating_sub(estimated_deleted_tokens.min(max_local_deleted_tokens))
                });
        }
    }

    let trace_input_history = history.raw_items().to_vec();
    let prompt_input = history.for_prompt(&turn_context.model_info.input_modalities);
    let tool_router = built_tools_without_exact_tail_tool_surface_hint(
        sess.as_ref(),
        turn_context.as_ref(),
        &CancellationToken::new(),
    )
    .await?;
    let mut input = prompt_input.clone();
    input.push(ResponseItem::CompactionTrigger {});
    let prompt = Prompt {
        input,
        tools: tool_router.model_visible_specs(),
        parallel_tool_calls: turn_context.model_info.supports_parallel_tool_calls,
        base_instructions,
        output_schema: None,
        output_schema_strict: true,
    };

    let window_id = sess.current_window_id().await;
    let responses_metadata = turn_context.turn_metadata_state.to_responses_metadata(
        sess.installation_id.clone(),
        window_id,
        CodexResponsesRequestKind::Compaction(compaction_metadata),
    );
    let trace_attempt = compaction_trace.start_attempt(&serde_json::json!({
        "model": turn_context.model_info.slug.as_str(),
        "instructions": prompt.base_instructions.text.as_str(),
        "input": &prompt.input,
        "parallel_tool_calls": prompt.parallel_tool_calls,
    }));

    let mut client_session = client_session;
    let mut owned_client_session;
    let compaction_output_result = if exact_tail_plan.is_some() {
        let mut isolated_client_session = sess
            .services
            .model_client
            .clone()
            .with_prompt_cache_key_override(Some(format!(
                "exact-tail-v2-compaction:{}",
                turn_context.sub_id
            )))
            .with_beta_feature_advertised(Feature::RemoteCompactionV2.key())
            .new_ephemeral_session();
        let turn_state = client_session.as_ref().map(|session| session.turn_state());
        if let Some(turn_state) = turn_state {
            isolated_client_session = isolated_client_session.with_turn_state(turn_state);
        }
        run_remote_compaction_request_v2(
            sess,
            turn_context,
            &mut isolated_client_session,
            &prompt,
            &responses_metadata,
        )
        .await
    } else {
        let standard_client_session = match client_session.as_deref_mut() {
            Some(client_session) => client_session,
            None => {
                owned_client_session = sess.services.model_client.new_session();
                &mut owned_client_session
            }
        };
        run_remote_compaction_request_v2(
            sess,
            turn_context,
            standard_client_session,
            &prompt,
            &responses_metadata,
        )
        .await
    };

    trace_attempt.record_result(
        compaction_output_result
            .as_ref()
            .map(|output| std::slice::from_ref(&output.compaction_output)),
    );
    let RemoteCompactionV2Output {
        compaction_output,
        token_usage,
    } = match compaction_output_result {
        Ok(output) => output,
        Err(error) => {
            if let Some(prepared) = &exact_tail_plan
                && matches!(error, CodexErr::ContextWindowExceeded)
            {
                emit_exact_tail_compaction_diagnostic(
                    sess.as_ref(),
                    turn_context.as_ref(),
                    &compaction_id,
                    trigger,
                    &prepared.plan,
                    None,
                    Some(crate::compact_exact_tail::ExactTailFailReason::BackendContextExceeded),
                )
                .await;
                return Err(exact_tail_backend_context_exceeded_error());
            }
            return Err(error);
        }
    };
    if let Some(token_usage) = token_usage {
        sess.record_rollout_budget_usage(&token_usage)?;
        if exact_tail_plan.is_none() {
            analytics_details.active_context_tokens_before = Some(token_usage.input_tokens);
        }
        analytics_details.compaction_summary_tokens = Some(token_usage.output_tokens);
        analytics_details.cached_input_tokens = Some(token_usage.cached_input_tokens);
    }
    let compaction_output_tokens =
        estimate_response_items_token_count(std::slice::from_ref(&compaction_output));
    if let Some(prepared) = &exact_tail_plan
        && let Err(error) = ensure_replacement_has_cold_summary(
            &prepared.plan,
            std::slice::from_ref(&compaction_output),
        )
    {
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
        return Err(error.into_codex_err());
    }
    let (compacted_history, retained_images) =
        build_v2_compacted_history(&prompt_input, compaction_output);
    analytics_details.retained_image_count = Some(retained_images);
    let mut exact_tail_tool_surface_hint = None;
    let new_history = if let Some(prepared) = &exact_tail_plan {
        let actual_summary_tokens = compaction_output_tokens;
        let replacement = match build_exact_tail_replacement(
            prepared,
            compacted_history,
            initial_context_injection,
            actual_summary_tokens,
        ) {
            Ok(replacement) => replacement,
            Err(error) => {
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
                return Err(error.into_codex_err());
            }
        };
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
        if let Some(client_session) = client_session {
            client_session.reset_responses_continuation();
        }
        sess.services
            .model_client
            .reset_cached_responses_continuation();
        replacement.replacement_history
    } else {
        process_compacted_history(
            sess.as_ref(),
            turn_context.as_ref(),
            compacted_history,
            initial_context_injection,
        )
        .await
    };
    let (new_window_number, new_window_ids) = sess.advance_auto_compact_window().await;

    let reference_context_item = match initial_context_injection {
        InitialContextInjection::DoNotInject => None,
        InitialContextInjection::BeforeLastUserMessage => Some(turn_context.to_turn_context_item()),
    };
    let compacted_item = CompactedItem {
        message: String::new(),
        replacement_history: Some(new_history.clone()),
        window_number: Some(new_window_number),
        first_window_id: Some(new_window_ids.first_window_id.to_string()),
        previous_window_id: new_window_ids.previous_window_id.map(|id| id.to_string()),
        window_id: Some(new_window_ids.window_id.to_string()),
    };
    compaction_trace.record_installed(&CompactionCheckpointTracePayload {
        input_history: &trace_input_history,
        replacement_history: &new_history,
    });
    sess.replace_compacted_history(
        turn_context.as_ref(),
        new_history,
        reference_context_item,
        compacted_item,
    )
    .await;
    if let Some(hint) = exact_tail_tool_surface_hint {
        sess.set_pending_exact_tail_tool_surface_hint(hint).await;
    }
    sess.recompute_token_usage(turn_context).await;

    sess.emit_turn_item_completed(turn_context, compaction_item)
        .await;
    Ok(())
}

pub(crate) struct RemoteCompactionV2Output {
    pub(crate) compaction_output: ResponseItem,
    pub(crate) token_usage: Option<TokenUsage>,
}

pub(crate) async fn run_remote_compaction_request_v2(
    sess: &Session,
    turn_context: &TurnContext,
    client_session: &mut ModelClientSession,
    prompt: &Prompt,
    responses_metadata: &CodexResponsesMetadata,
) -> CodexResult<RemoteCompactionV2Output> {
    let max_retries = turn_context
        .provider
        .info()
        .stream_max_retries()
        .min(MAX_REMOTE_COMPACTION_V2_STREAM_RETRIES);
    let mut retries = 0;
    loop {
        let result = match client_session
            .stream(
                prompt,
                &turn_context.model_info,
                &turn_context.session_telemetry,
                turn_context.reasoning_effort.clone(),
                turn_context.reasoning_summary,
                turn_context.config.service_tier.clone(),
                responses_metadata,
                &InferenceTraceContext::disabled(),
            )
            .await
        {
            Ok(stream) => collect_compaction_output(stream).await,
            Err(err) => Err(err),
        };

        match result {
            Ok(compaction_output) => return Ok(compaction_output),
            Err(err) if !err.is_retryable() => return Err(err),
            Err(err) => {
                handle_retryable_response_stream_error(
                    &mut retries,
                    max_retries,
                    err,
                    client_session,
                    sess,
                    turn_context,
                    ResponsesStreamRequest::RemoteCompactionV2,
                )
                .await?;
            }
        }
    }
}

async fn collect_compaction_output(
    mut stream: ResponseStream,
) -> CodexResult<RemoteCompactionV2Output> {
    let mut output_item_count = 0usize;
    let mut compaction_count = 0usize;
    let mut compaction_output = None;
    let mut saw_completed = false;
    let mut completed_token_usage = None;
    while let Some(event) = stream.next().await {
        match event? {
            ResponseEvent::OutputItemDone(item) => {
                output_item_count += 1;
                if let ResponseItem::Compaction { .. } = item {
                    compaction_count += 1;
                    if compaction_output.is_none() {
                        compaction_output = Some(item);
                    }
                }
            }
            ResponseEvent::Completed { token_usage, .. } => {
                saw_completed = true;
                completed_token_usage = token_usage;
                break;
            }
            _ => {}
        }
    }

    if !saw_completed {
        return Err(CodexErr::Stream(
            "remote compaction v2 stream closed before response.completed".to_string(),
            None,
        ));
    }

    if compaction_count != 1 {
        return Err(CodexErr::Fatal(format!(
            "remote compaction v2 expected exactly one compaction output item, got {compaction_count} from {output_item_count} output items"
        )));
    }

    let Some(compaction_output) = compaction_output else {
        unreachable!("compaction output must exist when count is exactly one");
    };
    Ok(RemoteCompactionV2Output {
        compaction_output,
        token_usage: completed_token_usage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::models::ContentItem;
    use codex_protocol::models::MessagePhase;
    use pretty_assertions::assert_eq;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    fn message(role: &str, text: &str, phase: Option<MessagePhase>) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: role.to_string(),
            content: vec![ContentItem::InputText {
                text: text.to_string(),
            }],
            phase,
            internal_chat_message_metadata_passthrough: None,
        }
    }

    fn response_stream(events: Vec<CodexResult<ResponseEvent>>) -> ResponseStream {
        let (tx_event, rx_event) = mpsc::channel(events.len().max(1));
        for event in events {
            tx_event
                .try_send(event)
                .expect("response stream test channel should have capacity");
        }
        drop(tx_event);
        ResponseStream {
            rx_event,
            consumer_dropped: CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn collect_compaction_output_accepts_additional_output_items() {
        let compaction = ResponseItem::Compaction {
            id: None,
            encrypted_content: "encrypted".to_string(),
            internal_chat_message_metadata_passthrough: None,
        };
        let stream = response_stream(vec![
            Ok(ResponseEvent::OutputItemDone(message(
                "assistant",
                "IGNORED_COMPACT_REPLY",
                Some(MessagePhase::FinalAnswer),
            ))),
            Ok(ResponseEvent::OutputItemDone(compaction.clone())),
            Ok(ResponseEvent::Completed {
                response_id: "resp-compact".to_string(),
                token_usage: Some(TokenUsage {
                    input_tokens: 123_456,
                    cached_input_tokens: 7_890,
                    output_tokens: 42,
                    reasoning_output_tokens: 5,
                    total_tokens: 123_498,
                }),
                end_turn: Some(true),
            }),
        ]);

        let output = collect_compaction_output(stream)
            .await
            .expect("compaction should be collected");

        assert_eq!(output.compaction_output, compaction);
        assert_eq!(
            output.token_usage,
            Some(TokenUsage {
                input_tokens: 123_456,
                cached_input_tokens: 7_890,
                output_tokens: 42,
                reasoning_output_tokens: 5,
                total_tokens: 123_498,
            })
        );
    }
}
