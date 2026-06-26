use crate::Prompt;
use crate::client::ModelClientSession;
use crate::compact::CompactionAnalyticsAttempt;
use crate::compact::CompactionAnalyticsDetails;
use crate::compact::InitialContextInjection;
use crate::compact::compaction_status_from_result;
use crate::compact_exact_tail::CompactionHistoryPolicy;
use crate::compact_exact_tail::ExactTailFailReason;
use crate::compact_exact_tail::ExactTailImplementation;
use crate::compact_exact_tail::ExactTailPrepareInput;
use crate::compact_exact_tail::append_hot_suffix_to_replacement;
use crate::compact_exact_tail::build_exact_tail_replacement;
use crate::compact_exact_tail::check_cold_input_fits;
use crate::compact_exact_tail::emit_exact_tail_compaction_diagnostic;
use crate::compact_exact_tail::emit_exact_tail_prepare_failure_diagnostic;
use crate::compact_exact_tail::ensure_replacement_has_cold_summary;
use crate::compact_exact_tail::exact_tail_backend_context_exceeded_error;
use crate::compact_exact_tail::exact_tail_cold_input_too_large_error;
use crate::compact_exact_tail::normalize_tool_outputs_for_exact_tail_policy;
use crate::compact_exact_tail::prepare_exact_tail_plan;
use crate::compact_remote_v2::RemoteCompactionV2Output;
use crate::compact_remote_v2::run_remote_compaction_request_v2;
use crate::compact_remote_v2_retention::retained_messages_for_remote_compaction_v2_with_item_cap;
use crate::context_manager::ContextManager;
use crate::context_manager::estimate_response_items_token_count;
use crate::hook_runtime::PostCompactHookOutcome;
use crate::hook_runtime::PreCompactHookOutcome;
use crate::hook_runtime::run_post_compact_hooks;
use crate::hook_runtime::run_pre_compact_hooks;
use crate::responses_metadata::CodexResponsesRequestKind;
use crate::responses_metadata::CompactionTurnMetadata;
use crate::session::session::Session;
use crate::session::turn::built_tools;
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
use codex_protocol::protocol::TurnStartedEvent;
use codex_rollout_trace::CompactionCheckpointTracePayload;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

mod renderer;
pub(crate) use renderer::render_semantic_transcript;

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
        None,
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
    let exact_tail_implementation = ExactTailImplementation::SemanticTranscript;
    let retained_budget = turn_context
        .config
        .compact_exact_tail_semantic_transcript_retained_message_token_budget;
    let mut exact_tail_plan = match prepare_exact_tail_plan(ExactTailPrepareInput {
        sess: sess.as_ref(),
        turn_context: turn_context.as_ref(),
        history_items: &source_history_items,
        base_instructions: &base_instructions,
        policy,
        trigger,
        initial_context_injection,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: retained_budget,
        normalized_tool_output_count,
        implementation: exact_tail_implementation,
    })
    .await
    {
        Ok(Some(plan)) => plan,
        Ok(None) => {
            return Err(CodexErr::Stream(
                "Semantic transcript compaction requires active exact-tail configuration."
                    .to_string(),
                None,
            ));
        }
        Err(error) => {
            let failure_reason = error.reason;
            emit_exact_tail_prepare_failure_diagnostic(
                sess.as_ref(),
                turn_context.as_ref(),
                &compaction_id,
                trigger,
                exact_tail_implementation,
                failure_reason,
            )
            .await;
            return Err(error.into_codex_err());
        }
    };

    let raw_cold_tokens = estimate_response_items_token_count(&exact_tail_plan.plan.cold_history);
    let semantic_item_cap = exact_tail_plan
        .plan
        .diagnostics
        .max_model_visible_item_tokens;
    let transcript =
        match render_semantic_transcript(&exact_tail_plan.plan.cold_history, semantic_item_cap) {
            Ok(transcript) => transcript,
            Err(error) => {
                let failure_reason = error.reason;
                emit_exact_tail_compaction_diagnostic(
                    sess.as_ref(),
                    turn_context.as_ref(),
                    &compaction_id,
                    trigger,
                    &exact_tail_plan.plan,
                    None,
                    Some(failure_reason),
                )
                .await;
                return Err(error.into_codex_err());
            }
        };
    let (retained_messages, retained_images) =
        retained_messages_for_remote_compaction_v2_with_item_cap(
            &exact_tail_plan.plan.cold_history,
            usize::try_from(retained_budget).unwrap_or(usize::MAX),
            usize::try_from(semantic_item_cap).unwrap_or(10_000),
        );
    analytics_details.retained_image_count = Some(retained_images);
    let retained_message_tokens = estimate_response_items_token_count(&retained_messages);
    exact_tail_plan.plan.diagnostics.raw_cold_tokens = Some(raw_cold_tokens);
    exact_tail_plan.plan.diagnostics.semantic_transcript_tokens = Some(transcript.estimated_tokens);
    exact_tail_plan
        .plan
        .diagnostics
        .semantic_transcript_item_count = Some(transcript.items.len());
    exact_tail_plan
        .plan
        .diagnostics
        .semantic_transcript_tool_observation_count = Some(transcript.tool_observation_count);
    exact_tail_plan
        .plan
        .diagnostics
        .semantic_transcript_reduction_tokens =
        Some(raw_cold_tokens.saturating_sub(transcript.estimated_tokens));
    exact_tail_plan
        .plan
        .diagnostics
        .retained_cold_message_tokens = Some(retained_message_tokens);
    exact_tail_plan.plan.diagnostics.retained_cold_message_count = Some(retained_messages.len());

    let mut history = ContextManager::new();
    history.replace(transcript.items);
    if let Err(error) = check_cold_input_fits(
        &exact_tail_plan.plan,
        history.raw_items(),
        turn_context.model_context_window(),
    ) {
        let failure_reason = error.reason;
        emit_exact_tail_compaction_diagnostic(
            sess.as_ref(),
            turn_context.as_ref(),
            &compaction_id,
            trigger,
            &exact_tail_plan.plan,
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
            &exact_tail_plan.plan,
            None,
            Some(ExactTailFailReason::ColdInputTooLarge),
        )
        .await;
        return Err(exact_tail_cold_input_too_large_error(
            request_tokens,
            context_window,
        ));
    }

    let trace_input_history = history.raw_items().to_vec();
    let prompt_input = history.for_prompt(&turn_context.model_info.input_modalities);
    let tool_router = built_tools(
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

    let mut isolated_client_session = sess
        .services
        .model_client
        .clone()
        .with_beta_feature_advertised(Feature::RemoteCompactionV2.key())
        .with_prompt_cache_key_override(Some(format!(
            "semantic-transcript-v2-compaction:{}",
            turn_context.sub_id
        )))
        .new_ephemeral_session();
    let turn_state = client_session.as_ref().map(|session| session.turn_state());
    if let Some(turn_state) = turn_state {
        isolated_client_session = isolated_client_session.with_turn_state(turn_state);
    }
    let compaction_output_result = run_remote_compaction_request_v2(
        sess,
        turn_context,
        &mut isolated_client_session,
        &prompt,
        &responses_metadata,
    )
    .await;
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
            if matches!(error, CodexErr::ContextWindowExceeded) {
                emit_exact_tail_compaction_diagnostic(
                    sess.as_ref(),
                    turn_context.as_ref(),
                    &compaction_id,
                    trigger,
                    &exact_tail_plan.plan,
                    None,
                    Some(ExactTailFailReason::BackendContextExceeded),
                )
                .await;
                return Err(exact_tail_backend_context_exceeded_error());
            }
            return Err(error);
        }
    };
    if let Some(token_usage) = token_usage {
        analytics_details.compaction_summary_tokens = Some(token_usage.output_tokens);
        analytics_details.cached_input_tokens = Some(token_usage.cached_input_tokens);
    }

    if let Err(error) = ensure_replacement_has_cold_summary(
        &exact_tail_plan.plan,
        std::slice::from_ref(&compaction_output),
    ) {
        let failure_reason = error.reason;
        emit_exact_tail_compaction_diagnostic(
            sess.as_ref(),
            turn_context.as_ref(),
            &compaction_id,
            trigger,
            &exact_tail_plan.plan,
            None,
            Some(failure_reason),
        )
        .await;
        return Err(error.into_codex_err());
    }
    let actual_summary_tokens =
        estimate_response_items_token_count(std::slice::from_ref(&compaction_output));
    exact_tail_plan.plan.diagnostics.actual_summary_tokens = Some(actual_summary_tokens);
    analytics_details.compaction_summary_tokens = Some(actual_summary_tokens);

    let mut compacted_history = retained_messages;
    compacted_history.push(compaction_output);
    let attempted_replacement = append_hot_suffix_to_replacement(
        compacted_history.clone(),
        exact_tail_plan.initial_context.clone(),
        exact_tail_plan.plan.hot_suffix.clone(),
        initial_context_injection,
    );
    let attempted_replacement_tokens = estimate_response_items_token_count(&attempted_replacement);
    exact_tail_plan
        .plan
        .diagnostics
        .attempted_replacement_tokens_estimate = Some(attempted_replacement_tokens);
    exact_tail_plan
        .plan
        .diagnostics
        .attempted_final_replacement_tokens_estimate = Some(
        attempted_replacement_tokens.saturating_add(
            exact_tail_plan
                .plan
                .diagnostics
                .final_replacement_extra_budget_tokens,
        ),
    );
    let replacement = match build_exact_tail_replacement(
        &exact_tail_plan,
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
                &exact_tail_plan.plan,
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
        &exact_tail_plan.plan,
        Some(&replacement.diagnostics),
        None,
    )
    .await;
    let new_history = replacement.replacement_history;
    if let Some(client_session) = client_session {
        client_session.reset_responses_continuation();
    }
    sess.services
        .model_client
        .reset_cached_responses_continuation();
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
    sess.recompute_token_usage(turn_context).await;

    sess.emit_turn_item_completed(turn_context, compaction_item)
        .await;
    Ok(())
}

#[cfg(test)]
#[path = "compact_semantic_transcript_tests.rs"]
mod tests;
