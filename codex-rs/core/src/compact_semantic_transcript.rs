use crate::client::ModelClientSession;
use crate::compact::CompactionAnalyticsAttempt;
use crate::compact::CompactionAnalyticsDetails;
use crate::compact::InitialContextInjection;
use crate::compact::compaction_status_from_result;
use crate::compact_exact_tail::append_hot_suffix_to_replacement;
use crate::compact_exact_tail::build_exact_tail_replacement;
use crate::compact_exact_tail::emit_exact_tail_compaction_diagnostic;
use crate::compact_exact_tail::ensure_replacement_has_cold_summary;
use crate::compact_exact_tail::pending_exact_tail_tool_surface_hint;
use crate::compact_model_fallback::record_model_fallback;
use crate::context_manager::estimate_response_items_token_count;
use crate::hook_runtime::PostCompactHookOutcome;
use crate::hook_runtime::PreCompactHookOutcome;
use crate::hook_runtime::run_post_compact_hooks;
use crate::hook_runtime::run_pre_compact_hooks;
use crate::responses_metadata::CompactionTurnMetadata;
use crate::session::session::Session;
use crate::session::step_context::StepContext;
use crate::session::turn_context::TurnContext;
use codex_analytics::CompactionImplementation;
use codex_analytics::CompactionPhase;
use codex_analytics::CompactionReason;
use codex_analytics::CompactionTrigger;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::items::ContextCompactionItem;
use codex_protocol::items::TurnItem;
use codex_protocol::protocol::CompactedItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::TurnStartedEvent;
use codex_rollout_trace::CompactionCheckpointTracePayload;
use std::sync::Arc;

mod renderer;
pub(crate) use renderer::render_semantic_transcript;

#[path = "compact_semantic_transcript_attempt.rs"]
mod attempt;
use attempt::SemanticTranscriptAttempt;
use attempt::run_semantic_transcript_attempt;

pub(crate) async fn run_inline_remote_auto_compact_task(
    sess: Arc<Session>,
    step_context: Arc<StepContext>,
    fallback_step_context: Option<Arc<StepContext>>,
    client_session: &mut ModelClientSession,
    initial_context_injection: InitialContextInjection,
    reason: CompactionReason,
    phase: CompactionPhase,
) -> CodexResult<()> {
    run_remote_compact_task_inner(
        &sess,
        &step_context,
        fallback_step_context.as_ref(),
        Some(client_session),
        initial_context_injection,
        CompactionTurnMetadata::new(
            CompactionTrigger::Auto,
            reason,
            CompactionImplementation::ResponsesCompactionV2,
            phase,
        ),
    )
    .await
}

pub(crate) async fn run_remote_compact_task(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
) -> CodexResult<()> {
    let step_context = sess.capture_step_context(Arc::clone(&turn_context)).await;
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
        &step_context,
        /*fallback_step_context*/ None,
        /*client_session*/ None,
        InitialContextInjection::DoNotInject,
        CompactionTurnMetadata::new(
            CompactionTrigger::Manual,
            CompactionReason::UserRequested,
            CompactionImplementation::ResponsesCompactionV2,
            CompactionPhase::StandaloneTurn,
        ),
    )
    .await
}

async fn run_remote_compact_task_inner(
    sess: &Arc<Session>,
    step_context: &Arc<StepContext>,
    fallback_step_context: Option<&Arc<StepContext>>,
    client_session: Option<&mut ModelClientSession>,
    initial_context_injection: InitialContextInjection,
    compaction_metadata: CompactionTurnMetadata,
) -> CodexResult<()> {
    let turn_context = &step_context.turn;
    let trigger = compaction_metadata.trigger();
    let reason = compaction_metadata.reason();
    let phase = compaction_metadata.phase();
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
        step_context,
        fallback_step_context,
        client_session,
        initial_context_injection,
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
    step_context: &Arc<StepContext>,
    fallback_step_context: Option<&Arc<StepContext>>,
    client_session: Option<&mut ModelClientSession>,
    initial_context_injection: InitialContextInjection,
    compaction_metadata: CompactionTurnMetadata,
    analytics_details: &mut CompactionAnalyticsDetails,
) -> CodexResult<()> {
    let turn_context = &step_context.turn;
    let trigger = compaction_metadata.trigger();
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

    let attempt = run_semantic_transcript_attempt(
        sess,
        step_context,
        client_session.as_deref(),
        &compaction_trace,
        &compaction_id,
        &initial_context_injection,
        compaction_metadata,
        analytics_details,
    )
    .await;
    let (attempt, compaction_turn_context) = match attempt {
        Ok(attempt) => (attempt, turn_context),
        Err(error) => {
            let Some(fallback_step_context) = fallback_step_context else {
                return Err(error);
            };
            if !matches!(&error, CodexErr::InvalidRequest(_)) {
                return Err(error);
            }
            let fallback_turn_context = &fallback_step_context.turn;
            let fallback_compaction_trace =
                sess.services.rollout_thread_trace.compaction_trace_context(
                    fallback_turn_context.sub_id.as_str(),
                    compaction_id.as_str(),
                    fallback_turn_context.model_info.slug.as_str(),
                    fallback_turn_context.provider.info().name.as_str(),
                );
            let fallback_result = run_semantic_transcript_attempt(
                sess,
                fallback_step_context,
                client_session.as_deref(),
                &fallback_compaction_trace,
                &compaction_id,
                &initial_context_injection,
                compaction_metadata,
                analytics_details,
            )
            .await;
            record_model_fallback(
                &sess.services.session_telemetry,
                turn_context.model_info.slug.as_str(),
                fallback_turn_context.model_info.slug.as_str(),
                compaction_metadata.reason(),
                compaction_metadata.implementation(),
                fallback_result.as_ref().err(),
            );
            match fallback_result {
                Ok(attempt) => (attempt, fallback_turn_context),
                Err(_) => return Err(error),
            }
        }
    };
    let SemanticTranscriptAttempt {
        trace_input_history,
        retained_messages,
        compaction_output,
        token_usage,
        mut exact_tail_plan,
    } = attempt;
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
            compaction_turn_context.as_ref(),
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
        &initial_context_injection,
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
        &initial_context_injection,
        actual_summary_tokens,
    ) {
        Ok(replacement) => replacement,
        Err(error) => {
            let failure_reason = error.reason;
            emit_exact_tail_compaction_diagnostic(
                sess.as_ref(),
                compaction_turn_context.as_ref(),
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
        compaction_turn_context.as_ref(),
        &compaction_id,
        trigger,
        &exact_tail_plan.plan,
        Some(&replacement.diagnostics),
        None,
    )
    .await;
    let exact_tail_tool_surface_hint = pending_exact_tail_tool_surface_hint(
        &compaction_id,
        &replacement,
        exact_tail_plan.plan.diagnostics.implementation,
    );
    let new_history = replacement.replacement_history;
    if let Some(client_session) = client_session {
        client_session.reset_responses_continuation();
    }
    sess.services
        .model_client
        .reset_cached_responses_continuation();
    let (new_window_number, new_window_ids) = sess.advance_auto_compact_window().await;

    let (reference_context_item, world_state_baseline) = match &initial_context_injection {
        InitialContextInjection::DoNotInject => (None, None),
        InitialContextInjection::BeforeLastUserMessage(world_state) => (
            Some(compaction_turn_context.to_turn_context_item()),
            Some(Arc::clone(world_state)),
        ),
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
        compaction_turn_context.as_ref(),
        new_history,
        reference_context_item,
        world_state_baseline,
        compacted_item,
    )
    .await;
    sess.set_active_exact_tail_tool_surface_hint(exact_tail_tool_surface_hint)
        .await;
    sess.recompute_token_usage(compaction_turn_context).await;

    sess.emit_turn_item_completed(compaction_turn_context, compaction_item)
        .await;
    Ok(())
}

#[cfg(test)]
#[path = "compact_semantic_transcript_tests.rs"]
mod tests;
