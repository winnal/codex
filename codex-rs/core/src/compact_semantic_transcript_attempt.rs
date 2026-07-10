use std::sync::Arc;

use super::render_semantic_transcript;
use crate::Prompt;
use crate::client::ModelClientSession;
use crate::compact::CompactionAnalyticsDetails;
use crate::compact::InitialContextInjection;
use crate::compact_exact_tail::CompactionHistoryPolicy;
use crate::compact_exact_tail::ExactTailFailReason;
use crate::compact_exact_tail::ExactTailImplementation;
use crate::compact_exact_tail::ExactTailPrepareInput;
use crate::compact_exact_tail::PreparedExactTailPlan;
use crate::compact_exact_tail::check_cold_input_fits;
use crate::compact_exact_tail::emit_exact_tail_compaction_diagnostic;
use crate::compact_exact_tail::emit_exact_tail_prepare_failure_diagnostic;
use crate::compact_exact_tail::exact_tail_backend_context_exceeded_error;
use crate::compact_exact_tail::exact_tail_cold_input_too_large_error;
use crate::compact_exact_tail::normalize_tool_outputs_for_exact_tail_policy;
use crate::compact_exact_tail::prepare_exact_tail_plan;
use crate::compact_remote_v2::RemoteCompactionV2Output;
use crate::compact_remote_v2::run_remote_compaction_request_v2;
use crate::compact_remote_v2_retention::retained_messages_for_remote_compaction_v2_with_item_cap;
use crate::context_manager::ContextManager;
use crate::context_manager::estimate_response_items_token_count;
use crate::responses_metadata::CodexResponsesRequestKind;
use crate::responses_metadata::CompactionTurnMetadata;
use crate::session::session::Session;
use crate::session::step_context::StepContext;
use crate::session::turn::built_tools_without_exact_tail_tool_surface_hint;
use codex_features::Feature;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use codex_rollout_trace::CompactionTraceContext;
use tokio_util::sync::CancellationToken;

pub(super) struct SemanticTranscriptAttempt {
    pub(super) trace_input_history: Vec<ResponseItem>,
    pub(super) retained_messages: Vec<ResponseItem>,
    pub(super) compaction_output: ResponseItem,
    pub(super) token_usage: Option<TokenUsage>,
    pub(super) exact_tail_plan: PreparedExactTailPlan,
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn run_semantic_transcript_attempt(
    sess: &Arc<Session>,
    step_context: &Arc<StepContext>,
    client_session: Option<&ModelClientSession>,
    compaction_trace: &CompactionTraceContext,
    compaction_id: &str,
    initial_context_injection: &InitialContextInjection,
    compaction_metadata: CompactionTurnMetadata,
    analytics_details: &mut CompactionAnalyticsDetails,
) -> CodexResult<SemanticTranscriptAttempt> {
    let turn_context = &step_context.turn;
    let trigger = compaction_metadata.trigger();
    let mut source_history = sess.clone_history().await;
    let base_instructions = sess.get_base_instructions().await;
    let policy = CompactionHistoryPolicy::from_config(&turn_context.config);
    let normalized_tool_output_count = normalize_tool_outputs_for_exact_tail_policy(
        &mut source_history,
        policy,
        turn_context.model_info.truncation_policy.into(),
    );
    let source_history_items = source_history.raw_items().to_vec();
    let implementation = ExactTailImplementation::SemanticTranscript;
    let retained_budget = turn_context
        .config
        .compact_exact_tail_semantic_transcript_retained_message_token_budget;
    let mut exact_tail_plan = match prepare_exact_tail_plan(ExactTailPrepareInput {
        sess,
        turn_context,
        history_items: &source_history_items,
        base_instructions: &base_instructions,
        policy,
        trigger,
        initial_context_injection,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: retained_budget,
        normalized_tool_output_count,
        implementation,
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
            emit_exact_tail_prepare_failure_diagnostic(
                sess.as_ref(),
                turn_context.as_ref(),
                compaction_id,
                trigger,
                implementation,
                &error,
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
                emit_exact_tail_compaction_diagnostic(
                    sess.as_ref(),
                    turn_context.as_ref(),
                    compaction_id,
                    trigger,
                    &exact_tail_plan.plan,
                    None,
                    Some(error.reason),
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
        emit_exact_tail_compaction_diagnostic(
            sess.as_ref(),
            turn_context.as_ref(),
            compaction_id,
            trigger,
            &exact_tail_plan.plan,
            None,
            Some(error.reason),
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
            compaction_id,
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
    let tool_router = built_tools_without_exact_tail_tool_surface_hint(
        sess.as_ref(),
        step_context.as_ref(),
        &CancellationToken::new(),
    )
    .await?;
    let mut input = prompt_input;
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
    if let Some(turn_state) = client_session.map(ModelClientSession::turn_state) {
        isolated_client_session = isolated_client_session.with_turn_state(turn_state);
    }
    let result = run_remote_compaction_request_v2(
        sess,
        turn_context,
        &mut isolated_client_session,
        &prompt,
        &responses_metadata,
    )
    .await;
    trace_attempt.record_result(
        result
            .as_ref()
            .map(|output| std::slice::from_ref(&output.compaction_output)),
    );
    let RemoteCompactionV2Output {
        compaction_output,
        token_usage,
    } = match result {
        Ok(output) => output,
        Err(CodexErr::ContextWindowExceeded) => {
            emit_exact_tail_compaction_diagnostic(
                sess.as_ref(),
                turn_context.as_ref(),
                compaction_id,
                trigger,
                &exact_tail_plan.plan,
                None,
                Some(ExactTailFailReason::BackendContextExceeded),
            )
            .await;
            return Err(exact_tail_backend_context_exceeded_error());
        }
        Err(error) => return Err(error),
    };

    Ok(SemanticTranscriptAttempt {
        trace_input_history,
        retained_messages,
        compaction_output,
        token_usage,
        exact_tail_plan,
    })
}
