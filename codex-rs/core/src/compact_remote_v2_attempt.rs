use std::sync::Arc;

use super::RemoteCompactionV2Output;
use super::run_remote_compaction_request_v2;
use crate::Prompt;
use crate::client::ModelClientSession;
use crate::compact::CompactionAnalyticsDetails;
use crate::compact::InitialContextInjection;
use crate::compact_exact_tail::CompactionHistoryPolicy;
use crate::compact_exact_tail::ExactTailCompactInputExpectation;
use crate::compact_exact_tail::ExactTailImplementation;
use crate::compact_exact_tail::ExactTailPrepareInput;
use crate::compact_exact_tail::PreparedExactTailPlan;
use crate::compact_exact_tail::check_cold_input_fits;
use crate::compact_exact_tail::emit_exact_tail_compaction_diagnostic;
use crate::compact_exact_tail::emit_exact_tail_prepare_failure_diagnostic;
use crate::compact_exact_tail::exact_tail_backend_context_exceeded_error;
use crate::compact_exact_tail::normalize_tool_outputs_for_exact_tail_policy;
use crate::compact_exact_tail::prepare_exact_tail_plan;
use crate::compact_remote::trim_function_call_history_to_fit_context_window;
use crate::compact_remote_v2_retention::REMOTE_COMPACTION_V2_RETAINED_MESSAGE_TOKEN_BUDGET;
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
use tracing::info;

pub(super) struct RemoteCompactV2Attempt {
    pub(super) trace_input_history: Vec<ResponseItem>,
    pub(super) prompt_input: Vec<ResponseItem>,
    pub(super) compaction_output: ResponseItem,
    pub(super) token_usage: Option<TokenUsage>,
    pub(super) exact_tail_plan: Option<PreparedExactTailPlan>,
    /// Keeps a session created for standalone compaction alive through lifecycle completion.
    pub(super) owned_client_session: Option<ModelClientSession>,
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn run_remote_compact_v2_attempt(
    sess: &Arc<Session>,
    step_context: &Arc<StepContext>,
    client_session: Option<&mut ModelClientSession>,
    compaction_trace: &CompactionTraceContext,
    compaction_id: &str,
    initial_context_injection: &InitialContextInjection,
    compaction_metadata: CompactionTurnMetadata,
    analytics_details: &mut CompactionAnalyticsDetails,
) -> CodexResult<RemoteCompactV2Attempt> {
    let turn_context = &step_context.turn;
    let mut history = sess.clone_history().await;
    let base_instructions = sess.get_base_instructions().await;
    let policy = CompactionHistoryPolicy::from_config(&turn_context.config);
    let normalized_tool_output_count = normalize_tool_outputs_for_exact_tail_policy(
        &mut history,
        policy,
        turn_context.model_info.truncation_policy.into(),
    );
    let source_history_items = history.raw_items().to_vec();
    let source_history_item_provenance = history.item_provenance().to_vec();
    let exact_tail_implementation = ExactTailImplementation::RemoteV2;
    let exact_tail_plan = match prepare_exact_tail_plan(ExactTailPrepareInput {
        sess,
        turn_context,
        history_items: &source_history_items,
        history_item_provenance: &source_history_item_provenance,
        base_instructions: &base_instructions,
        policy,
        trigger: compaction_metadata.trigger(),
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
                compaction_id,
                compaction_metadata.trigger(),
                exact_tail_implementation,
                &error,
            )
            .await;
            return Err(error.into_codex_err());
        }
    };

    if let Some(prepared) = &exact_tail_plan {
        history.replace(prepared.plan.cold_history.clone());
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
    let prompt_input = if exact_tail_plan.is_some() {
        history.for_exact_tail_prompt(&turn_context.model_info.input_modalities)
    } else {
        history.for_prompt(&turn_context.model_info.input_modalities)
    };
    let compaction_trigger = ResponseItem::CompactionTrigger {};
    let mut input = prompt_input.clone();
    input.push(compaction_trigger.clone());
    if let Some(prepared) = &exact_tail_plan
        && let Err(error) = check_cold_input_fits(
            &prepared.plan,
            &input,
            ExactTailCompactInputExpectation::Source {
                derived_items: std::slice::from_ref(&compaction_trigger),
            },
            prepared.model_context_window,
            &base_instructions,
        )
    {
        emit_exact_tail_compaction_diagnostic(
            sess.as_ref(),
            turn_context.as_ref(),
            compaction_id,
            compaction_metadata.trigger(),
            &prepared.plan,
            None,
            Some(error.reason),
        )
        .await;
        return Err(error.into_codex_err());
    }
    let tool_router = built_tools_without_exact_tail_tool_surface_hint(
        sess.as_ref(),
        step_context.as_ref(),
        &CancellationToken::new(),
    )
    .await?;
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
    let turn_state = client_session.as_ref().map(|session| session.turn_state());
    let mut owned_client_session = None;
    let client_session = if exact_tail_plan.is_some() {
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
        if let Some(turn_state) = turn_state {
            isolated_client_session = isolated_client_session.with_turn_state(turn_state);
        }
        owned_client_session.insert(isolated_client_session)
    } else {
        match client_session {
            Some(client_session) => client_session,
            None => owned_client_session.insert(sess.services.model_client.new_session()),
        }
    };
    let compaction_output_result = run_remote_compaction_request_v2(
        sess,
        turn_context.as_ref(),
        client_session,
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
            if matches!(&error, CodexErr::ContextWindowExceeded)
                && let Some(prepared) = &exact_tail_plan
            {
                emit_exact_tail_compaction_diagnostic(
                    sess.as_ref(),
                    turn_context.as_ref(),
                    compaction_id,
                    compaction_metadata.trigger(),
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
    Ok(RemoteCompactV2Attempt {
        trace_input_history,
        prompt_input,
        compaction_output,
        token_usage,
        exact_tail_plan,
        owned_client_session,
    })
}
