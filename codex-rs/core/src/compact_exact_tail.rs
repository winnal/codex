use crate::compact::InitialContextInjection;
use crate::compact::SUMMARY_PREFIX;
use crate::compact::build_compacted_history;
use crate::compact::is_summary_message;
use crate::context_manager::ContextManager;
use crate::context_manager::estimate_response_items_token_count;
use crate::context_manager::is_user_turn_boundary;
use crate::session::session::Session;
use crate::session::turn_context::TurnContext;
use codex_analytics::CompactionTrigger;
use codex_protocol::error::CodexErr;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::plaintext_agent_message_content;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::ExactTailCompactionDiagnosticEvent;
use codex_protocol::protocol::ExactTailCompactionFitResult;
use codex_protocol::protocol::ExactTailCompactionRoute;
use codex_protocol::protocol::ExactTailCompactionTrigger;
use codex_utils_output_truncation::TruncationPolicy;
use codex_utils_output_truncation::approx_token_count;
use tracing::warn;

mod groups;
mod planner;
mod types;

pub(crate) use planner::classify_exact_tail_history_item;
pub(crate) use planner::exact_tail_replacement_budget;
pub(crate) use planner::plan_exact_tail;
pub(crate) use types::*;

pub(crate) fn normalize_tool_outputs_for_exact_tail_policy(
    history: &mut ContextManager,
    policy: CompactionHistoryPolicy,
    truncation_policy: TruncationPolicy,
) -> usize {
    match policy {
        CompactionHistoryPolicy::PreserveRecentExact { .. } => {
            history.normalize_tool_outputs_to_policy(truncation_policy)
        }
        CompactionHistoryPolicy::Standard => 0,
    }
}

pub(crate) async fn prepare_exact_tail_plan(
    input: ExactTailPrepareInput<'_>,
) -> Result<Option<PreparedExactTailPlan>, ExactTailError> {
    let ExactTailPrepareInput {
        sess,
        turn_context,
        history_items,
        base_instructions,
        policy,
        trigger,
        initial_context_injection,
        estimated_summary_scaffold_overhead_tokens,
        retained_cold_user_message_budget_tokens,
        normalized_tool_output_count,
        implementation,
    } = input;

    let CompactionHistoryPolicy::PreserveRecentExact {
        target_tokens,
        max_model_visible_item_tokens,
    } = policy
    else {
        return Ok(None);
    };

    let current_context = match initial_context_injection {
        InitialContextInjection::BeforeLastUserMessage => {
            sess.build_initial_context(turn_context).await
        }
        InitialContextInjection::DoNotInject => {
            sess.build_initial_context_without_side_effects(turn_context)
                .await
        }
    };
    ensure_model_visible_items_within_limit(&current_context, max_model_visible_item_tokens)?;
    let initial_context = match initial_context_injection {
        InitialContextInjection::BeforeLastUserMessage => current_context.clone(),
        InitialContextInjection::DoNotInject => Vec::new(),
    };
    let base_instruction_tokens =
        i64::try_from(approx_token_count(&base_instructions.text)).unwrap_or(i64::MAX);
    let current_context_tokens = estimate_response_items_token_count(&current_context);
    let required_current_context_budget =
        base_instruction_tokens.saturating_add(current_context_tokens);
    let final_replacement_extra_budget_tokens = match initial_context_injection {
        InitialContextInjection::DoNotInject => required_current_context_budget,
        InitialContextInjection::BeforeLastUserMessage => base_instruction_tokens,
    };
    let mut plan = plan_exact_tail(ExactTailPlanInput {
        history_items,
        target_tokens,
        effective_replacement_budget: exact_tail_replacement_budget(
            turn_context.model_context_window(),
            turn_context.config.model_auto_compact_token_limit,
            turn_context.config.model_auto_compact_token_limit_scope,
            trigger,
        ),
        required_current_context_budget,
        final_replacement_extra_budget_tokens,
        max_model_visible_item_tokens,
        estimated_summary_scaffold_overhead_tokens,
        retained_cold_user_message_budget_tokens,
        implementation,
    })?;
    plan.diagnostics.normalized_tool_output_count = normalized_tool_output_count;

    Ok(Some(PreparedExactTailPlan {
        plan,
        initial_context,
    }))
}

pub(crate) fn local_summary_scaffold_overhead_tokens() -> i64 {
    let minimal_summary = format!("{SUMMARY_PREFIX}\n");
    let scaffold = build_compacted_history(Vec::new(), &[], &minimal_summary);
    estimate_response_items_token_count(&scaffold)
}

pub(crate) fn remote_legacy_summary_scaffold_overhead_tokens() -> i64 {
    // The legacy remote compact endpoint returns the replacement history directly. The client only
    // filters server-returned items and appends the exact tail, so there is no client-side summary
    // wrapper to reserve beyond the conservative summary budget itself.
    0
}

pub(crate) fn ensure_non_empty_local_summary(
    plan: &ExactTailPlan,
    summary_suffix: &str,
) -> Result<(), ExactTailError> {
    if !summary_suffix.trim().is_empty() {
        return Ok(());
    }
    warn!(
        exact_tail_enabled = true,
        implementation = plan.diagnostics.implementation.as_str(),
        exact_tail_fail_reason = ExactTailFailReason::NoUsableColdSummary.as_str(),
        "exact-tail compaction returned no usable cold summary"
    );
    Err(ExactTailError::new(
        ExactTailFailReason::NoUsableColdSummary,
        "Exact-tail compaction returned no usable cold summary for the cold prefix.",
    ))
}

pub(crate) fn ensure_replacement_has_cold_summary(
    plan: &ExactTailPlan,
    compacted_history: &[ResponseItem],
) -> Result<(), ExactTailError> {
    if compacted_history
        .iter()
        .any(response_item_has_usable_cold_summary)
    {
        return Ok(());
    }
    warn!(
        exact_tail_enabled = true,
        implementation = plan.diagnostics.implementation.as_str(),
        exact_tail_fail_reason = ExactTailFailReason::NoUsableColdSummary.as_str(),
        "exact-tail compaction replacement contains no usable cold summary"
    );
    Err(ExactTailError::new(
        ExactTailFailReason::NoUsableColdSummary,
        "Exact-tail compaction returned no usable cold summary for the cold prefix.",
    ))
}

fn response_item_has_usable_cold_summary(item: &ResponseItem) -> bool {
    match item {
        ResponseItem::Message { role, content, .. } if role == "assistant" => {
            content.iter().any(content_item_has_non_empty_text)
        }
        ResponseItem::Message { role, content, .. } if role == "user" => {
            content.iter().any(content_item_is_summary_text)
        }
        ResponseItem::Message { .. } => false,
        ResponseItem::AgentMessage { content, .. } => {
            plaintext_agent_message_content(content).is_some()
        }
        ResponseItem::Compaction {
            encrypted_content, ..
        } => !encrypted_content.trim().is_empty(),
        ResponseItem::ContextCompaction {
            encrypted_content, ..
        } => encrypted_content
            .as_deref()
            .is_some_and(|content| !content.trim().is_empty()),
        ResponseItem::Reasoning { .. }
        | ResponseItem::LocalShellCall { .. }
        | ResponseItem::FunctionCall { .. }
        | ResponseItem::ToolSearchCall { .. }
        | ResponseItem::FunctionCallOutput { .. }
        | ResponseItem::ToolSearchOutput { .. }
        | ResponseItem::CustomToolCall { .. }
        | ResponseItem::CustomToolCallOutput { .. }
        | ResponseItem::WebSearchCall { .. }
        | ResponseItem::ImageGenerationCall { .. }
        | ResponseItem::CompactionTrigger { .. }
        | ResponseItem::Other => false,
    }
}

fn content_item_is_summary_text(item: &ContentItem) -> bool {
    match item {
        ContentItem::InputText { text } | ContentItem::OutputText { text } => {
            is_summary_message(text.trim_start())
        }
        ContentItem::InputImage { .. } => false,
    }
}

fn content_item_has_non_empty_text(item: &ContentItem) -> bool {
    match item {
        ContentItem::InputText { text } | ContentItem::OutputText { text } => {
            !text.trim().is_empty()
        }
        ContentItem::InputImage { .. } => false,
    }
}

pub(crate) fn exact_tail_cold_input_too_large_error(
    request_tokens: i64,
    context_window: i64,
) -> CodexErr {
    CodexErr::Stream(
        format!(
            "{}: exact-tail compaction request estimates to {request_tokens} tokens, exceeding context window {context_window}; pruning cold input would lose coverage",
            ExactTailFailReason::ColdInputTooLarge.as_str()
        ),
        None,
    )
}

pub(crate) fn exact_tail_backend_context_exceeded_error() -> CodexErr {
    CodexErr::Stream(
        format!(
            "{}: exact-tail compaction request exceeded the backend context window and cannot be pruned without losing cold coverage.",
            ExactTailFailReason::BackendContextExceeded.as_str()
        ),
        None,
    )
}

pub(crate) fn append_hot_suffix_to_replacement(
    mut compacted_history: Vec<ResponseItem>,
    initial_context: Vec<ResponseItem>,
    hot_suffix: Vec<ResponseItem>,
    initial_context_injection: InitialContextInjection,
) -> Vec<ResponseItem> {
    match initial_context_injection {
        InitialContextInjection::DoNotInject => {
            compacted_history.extend(hot_suffix);
            compacted_history
        }
        InitialContextInjection::BeforeLastUserMessage if hot_suffix.is_empty() => {
            insert_exact_tail_initial_context(compacted_history, initial_context)
        }
        InitialContextInjection::BeforeLastUserMessage => {
            compacted_history.extend(initial_context);
            compacted_history.extend(hot_suffix);
            compacted_history
        }
    }
}

fn insert_exact_tail_initial_context(
    mut replacement_history: Vec<ResponseItem>,
    initial_context: Vec<ResponseItem>,
) -> Vec<ResponseItem> {
    let mut last_summary_index = None;
    let mut last_boundary_index = None;
    for (index, item) in replacement_history.iter().enumerate().rev() {
        if response_item_is_summary_user_message(item) {
            last_summary_index.get_or_insert(index);
            continue;
        }
        if is_user_turn_boundary(item) {
            last_boundary_index = Some(index);
            break;
        }
    }
    let last_compaction_index =
        replacement_history
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, item)| {
                matches!(
                    item,
                    ResponseItem::Compaction { .. } | ResponseItem::ContextCompaction { .. }
                )
                .then_some(index)
            });
    let insertion_index = last_boundary_index
        .or(last_summary_index)
        .or(last_compaction_index);

    if let Some(insertion_index) = insertion_index {
        replacement_history.splice(insertion_index..insertion_index, initial_context);
    } else {
        replacement_history.extend(initial_context);
    }

    replacement_history
}

fn response_item_is_summary_user_message(item: &ResponseItem) -> bool {
    let ResponseItem::Message { role, content, .. } = item else {
        return false;
    };
    role == "user" && content.iter().any(content_item_is_summary_text)
}

pub(crate) fn build_exact_tail_replacement(
    prepared: &PreparedExactTailPlan,
    compacted_history: Vec<ResponseItem>,
    initial_context_injection: InitialContextInjection,
    actual_summary_tokens: i64,
) -> Result<ExactTailReplacement, ExactTailError> {
    let replacement_history = append_hot_suffix_to_replacement(
        compacted_history,
        prepared.initial_context.clone(),
        prepared.plan.hot_suffix.clone(),
        initial_context_injection,
    );
    let final_replacement_tokens_estimate =
        check_replacement_fits(&prepared.plan, &replacement_history, actual_summary_tokens)?;
    let hot_suffix_proof = verify_exact_hot_suffix_preserved(prepared, &replacement_history)?;
    Ok(ExactTailReplacement {
        diagnostics: ExactTailReplacementDiagnostics {
            actual_summary_tokens,
            replacement_tokens_estimate: estimate_response_items_token_count(&replacement_history),
            final_replacement_tokens_estimate,
            hot_suffix_exact_match: hot_suffix_proof.exact_match,
            planned_hot_suffix_item_count: hot_suffix_proof.planned_item_count,
            installed_hot_suffix_item_count: hot_suffix_proof.installed_item_count,
        },
        replacement_history,
    })
}

pub(crate) fn verify_exact_hot_suffix_preserved(
    prepared: &PreparedExactTailPlan,
    replacement_history: &[ResponseItem],
) -> Result<ExactTailHotSuffixProof, ExactTailError> {
    let planned_hot_suffix = prepared.plan.hot_suffix.as_slice();
    if planned_hot_suffix.is_empty() {
        return Ok(ExactTailHotSuffixProof {
            exact_match: true,
            planned_item_count: 0,
            installed_item_count: 0,
        });
    }
    if replacement_history.ends_with(planned_hot_suffix) {
        return Ok(ExactTailHotSuffixProof {
            exact_match: true,
            planned_item_count: planned_hot_suffix.len(),
            installed_item_count: planned_hot_suffix.len(),
        });
    }

    Err(ExactTailError::new(
        ExactTailFailReason::HotSuffixMismatch,
        "Installed exact-tail replacement does not preserve the planned hot suffix exactly.",
    ))
}

pub(crate) async fn emit_exact_tail_compaction_diagnostic(
    sess: &Session,
    turn_context: &TurnContext,
    compaction_id: &str,
    trigger: CompactionTrigger,
    plan: &ExactTailPlan,
    replacement: Option<&ExactTailReplacementDiagnostics>,
    failure_reason: Option<ExactTailFailReason>,
) {
    let diagnostics = &plan.diagnostics;
    let mut event = exact_tail_diagnostic_event(
        sess,
        turn_context,
        compaction_id,
        diagnostic_route(diagnostics.implementation),
        trigger,
        if failure_reason.is_some() {
            ExactTailCompactionFitResult::Failure
        } else {
            ExactTailCompactionFitResult::Success
        },
        failure_reason,
    );
    event.requested_hot_tokens = Some(diagnostics.requested_hot_tokens);
    event.actual_hot_tokens = Some(diagnostics.actual_hot_tokens);
    event.hot_group_count = Some(diagnostics.hot_group_count);
    event.cold_group_count = Some(diagnostics.cold_group_count);
    event.raw_item_count = Some(diagnostics.raw_item_count);
    event.group_count = Some(diagnostics.group_count);
    event.cold_tokens = Some(diagnostics.cold_tokens);
    event.summary_tokens = replacement
        .map(|replacement| replacement.actual_summary_tokens)
        .or(diagnostics.actual_summary_tokens);
    event.replacement_tokens_estimate = replacement
        .map(|replacement| replacement.replacement_tokens_estimate)
        .or(diagnostics.attempted_replacement_tokens_estimate);
    event.final_replacement_tokens_estimate = replacement
        .map(|replacement| replacement.final_replacement_tokens_estimate)
        .or(diagnostics.attempted_final_replacement_tokens_estimate);
    event.effective_replacement_budget = Some(diagnostics.effective_replacement_budget);
    event.safety_margin = Some(diagnostics.safety_margin);
    event.max_model_visible_item_tokens = Some(diagnostics.max_model_visible_item_tokens);
    event.normalized_tool_output_count = Some(diagnostics.normalized_tool_output_count);
    event.largest_hot_item_tokens = Some(diagnostics.largest_hot_item_tokens);
    event.hot_suffix_exact_match =
        replacement.map(|replacement| replacement.hot_suffix_exact_match);
    event.planned_hot_suffix_item_count =
        replacement.map(|replacement| replacement.planned_hot_suffix_item_count);
    event.installed_hot_suffix_item_count =
        replacement.map(|replacement| replacement.installed_hot_suffix_item_count);
    event.filtered_stale_group_count = Some(plan.coverage.filtered_stale_groups.len());
    event.filtered_context_item_count = Some(diagnostics.filtered_context_item_count);
    event.required_current_context_budget = Some(diagnostics.required_current_context_budget);
    event.final_replacement_extra_budget_tokens =
        Some(diagnostics.final_replacement_extra_budget_tokens);
    event.conservative_cold_summary_budget = Some(diagnostics.conservative_cold_summary_budget);
    event.conservative_summary_budget_tokens = Some(diagnostics.conservative_summary_budget_tokens);
    event.estimated_summary_scaffold_overhead_tokens =
        Some(diagnostics.estimated_summary_scaffold_overhead_tokens);
    event.retained_cold_user_message_budget_tokens =
        Some(diagnostics.retained_cold_user_message_budget_tokens);
    event.replacement_overhead_margin_tokens = Some(diagnostics.replacement_overhead_margin_tokens);
    event.available_for_hot = Some(diagnostics.available_for_hot);
    event.post_summary_cold_reserve_target_tokens =
        Some(diagnostics.post_summary_cold_reserve_target_tokens);
    event.post_summary_cold_reserve_tokens = Some(diagnostics.post_summary_cold_reserve_tokens);
    event.post_summary_cold_reserve_group_count =
        Some(diagnostics.post_summary_cold_reserve_group_count);
    event.semantic_transcript_tokens = diagnostics.semantic_transcript_tokens;
    event.semantic_transcript_item_count = diagnostics.semantic_transcript_item_count;
    event.semantic_transcript_tool_observation_count =
        diagnostics.semantic_transcript_tool_observation_count;
    event.raw_cold_tokens = diagnostics.raw_cold_tokens;
    event.semantic_transcript_reduction_tokens = diagnostics.semantic_transcript_reduction_tokens;
    event.retained_cold_message_tokens = diagnostics.retained_cold_message_tokens;
    event.retained_cold_message_count = diagnostics.retained_cold_message_count;
    sess.send_event(turn_context, EventMsg::ExactTailCompactionDiagnostic(event))
        .await;
}

pub(crate) async fn emit_exact_tail_prepare_failure_diagnostic(
    sess: &Session,
    turn_context: &TurnContext,
    compaction_id: &str,
    trigger: CompactionTrigger,
    implementation: ExactTailImplementation,
    failure_reason: ExactTailFailReason,
) {
    let event = exact_tail_diagnostic_event(
        sess,
        turn_context,
        compaction_id,
        diagnostic_route(implementation),
        trigger,
        ExactTailCompactionFitResult::Failure,
        Some(failure_reason),
    );
    sess.send_event(turn_context, EventMsg::ExactTailCompactionDiagnostic(event))
        .await;
}

fn exact_tail_diagnostic_event(
    sess: &Session,
    turn_context: &TurnContext,
    compaction_id: &str,
    route: ExactTailCompactionRoute,
    trigger: CompactionTrigger,
    fit_result: ExactTailCompactionFitResult,
    failure_reason: Option<ExactTailFailReason>,
) -> ExactTailCompactionDiagnosticEvent {
    ExactTailCompactionDiagnosticEvent {
        thread_id: sess.thread_id().to_string(),
        turn_id: turn_context.sub_id.clone(),
        compaction_id: compaction_id.to_string(),
        route,
        trigger: diagnostic_trigger(trigger),
        fit_result,
        failure_reason: failure_reason.map(|reason| reason.as_str().to_string()),
        requested_hot_tokens: None,
        actual_hot_tokens: None,
        hot_group_count: None,
        cold_group_count: None,
        raw_item_count: None,
        group_count: None,
        cold_tokens: None,
        summary_tokens: None,
        replacement_tokens_estimate: None,
        final_replacement_tokens_estimate: None,
        effective_replacement_budget: None,
        safety_margin: None,
        max_model_visible_item_tokens: None,
        normalized_tool_output_count: None,
        largest_hot_item_tokens: None,
        hot_suffix_exact_match: None,
        planned_hot_suffix_item_count: None,
        installed_hot_suffix_item_count: None,
        filtered_stale_group_count: None,
        filtered_context_item_count: None,
        required_current_context_budget: None,
        final_replacement_extra_budget_tokens: None,
        conservative_cold_summary_budget: None,
        conservative_summary_budget_tokens: None,
        estimated_summary_scaffold_overhead_tokens: None,
        retained_cold_user_message_budget_tokens: None,
        replacement_overhead_margin_tokens: None,
        available_for_hot: None,
        post_summary_cold_reserve_target_tokens: None,
        post_summary_cold_reserve_tokens: None,
        post_summary_cold_reserve_group_count: None,
        semantic_transcript_tokens: None,
        semantic_transcript_item_count: None,
        semantic_transcript_tool_observation_count: None,
        raw_cold_tokens: None,
        semantic_transcript_reduction_tokens: None,
        retained_cold_message_tokens: None,
        retained_cold_message_count: None,
    }
}

fn diagnostic_route(implementation: ExactTailImplementation) -> ExactTailCompactionRoute {
    match implementation {
        ExactTailImplementation::Local => ExactTailCompactionRoute::Local,
        ExactTailImplementation::RemoteLegacy => ExactTailCompactionRoute::RemoteLegacy,
        ExactTailImplementation::RemoteV2 => ExactTailCompactionRoute::RemoteV2,
        ExactTailImplementation::SemanticTranscript => ExactTailCompactionRoute::SemanticTranscript,
    }
}

fn diagnostic_trigger(trigger: CompactionTrigger) -> ExactTailCompactionTrigger {
    match trigger {
        CompactionTrigger::Manual => ExactTailCompactionTrigger::Manual,
        CompactionTrigger::Auto => ExactTailCompactionTrigger::Auto,
    }
}

pub(crate) fn check_replacement_fits(
    plan: &ExactTailPlan,
    replacement_history: &[ResponseItem],
    actual_summary_tokens: i64,
) -> Result<i64, ExactTailError> {
    ensure_model_visible_items_within_limit(
        replacement_history,
        plan.diagnostics.max_model_visible_item_tokens,
    )?;
    let replacement_tokens_estimate = estimate_response_items_token_count(replacement_history);
    let final_tokens_estimate = replacement_tokens_estimate
        .saturating_add(plan.diagnostics.final_replacement_extra_budget_tokens);
    let fits = final_tokens_estimate <= plan.diagnostics.effective_replacement_budget;
    tracing::info!(
        exact_tail_enabled = true,
        implementation = plan.diagnostics.implementation.as_str(),
        requested_hot_tokens = plan.diagnostics.requested_hot_tokens,
        actual_hot_tokens = plan.diagnostics.actual_hot_tokens,
        cold_tokens = plan.diagnostics.cold_tokens,
        replacement_tokens_estimate,
        final_replacement_tokens_estimate = final_tokens_estimate,
        conservative_cold_summary_budget = plan.diagnostics.conservative_cold_summary_budget,
        estimated_summary_scaffold_overhead_tokens =
            plan.diagnostics.estimated_summary_scaffold_overhead_tokens,
        retained_cold_user_message_budget_tokens =
            plan.diagnostics.retained_cold_user_message_budget_tokens,
        replacement_overhead_margin_tokens = plan.diagnostics.replacement_overhead_margin_tokens,
        required_current_context_budget = plan.diagnostics.required_current_context_budget,
        final_replacement_extra_budget_tokens =
            plan.diagnostics.final_replacement_extra_budget_tokens,
        max_model_visible_item_tokens = plan.diagnostics.max_model_visible_item_tokens,
        safety_margin = plan.diagnostics.safety_margin,
        available_for_hot = plan.diagnostics.available_for_hot,
        actual_summary_tokens,
        post_summary_fit_result = fits,
        hot_group_count = plan.diagnostics.hot_group_count,
        cold_group_count = plan.diagnostics.cold_group_count,
        filtered_context_item_count = plan.diagnostics.filtered_context_item_count,
        compaction_input_pruning_attempted = false,
        coverage_cold_group_count = plan.coverage.cold_covered_groups.len(),
        coverage_hot_group_count = plan.coverage.hot_exact_groups.len(),
        coverage_filtered_stale_group_count = plan.coverage.filtered_stale_groups.len(),
        "exact-tail replacement fit checked"
    );
    if fits {
        Ok(final_tokens_estimate)
    } else {
        Err(ExactTailError::new(
            ExactTailFailReason::ReplacementTooLarge,
            format!(
                "{}: replacement history plus reserved prompt overhead estimates to {} tokens, exceeding budget {}",
                ExactTailFailReason::ReplacementTooLarge.as_str(),
                final_tokens_estimate,
                plan.diagnostics.effective_replacement_budget
            ),
        ))
    }
}

pub(crate) fn check_cold_input_fits(
    plan: &ExactTailPlan,
    compact_request_items: &[ResponseItem],
    context_window: Option<i64>,
) -> Result<(), ExactTailError> {
    ensure_model_visible_items_within_limit(
        compact_request_items,
        plan.diagnostics.max_model_visible_item_tokens,
    )?;
    let Some(context_window) = context_window else {
        return Ok(());
    };
    let request_tokens = estimate_response_items_token_count(compact_request_items);
    if request_tokens <= context_window {
        return Ok(());
    }
    warn!(
        exact_tail_enabled = true,
        implementation = plan.diagnostics.implementation.as_str(),
        exact_tail_fail_reason = ExactTailFailReason::ColdInputTooLarge.as_str(),
        request_tokens,
        context_window,
        cold_group_count = plan.diagnostics.cold_group_count,
        "exact-tail cold input exceeds context window"
    );
    Err(ExactTailError::new(
        ExactTailFailReason::ColdInputTooLarge,
        "Exact-tail compaction could not safely summarize the cold prefix without losing coverage. Try a smaller compact_preserve_recent_tokens, compact earlier, or disable exact-tail.",
    ))
}

pub(super) fn ensure_model_visible_items_within_limit(
    items: &[ResponseItem],
    max_model_visible_item_tokens: i64,
) -> Result<(), ExactTailError> {
    for item in items {
        if matches!(
            classify_exact_tail_history_item(item),
            ExactTailItemClass::StaleContextWrapper
        ) {
            continue;
        }
        let item_tokens = estimate_response_items_token_count(std::slice::from_ref(item));
        if item_tokens > max_model_visible_item_tokens {
            let item_kind = model_visible_item_kind(item);
            let item_label = model_visible_item_label(item);
            warn!(
                exact_tail_enabled = true,
                exact_tail_fail_reason = ExactTailFailReason::ModelVisibleItemTooLarge.as_str(),
                item_kind,
                item_label = item_label.as_str(),
                item_tokens,
                max_item_tokens = max_model_visible_item_tokens,
                "exact-tail model-visible item exceeds per-item cap"
            );
            return Err(ExactTailError::new(
                ExactTailFailReason::ModelVisibleItemTooLarge,
                format!(
                    "{}: exact-tail {item_label} estimates to {} tokens, exceeding per-item cap {}",
                    ExactTailFailReason::ModelVisibleItemTooLarge.as_str(),
                    item_tokens,
                    max_model_visible_item_tokens
                ),
            ));
        }
    }
    Ok(())
}

fn model_visible_item_kind(item: &ResponseItem) -> &'static str {
    match item {
        ResponseItem::Message { .. } => "message",
        ResponseItem::AgentMessage { .. } => "agent_message",
        ResponseItem::Reasoning { .. } => "reasoning",
        ResponseItem::LocalShellCall { .. } => "local_shell_call",
        ResponseItem::FunctionCall { .. } => "function_call",
        ResponseItem::ToolSearchCall { .. } => "tool_search_call",
        ResponseItem::FunctionCallOutput { .. } => "function_call_output",
        ResponseItem::ToolSearchOutput { .. } => "tool_search_output",
        ResponseItem::CustomToolCall { .. } => "custom_tool_call",
        ResponseItem::CustomToolCallOutput { .. } => "custom_tool_call_output",
        ResponseItem::WebSearchCall { .. } => "web_search_call",
        ResponseItem::ImageGenerationCall { .. } => "image_generation_call",
        ResponseItem::Compaction { .. } => "compaction",
        ResponseItem::CompactionTrigger { .. } => "compaction_trigger",
        ResponseItem::ContextCompaction { .. } => "context_compaction",
        ResponseItem::Other => "other",
    }
}

fn model_visible_item_label(item: &ResponseItem) -> String {
    let kind = model_visible_item_kind(item);
    match item {
        ResponseItem::Message { role, .. } => format!("{kind} role={role}"),
        ResponseItem::FunctionCall { call_id, name, .. }
        | ResponseItem::CustomToolCall { call_id, name, .. } => {
            format!("{kind} name={name} call_id={call_id}")
        }
        ResponseItem::FunctionCallOutput { call_id, .. }
        | ResponseItem::CustomToolCallOutput { call_id, .. } => {
            format!("{kind} call_id={call_id}")
        }
        ResponseItem::ToolSearchCall { call_id, .. }
        | ResponseItem::ToolSearchOutput { call_id, .. }
        | ResponseItem::LocalShellCall { call_id, .. } => match item.id() {
            Some(id) => format!("{kind} id={id} call_id={call_id:?}"),
            None => format!("{kind} call_id={call_id:?}"),
        },
        _ => match item.id() {
            Some(id) => format!("{kind} id={id}"),
            None => kind.to_string(),
        },
    }
}

#[cfg(test)]
#[path = "compact_exact_tail_tests.rs"]
mod tests;
