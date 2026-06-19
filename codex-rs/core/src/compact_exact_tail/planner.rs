use super::ensure_model_visible_items_within_limit;
use super::groups::build_groups;
use crate::compact::collect_user_messages;
use crate::context::parse_visible_hook_prompt_message;
use crate::context_manager::estimate_response_items_token_count;
use crate::event_mapping::has_non_contextual_dev_message_content;
use crate::event_mapping::is_contextual_dev_message_content;
use crate::event_mapping::is_contextual_user_message_content;
use codex_analytics::CompactionTrigger;
use codex_protocol::config_types::AutoCompactTokenLimitScope;
use codex_protocol::models::ResponseItem;

use super::EXACT_TAIL_CONSERVATIVE_SUMMARY_BUDGET_TOKENS;
use super::EXACT_TAIL_MIN_SAFETY_MARGIN_TOKENS;
use super::EXACT_TAIL_REPLACEMENT_OVERHEAD_MARGIN_TOKENS;
use super::ExactTailBudgetReservation;
use super::ExactTailCoverage;
use super::ExactTailDiagnostics;
use super::ExactTailError;
use super::ExactTailFailReason;
use super::ExactTailItemClass;
use super::ExactTailPlan;
use super::ExactTailPlanInput;

pub(crate) fn classify_exact_tail_history_item(item: &ResponseItem) -> ExactTailItemClass {
    match item {
        ResponseItem::Message { role, content, .. } if role == "developer" => {
            if is_contextual_dev_message_content(content) {
                if has_non_contextual_dev_message_content(content) {
                    ExactTailItemClass::MixedDeveloperContext
                } else {
                    ExactTailItemClass::StaleContextWrapper
                }
            } else {
                ExactTailItemClass::ConversationOrProtocol
            }
        }
        ResponseItem::Message {
            id, role, content, ..
        } if role == "user" => {
            if parse_visible_hook_prompt_message(id.as_ref(), content).is_some() {
                ExactTailItemClass::ConversationOrProtocol
            } else if is_contextual_user_message_content(content) {
                ExactTailItemClass::StaleContextWrapper
            } else {
                ExactTailItemClass::ConversationOrProtocol
            }
        }
        ResponseItem::Message { .. }
        | ResponseItem::AgentMessage { .. }
        | ResponseItem::Reasoning { .. }
        | ResponseItem::Compaction { .. }
        | ResponseItem::ContextCompaction { .. } => ExactTailItemClass::ConversationOrProtocol,
        ResponseItem::LocalShellCall { .. }
        | ResponseItem::FunctionCall { .. }
        | ResponseItem::ToolSearchCall { .. }
        | ResponseItem::FunctionCallOutput { .. }
        | ResponseItem::ToolSearchOutput { .. }
        | ResponseItem::CustomToolCall { .. }
        | ResponseItem::CustomToolCallOutput { .. }
        | ResponseItem::WebSearchCall { .. }
        | ResponseItem::ImageGenerationCall { .. } => ExactTailItemClass::DependencyProtocol,
        ResponseItem::CompactionTrigger { .. } => ExactTailItemClass::StaleContextWrapper,
        ResponseItem::Other => ExactTailItemClass::Unsupported,
    }
}

pub(crate) fn exact_tail_group_count(history_items: &[ResponseItem]) -> usize {
    let (groups, _, _) = build_groups(history_items);
    groups.len()
}

pub(crate) fn plan_exact_tail(
    input: ExactTailPlanInput<'_>,
) -> Result<ExactTailPlan, ExactTailError> {
    let ExactTailPlanInput {
        history_items,
        target_tokens,
        effective_replacement_budget,
        required_current_context_budget,
        final_replacement_extra_budget_tokens,
        max_model_visible_item_tokens,
        estimated_summary_scaffold_overhead_tokens,
        retained_cold_user_message_budget_tokens,
        implementation,
    } = input;
    let Some(effective_replacement_budget) = effective_replacement_budget else {
        return Err(ExactTailError::new(
            ExactTailFailReason::BudgetUnavailable,
            "Exact-tail compaction requires a model context window or auto-compaction budget.",
        ));
    };
    let budget_reservation = exact_tail_budget_reservation(
        effective_replacement_budget,
        required_current_context_budget,
        estimated_summary_scaffold_overhead_tokens,
        retained_cold_user_message_budget_tokens,
    );

    let raw_item_count = history_items.len();
    let (groups, filtered_stale_groups, filtered_context_item_count) = build_groups(history_items);
    let group_count = groups.len();

    let mut selected_group_count = 0usize;
    let mut actual_hot_tokens = 0i64;
    if let Some(newest_group) = groups.last()
        && newest_group.tokens > budget_reservation.available_for_hot
    {
        return Err(ExactTailError::new(
            ExactTailFailReason::MinimumHotSuffixTooLarge,
            format!(
                "{}: newest exact-tail group requires {} tokens but only {} are available",
                ExactTailFailReason::MinimumHotSuffixTooLarge.as_str(),
                newest_group.tokens,
                budget_reservation.available_for_hot
            ),
        ));
    }

    for group in groups.iter().rev() {
        let next_tokens = actual_hot_tokens.saturating_add(group.tokens);
        let required_to_reach_target =
            selected_group_count == 0 || actual_hot_tokens < target_tokens;
        if !required_to_reach_target {
            break;
        }
        if next_tokens > budget_reservation.available_for_hot {
            return Err(ExactTailError::new(
                ExactTailFailReason::MinimumHotSuffixTooLarge,
                format!(
                    "{}: requested exact-tail suffix requires {} tokens but only {} are available",
                    ExactTailFailReason::MinimumHotSuffixTooLarge.as_str(),
                    next_tokens,
                    budget_reservation.available_for_hot
                ),
            ));
        }
        selected_group_count += 1;
        actual_hot_tokens = next_tokens;
    }

    let cold_group_count = group_count.saturating_sub(selected_group_count);
    if cold_group_count == 0 {
        return Err(ExactTailError::new(
            ExactTailFailReason::NoColdPrefix,
            "ExactTailNoColdPrefix: exact-tail compaction requires at least one cold group to summarize.",
        ));
    }

    let mut cold_history = Vec::new();
    let mut hot_suffix = Vec::new();
    let mut cold_covered_groups = Vec::new();
    let mut hot_exact_groups = Vec::new();

    for group in groups.iter().take(cold_group_count) {
        cold_covered_groups.push(group.id);
        cold_history.extend(group.items.clone());
    }
    for group in groups.iter().skip(cold_group_count) {
        hot_exact_groups.push(group.id);
        hot_suffix.extend(group.items.clone());
    }
    ensure_model_visible_items_within_limit(&hot_suffix, max_model_visible_item_tokens)?;

    let cold_tokens = estimate_response_items_token_count(&cold_history);
    let cold_user_messages = collect_user_messages(&cold_history);
    let coverage = ExactTailCoverage {
        cold_covered_groups,
        hot_exact_groups,
        filtered_stale_groups,
    };
    let diagnostics = ExactTailDiagnostics {
        implementation,
        raw_item_count,
        group_count,
        cold_group_count,
        hot_group_count: selected_group_count,
        filtered_context_item_count,
        requested_hot_tokens: target_tokens,
        actual_hot_tokens,
        cold_tokens,
        effective_replacement_budget,
        conservative_summary_budget_tokens: EXACT_TAIL_CONSERVATIVE_SUMMARY_BUDGET_TOKENS,
        estimated_summary_scaffold_overhead_tokens,
        retained_cold_user_message_budget_tokens,
        replacement_overhead_margin_tokens: EXACT_TAIL_REPLACEMENT_OVERHEAD_MARGIN_TOKENS,
        conservative_cold_summary_budget: budget_reservation.conservative_cold_summary_budget,
        required_current_context_budget,
        final_replacement_extra_budget_tokens,
        max_model_visible_item_tokens,
        safety_margin: budget_reservation.safety_margin,
        available_for_hot: budget_reservation.available_for_hot,
    };
    trace_plan(&diagnostics);
    Ok(ExactTailPlan {
        cold_history,
        hot_suffix,
        cold_user_messages,
        diagnostics,
        coverage,
    })
}

pub(crate) fn exact_tail_budget_reservation(
    effective_replacement_budget: i64,
    required_current_context_budget: i64,
    estimated_summary_scaffold_overhead_tokens: i64,
    retained_cold_user_message_budget_tokens: i64,
) -> ExactTailBudgetReservation {
    let conservative_cold_summary_budget = EXACT_TAIL_CONSERVATIVE_SUMMARY_BUDGET_TOKENS
        .saturating_add(estimated_summary_scaffold_overhead_tokens)
        .saturating_add(retained_cold_user_message_budget_tokens)
        .saturating_add(EXACT_TAIL_REPLACEMENT_OVERHEAD_MARGIN_TOKENS);
    let safety_margin =
        EXACT_TAIL_MIN_SAFETY_MARGIN_TOKENS.max(effective_replacement_budget.saturating_div(100));
    let available_for_hot = effective_replacement_budget
        .saturating_sub(required_current_context_budget)
        .saturating_sub(conservative_cold_summary_budget)
        .saturating_sub(safety_margin);

    ExactTailBudgetReservation {
        conservative_cold_summary_budget,
        safety_margin,
        available_for_hot,
    }
}

pub(crate) fn exact_tail_replacement_budget(
    model_context_window: Option<i64>,
    model_auto_compact_token_limit: Option<i64>,
    model_auto_compact_token_limit_scope: AutoCompactTokenLimitScope,
    trigger: CompactionTrigger,
) -> Option<i64> {
    match trigger {
        CompactionTrigger::Auto => match model_auto_compact_token_limit_scope {
            AutoCompactTokenLimitScope::Total => {
                match (model_auto_compact_token_limit, model_context_window) {
                    (Some(limit), Some(window)) => Some(limit.min(window)),
                    (Some(limit), None) => Some(limit),
                    (None, window) => window,
                }
            }
            AutoCompactTokenLimitScope::BodyAfterPrefix => {
                model_context_window.or(model_auto_compact_token_limit)
            }
        },
        CompactionTrigger::Manual => model_context_window,
    }
}

fn trace_plan(diagnostics: &ExactTailDiagnostics) {
    tracing::info!(
        exact_tail_enabled = true,
        implementation = diagnostics.implementation.as_str(),
        raw_item_count = diagnostics.raw_item_count,
        group_count = diagnostics.group_count,
        requested_hot_tokens = diagnostics.requested_hot_tokens,
        actual_hot_tokens = diagnostics.actual_hot_tokens,
        cold_tokens = diagnostics.cold_tokens,
        conservative_cold_summary_budget = diagnostics.conservative_cold_summary_budget,
        estimated_summary_scaffold_overhead_tokens =
            diagnostics.estimated_summary_scaffold_overhead_tokens,
        retained_cold_user_message_budget_tokens =
            diagnostics.retained_cold_user_message_budget_tokens,
        replacement_overhead_margin_tokens = diagnostics.replacement_overhead_margin_tokens,
        required_current_context_budget = diagnostics.required_current_context_budget,
        final_replacement_extra_budget_tokens = diagnostics.final_replacement_extra_budget_tokens,
        max_model_visible_item_tokens = diagnostics.max_model_visible_item_tokens,
        safety_margin = diagnostics.safety_margin,
        available_for_hot = diagnostics.available_for_hot,
        hot_group_count = diagnostics.hot_group_count,
        cold_group_count = diagnostics.cold_group_count,
        filtered_context_item_count = diagnostics.filtered_context_item_count,
        "exact-tail compaction plan built"
    );
}
