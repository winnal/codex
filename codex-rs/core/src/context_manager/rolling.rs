use super::ContextManager;
use super::estimate_response_item_token_count;
use super::rolling_context_filter::is_historical_context_item_at;
use super::rolling_context_filter::is_turn_context_group_anchor;
use super::rolling_context_filter::is_turn_context_group_start;
use super::rolling_pairwise::PairwiseRollingPromptBuildOutcome;
use super::rolling_pairwise::PairwiseRollingPromptParams;
use super::rolling_pairwise::PairwiseSummaryMode;
use super::rolling_pairwise::PairwiseSummaryRequest;
use super::rolling_pairwise::build_pairwise_rolling_prompt;
use super::rolling_projection::project_rolling_message_item;
use super::rolling_projection::project_rolling_prompt_item;
use super::rolling_summary_tree::PairwiseNode;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::InputModality;
use codex_utils_output_truncation::approx_token_count;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;

const DEFAULT_ROLLING_CONTEXT_RESERVE_PERCENT: u8 = 10;
pub(super) const MAX_ROLLING_PROMPT_ITEM_TOKENS: i64 = 10_000;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RollingPromptState {
    pub(crate) history_version: u64,
    pub(crate) raw_history_start_index: usize,
    pub(crate) projection_basis_fingerprint: u64,
    pub(super) pairwise_summaries: Vec<PairwiseNode>,
}

impl RollingPromptState {
    pub(crate) fn commit_projection(&mut self, projected: &Self, raw_history_len: usize) {
        if self.history_version != projected.history_version {
            *self = projected.clone();
            return;
        }

        self.raw_history_start_index = if self.raw_history_start_index > raw_history_len {
            projected.raw_history_start_index
        } else {
            self.raw_history_start_index
                .max(projected.raw_history_start_index)
        };
        self.projection_basis_fingerprint = projected.projection_basis_fingerprint;
        self.pairwise_summaries = projected.pairwise_summaries.clone();
    }

    pub(super) fn add_pairwise_summary(&mut self, summary: PairwiseNode) {
        let coverage = summary.coverage;
        let level = summary.level;
        self.pairwise_summaries
            .retain(|existing| !coverage.covers(existing.coverage) || existing.level > level);
        if !self
            .pairwise_summaries
            .iter()
            .any(|existing| existing.level >= level && existing.coverage.covers(coverage))
        {
            self.pairwise_summaries.push(summary);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RollingPromptResult {
    pub(crate) prompt_input: Vec<ResponseItem>,
    pub(crate) estimated_prompt_tokens: i64,
    pub(crate) dropped_body_items: usize,
    pub(crate) raw_history_start_index: usize,
    pub(crate) target_tokens: i64,
    pub(crate) backoff_applied: bool,
    pub(crate) projected_state: RollingPromptState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RollingPromptError {
    NoContextWindow,
    InvariantPrefixExceedsTarget {
        pinned_tokens: i64,
        target_tokens: i64,
    },
    FrontierExceedsBudget {
        frontier_tokens: i64,
        body_budget: i64,
    },
    ItemExceedsLimit {
        item_tokens: i64,
        limit_tokens: i64,
    },
}

impl std::fmt::Display for RollingPromptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoContextWindow => {
                write!(f, "rolling prompt retention requires a context window")
            }
            Self::InvariantPrefixExceedsTarget {
                pinned_tokens,
                target_tokens,
            } => write!(
                f,
                "rolling prompt invariant prefix uses {pinned_tokens} tokens, exceeding target {target_tokens}"
            ),
            Self::FrontierExceedsBudget {
                frontier_tokens,
                body_budget,
            } => write!(
                f,
                "rolling prompt newest frontier uses {frontier_tokens} tokens, exceeding body budget {body_budget}"
            ),
            Self::ItemExceedsLimit {
                item_tokens,
                limit_tokens,
            } => write!(
                f,
                "rolling prompt item uses {item_tokens} tokens, exceeding per-item limit {limit_tokens}"
            ),
        }
    }
}

impl std::error::Error for RollingPromptError {}

pub(crate) struct RollingPromptParams<'a> {
    pub(crate) input_modalities: &'a [InputModality],
    pub(crate) base_instructions: &'a BaseInstructions,
    pub(crate) invariant_prefix: Vec<ResponseItem>,
    pub(crate) effective_context_window: Option<i64>,
    pub(crate) reserve_percent: Option<u8>,
    pub(crate) target_tokens: Option<i64>,
    pub(crate) target_scale_percent: Option<u8>,
    pub(crate) tool_output_limit_tokens: i64,
    pub(crate) pairwise_compaction: Option<PairwiseRollingPromptParams>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RollingPromptBuildOutcome {
    Ready(RollingPromptResult),
    NeedsPairSummary {
        request: PairwiseSummaryRequest,
        projected_state: RollingPromptState,
    },
}

pub(crate) fn build_rolling_prompt(
    history: &ContextManager,
    state: &RollingPromptState,
    params: RollingPromptParams<'_>,
) -> Result<RollingPromptResult, RollingPromptError> {
    match build_rolling_prompt_internal(history, state, params, PairwiseSummaryMode::Synthetic)? {
        RollingPromptBuildOutcome::Ready(result) => Ok(result),
        RollingPromptBuildOutcome::NeedsPairSummary { .. } => {
            unreachable!("synthetic pairwise summary mode must not request live summaries")
        }
    }
}

pub(crate) fn build_rolling_prompt_with_live_summaries(
    history: &ContextManager,
    state: &RollingPromptState,
    params: RollingPromptParams<'_>,
) -> Result<RollingPromptBuildOutcome, RollingPromptError> {
    build_rolling_prompt_internal(history, state, params, PairwiseSummaryMode::RequireReady)
}

fn build_rolling_prompt_internal(
    history: &ContextManager,
    state: &RollingPromptState,
    params: RollingPromptParams<'_>,
    pairwise_summary_mode: PairwiseSummaryMode,
) -> Result<RollingPromptBuildOutcome, RollingPromptError> {
    let Some(effective_context_window) =
        params.effective_context_window.filter(|window| *window > 0)
    else {
        return Err(RollingPromptError::NoContextWindow);
    };

    let fingerprint =
        projection_basis_fingerprint(&params, effective_context_window, &params.invariant_prefix);
    let invariant_prefix = params.invariant_prefix;
    let projected_invariant_prefix = invariant_prefix
        .iter()
        .cloned()
        .flat_map(|item| project_rolling_message_item(item, MAX_ROLLING_PROMPT_ITEM_TOKENS))
        .collect::<Vec<_>>();
    validate_prompt_items_within_limit(&projected_invariant_prefix)?;

    let mut projected_state = state.clone();
    reset_or_clamp_state(history, &mut projected_state);
    if projected_state.projection_basis_fingerprint != fingerprint {
        projected_state.projection_basis_fingerprint = fingerprint;
        projected_state.pairwise_summaries.clear();
        if params.pairwise_compaction.is_some() {
            projected_state.raw_history_start_index = 0;
        }
    }

    let target = apply_target_scale(
        rolling_target_tokens(
            effective_context_window,
            params.reserve_percent,
            params.target_tokens,
        ),
        params.target_scale_percent,
    );
    let backoff_applied = params
        .target_scale_percent
        .is_some_and(|percent| percent < 100);
    let pinned_tokens = estimate_base_instructions_tokens(params.base_instructions)
        .saturating_add(estimate_items_tokens(&projected_invariant_prefix));
    if pinned_tokens >= target {
        return Err(RollingPromptError::InvariantPrefixExceedsTarget {
            pinned_tokens,
            target_tokens: target,
        });
    }
    let body_budget = target.saturating_sub(pinned_tokens);
    let tool_output_limit_tokens = usize::try_from(
        params
            .tool_output_limit_tokens
            .clamp(1, MAX_ROLLING_PROMPT_ITEM_TOKENS),
    )
    .unwrap_or(MAX_ROLLING_PROMPT_ITEM_TOKENS as usize);

    let old_raw_start = projected_state.raw_history_start_index;
    let groups = projected_rolling_groups(
        history.raw_items(),
        old_raw_start,
        params.input_modalities,
        &invariant_prefix,
        tool_output_limit_tokens,
    );
    if let Some(pairwise_compaction) = params.pairwise_compaction {
        return match build_pairwise_rolling_prompt(
            history,
            projected_state,
            groups,
            pairwise_compaction,
            pinned_tokens,
            body_budget,
            target,
            backoff_applied,
            old_raw_start,
            &invariant_prefix,
            projected_invariant_prefix,
            pairwise_summary_mode,
        )? {
            PairwiseRollingPromptBuildOutcome::Ready(result) => {
                Ok(RollingPromptBuildOutcome::Ready(result))
            }
            PairwiseRollingPromptBuildOutcome::NeedsSummary {
                request,
                projected_state,
            } => Ok(RollingPromptBuildOutcome::NeedsPairSummary {
                request,
                projected_state,
            }),
        };
    }
    let mut kept_groups = Vec::new();
    let mut kept_tokens = 0i64;

    for group in groups.iter().rev() {
        if group.item_tokens_exceed_limit() {
            if kept_groups.is_empty() {
                return Err(group.item_limit_error());
            }
            break;
        }
        if kept_tokens.saturating_add(group.tokens) <= body_budget {
            kept_tokens = kept_tokens.saturating_add(group.tokens);
            kept_groups.push(group);
        } else {
            if kept_groups.is_empty() {
                return Err(RollingPromptError::FrontierExceedsBudget {
                    frontier_tokens: group.tokens,
                    body_budget,
                });
            }
            break;
        }
    }
    kept_groups.reverse();

    let new_raw_start = kept_groups
        .first()
        .map_or_else(|| history.raw_items().len(), |group| group.raw_start_index);
    let new_raw_start = old_raw_start.max(new_raw_start);

    let estimated_prompt_tokens = pinned_tokens.saturating_add(kept_tokens);
    let dropped_body_items = history
        .raw_items()
        .iter()
        .enumerate()
        .filter(|(index, _item)| {
            *index >= old_raw_start
                && *index < new_raw_start
                && !is_historical_context_item_at(history.raw_items(), *index, &invariant_prefix)
        })
        .count();

    let mut prompt_input = projected_invariant_prefix;
    prompt_input.extend(
        kept_groups
            .iter()
            .flat_map(|group| group.items.iter().cloned()),
    );

    projected_state.raw_history_start_index = new_raw_start;

    Ok(RollingPromptBuildOutcome::Ready(RollingPromptResult {
        prompt_input,
        estimated_prompt_tokens,
        dropped_body_items,
        raw_history_start_index: new_raw_start,
        target_tokens: target,
        backoff_applied,
        projected_state,
    }))
}

fn reset_or_clamp_state(history: &ContextManager, state: &mut RollingPromptState) {
    if state.history_version != history.history_version() {
        state.history_version = history.history_version();
        state.raw_history_start_index = 0;
        state.pairwise_summaries.clear();
    }
    state.raw_history_start_index = state.raw_history_start_index.min(history.raw_items().len());
    if state.raw_history_start_index == history.raw_items().len() {
        state.pairwise_summaries.clear();
    }
}

fn projection_basis_fingerprint(
    params: &RollingPromptParams<'_>,
    effective_context_window: i64,
    invariant_prefix: &[ResponseItem],
) -> u64 {
    let mut hasher = DefaultHasher::new();
    "rollctx-v2".hash(&mut hasher);
    params.input_modalities.hash(&mut hasher);
    params.reserve_percent.hash(&mut hasher);
    params.target_tokens.hash(&mut hasher);
    params.tool_output_limit_tokens.hash(&mut hasher);
    effective_context_window.hash(&mut hasher);
    params.pairwise_compaction.hash(&mut hasher);
    invariant_prefix.len().hash(&mut hasher);
    for item in invariant_prefix {
        serde_json::to_string(item)
            .unwrap_or_default()
            .hash(&mut hasher);
    }
    hasher.finish()
}

fn rolling_target_tokens(
    effective_context_window: i64,
    reserve_percent: Option<u8>,
    target_tokens: Option<i64>,
) -> i64 {
    let reserve_percent = i64::from(
        reserve_percent
            .unwrap_or(DEFAULT_ROLLING_CONTEXT_RESERVE_PERCENT)
            .min(90),
    );
    let reserve_tokens = effective_context_window.saturating_mul(reserve_percent) / 100;
    let window_target = effective_context_window
        .saturating_sub(reserve_tokens)
        .max(1);
    target_tokens
        .filter(|target| *target > 0)
        .map_or(window_target, |target| target.min(window_target))
}

fn apply_target_scale(target: i64, target_scale_percent: Option<u8>) -> i64 {
    let Some(percent) = target_scale_percent else {
        return target;
    };
    if percent >= 100 {
        return target;
    }
    target
        .saturating_mul(i64::from(percent))
        .saturating_div(100)
        .max(1)
}

fn estimate_base_instructions_tokens(base_instructions: &BaseInstructions) -> i64 {
    i64::try_from(approx_token_count(&base_instructions.text)).unwrap_or(i64::MAX)
}

fn estimate_items_tokens(items: &[ResponseItem]) -> i64 {
    items
        .iter()
        .map(estimate_response_item_token_count)
        .fold(0i64, i64::saturating_add)
}

fn validate_prompt_items_within_limit(items: &[ResponseItem]) -> Result<(), RollingPromptError> {
    if let Some(item_tokens) = items
        .iter()
        .map(item_token_estimate)
        .find(|tokens| *tokens > MAX_ROLLING_PROMPT_ITEM_TOKENS)
    {
        return Err(RollingPromptError::ItemExceedsLimit {
            item_tokens,
            limit_tokens: MAX_ROLLING_PROMPT_ITEM_TOKENS,
        });
    }
    Ok(())
}

#[derive(Debug)]
pub(super) struct RollingGroup {
    pub(super) raw_start_index: usize,
    pub(super) raw_end_exclusive: usize,
    pub(super) items: Vec<ResponseItem>,
    pub(super) tokens: i64,
}

impl RollingGroup {
    pub(super) fn item_tokens_exceed_limit(&self) -> bool {
        self.items
            .iter()
            .map(item_token_estimate)
            .any(|tokens| tokens > MAX_ROLLING_PROMPT_ITEM_TOKENS)
    }

    pub(super) fn item_limit_error(&self) -> RollingPromptError {
        let item_tokens = self
            .items
            .iter()
            .map(item_token_estimate)
            .filter(|tokens| *tokens > MAX_ROLLING_PROMPT_ITEM_TOKENS)
            .max()
            .unwrap_or(MAX_ROLLING_PROMPT_ITEM_TOKENS.saturating_add(1));
        RollingPromptError::ItemExceedsLimit {
            item_tokens,
            limit_tokens: MAX_ROLLING_PROMPT_ITEM_TOKENS,
        }
    }
}

pub(super) fn item_token_estimate(item: &ResponseItem) -> i64 {
    estimate_response_item_token_count(item)
}

fn projected_rolling_groups(
    raw_items: &[ResponseItem],
    raw_start_index: usize,
    input_modalities: &[InputModality],
    invariant_prefix: &[ResponseItem],
    tool_output_limit_tokens: usize,
) -> Vec<RollingGroup> {
    rolling_groups(raw_items, raw_start_index, invariant_prefix)
        .into_iter()
        .filter_map(|group| {
            let items = ContextManager::prompt_items_from_raw_items(group.items, input_modalities)
                .into_iter()
                .flat_map(|item| {
                    if is_turn_context_group_start(&item, invariant_prefix) {
                        project_rolling_message_item(item, MAX_ROLLING_PROMPT_ITEM_TOKENS)
                    } else {
                        vec![project_rolling_prompt_item(
                            item,
                            tool_output_limit_tokens,
                            MAX_ROLLING_PROMPT_ITEM_TOKENS,
                        )]
                    }
                })
                .collect::<Vec<_>>();
            if items.is_empty() {
                return None;
            }
            let tokens = estimate_items_tokens(&items);
            Some(RollingGroup {
                raw_start_index: group.raw_start_index,
                raw_end_exclusive: group.raw_end_exclusive,
                items,
                tokens,
            })
        })
        .collect()
}

#[derive(Debug)]
struct RawRollingGroup {
    raw_start_index: usize,
    raw_end_exclusive: usize,
    items: Vec<ResponseItem>,
}

fn rolling_groups(
    raw_items: &[ResponseItem],
    raw_start_index: usize,
    invariant_prefix: &[ResponseItem],
) -> Vec<RawRollingGroup> {
    let mut groups = Vec::new();
    let dependency_spans = tool_dependency_spans(raw_items);
    let mut index = 0usize;

    while index < raw_items.len() {
        let item = &raw_items[index];
        if is_historical_context_item_at(raw_items, index, invariant_prefix) {
            index += 1;
            continue;
        }

        if let Some(span_end) = dependency_spans.get(&index).copied() {
            if index >= raw_start_index {
                let carries_following_turn_input = raw_items[index..=span_end].iter().any(|item| {
                    match call_key(item).map(|key| key.kind) {
                        Some(ToolPairKind::Function | ToolPairKind::ToolSearch) => true,
                        Some(ToolPairKind::Custom) => matches!(
                            item,
                            ResponseItem::CustomToolCall { name, .. }
                                if name == codex_code_mode::PUBLIC_TOOL_NAME
                        ),
                        None => false,
                    }
                });
                let group_end = if carries_following_turn_input {
                    span_end
                        .checked_add(1)
                        .and_then(|next_index| {
                            following_turn_input_run_end(raw_items, next_index, invariant_prefix)
                        })
                        .unwrap_or(span_end)
                } else {
                    span_end
                };
                groups.push(RawRollingGroup {
                    raw_start_index: index,
                    raw_end_exclusive: group_end + 1,
                    items: raw_items[index..=group_end]
                        .iter()
                        .enumerate()
                        .filter(|&(offset, _item)| {
                            !is_historical_context_item_at(
                                raw_items,
                                index + offset,
                                invariant_prefix,
                            )
                        })
                        .map(|(_offset, item)| item.clone())
                        .collect(),
                });
                index = group_end + 1;
                continue;
            }
            index = span_end + 1;
            continue;
        }

        if output_key(item).is_some() && !is_standalone_tool_search_output(item) {
            index += 1;
            continue;
        }

        if call_key(item).is_some()
            && has_later_non_context_item(raw_items, index, invariant_prefix)
        {
            index += 1;
            continue;
        }

        if let Some(group_end) = turn_context_group_end(raw_items, index, invariant_prefix) {
            let group_start = index;
            if group_start >= raw_start_index {
                groups.push(RawRollingGroup {
                    raw_start_index: group_start,
                    raw_end_exclusive: group_end + 1,
                    items: raw_items[group_start..=group_end].to_vec(),
                });
            }
            index = group_end + 1;
            continue;
        }

        if index >= raw_start_index {
            groups.push(RawRollingGroup {
                raw_start_index: index,
                raw_end_exclusive: index + 1,
                items: vec![item.clone()],
            });
        }
        index += 1;
    }

    groups
}

fn following_turn_input_run_end(
    raw_items: &[ResponseItem],
    index: usize,
    invariant_prefix: &[ResponseItem],
) -> Option<usize> {
    let mut scan = index;
    let mut run_end = None;
    while let Some(item) = raw_items.get(scan) {
        if is_historical_context_item_at(raw_items, scan, invariant_prefix) {
            break;
        }
        if let Some(group_end) = turn_context_group_end(raw_items, scan, invariant_prefix) {
            run_end = Some(group_end);
            scan = group_end + 1;
            continue;
        }
        if matches!(item, ResponseItem::Message { role, .. } if role == "user") {
            run_end = Some(scan);
            scan += 1;
            continue;
        }
        break;
    }
    run_end
}

fn turn_context_group_end(
    raw_items: &[ResponseItem],
    index: usize,
    invariant_prefix: &[ResponseItem],
) -> Option<usize> {
    if is_turn_context_group_start(&raw_items[index], invariant_prefix) {
        let mut scan = index + 1;
        while scan < raw_items.len()
            && is_turn_context_group_start(&raw_items[scan], invariant_prefix)
        {
            scan += 1;
        }
        if scan < raw_items.len()
            && is_turn_context_group_anchor(&raw_items[scan], invariant_prefix)
        {
            scan += 1;
            while scan < raw_items.len()
                && is_turn_context_group_start(&raw_items[scan], invariant_prefix)
            {
                scan += 1;
            }
        }
        return Some(scan.saturating_sub(1));
    }

    if !is_turn_context_group_anchor(&raw_items[index], invariant_prefix) {
        return None;
    }
    let mut scan = index + 1;
    if scan >= raw_items.len() || !is_turn_context_group_start(&raw_items[scan], invariant_prefix) {
        return None;
    }
    while scan < raw_items.len() && is_turn_context_group_start(&raw_items[scan], invariant_prefix)
    {
        scan += 1;
    }
    if scan < raw_items.len()
        && is_turn_context_group_anchor(&raw_items[scan], invariant_prefix)
        && matches!(&raw_items[scan], ResponseItem::Message { role, .. } if role == "user")
    {
        return None;
    }
    Some(scan.saturating_sub(1))
}

fn has_later_non_context_item(
    raw_items: &[ResponseItem],
    index: usize,
    invariant_prefix: &[ResponseItem],
) -> bool {
    raw_items
        .iter()
        .enumerate()
        .skip(index.saturating_add(1))
        .any(|(index, _)| !is_historical_context_item_at(raw_items, index, invariant_prefix))
}

fn tool_dependency_spans(raw_items: &[ResponseItem]) -> HashMap<usize, usize> {
    let mut output_indexes = HashMap::<ToolPairKey, usize>::new();
    for (index, item) in raw_items.iter().enumerate() {
        if let Some(key) = output_key(item) {
            output_indexes.insert(key, index);
        }
    }

    let mut spans = HashMap::new();
    let mut index = 0usize;
    while index < raw_items.len() {
        let Some(key) = call_key(&raw_items[index]) else {
            index += 1;
            continue;
        };
        let Some(mut span_end) = output_indexes.get(&key).copied() else {
            index += 1;
            continue;
        };
        if span_end <= index {
            index += 1;
            continue;
        }

        let mut scan = index + 1;
        while scan <= span_end {
            if let Some(key) = call_key(&raw_items[scan])
                && let Some(output_index) = output_indexes.get(&key).copied()
            {
                span_end = span_end.max(output_index);
            }
            scan += 1;
        }

        spans.insert(index, span_end);
        index = span_end + 1;
    }
    spans
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ToolPairKey {
    kind: ToolPairKind,
    call_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ToolPairKind {
    Function,
    Custom,
    ToolSearch,
}

fn call_key(item: &ResponseItem) -> Option<ToolPairKey> {
    match item {
        ResponseItem::FunctionCall { call_id, .. } => Some(ToolPairKey {
            kind: ToolPairKind::Function,
            call_id: call_id.clone(),
        }),
        ResponseItem::LocalShellCall {
            call_id: Some(call_id),
            ..
        } => Some(ToolPairKey {
            kind: ToolPairKind::Function,
            call_id: call_id.clone(),
        }),
        ResponseItem::CustomToolCall { call_id, .. } => Some(ToolPairKey {
            kind: ToolPairKind::Custom,
            call_id: call_id.clone(),
        }),
        ResponseItem::ToolSearchCall {
            call_id: Some(call_id),
            ..
        } => Some(ToolPairKey {
            kind: ToolPairKind::ToolSearch,
            call_id: call_id.clone(),
        }),
        _ => None,
    }
}

fn output_key(item: &ResponseItem) -> Option<ToolPairKey> {
    match item {
        ResponseItem::FunctionCallOutput { call_id, .. } => Some(ToolPairKey {
            kind: ToolPairKind::Function,
            call_id: call_id.clone(),
        }),
        ResponseItem::CustomToolCallOutput { call_id, .. } => Some(ToolPairKey {
            kind: ToolPairKind::Custom,
            call_id: call_id.clone(),
        }),
        ResponseItem::ToolSearchOutput {
            call_id: Some(call_id),
            ..
        } => Some(ToolPairKey {
            kind: ToolPairKind::ToolSearch,
            call_id: call_id.clone(),
        }),
        _ => None,
    }
}

fn is_standalone_tool_search_output(item: &ResponseItem) -> bool {
    matches!(
        item,
        ResponseItem::ToolSearchOutput { execution, .. } if execution == "server"
    )
}

#[cfg(test)]
#[path = "rolling_test_support.rs"]
mod test_support;

#[cfg(test)]
#[path = "rolling_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "rolling_edge_tests.rs"]
mod edge_tests;
