use super::ContextManager;
use super::rolling::MAX_ROLLING_PROMPT_ITEM_TOKENS;
use super::rolling::RollingGroup;
use super::rolling::RollingPromptError;
use super::rolling::RollingPromptResult;
use super::rolling::RollingPromptState;
use super::rolling::item_token_estimate;
use super::rolling_context_filter::is_historical_context_item_at;
use super::rolling_summary_tree::CoverageInterval;
use super::rolling_summary_tree::PairwiseCompactionSettings;
use super::rolling_summary_tree::PairwiseMissingSummary;
use super::rolling_summary_tree::PairwiseNode;
use super::rolling_summary_tree::PairwiseNodeKind;
use super::rolling_summary_tree::PairwiseSummaryTree;
use codex_protocol::models::ResponseItem;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PairwiseRollingPromptParams {
    pub(crate) protected_hot_exact_tokens: Option<i64>,
    pub(crate) summary_group_token_cap: usize,
    pub(crate) max_summary_levels: u8,
    pub(crate) compact_when_level_group_count_gt: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PairwiseSummaryMode {
    Synthetic,
    RequireReady,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PairwiseSummaryRequest {
    pub(crate) input: Vec<ResponseItem>,
    pub(crate) raw_start_index: usize,
    pub(crate) raw_end_exclusive: usize,
    pub(crate) level: u8,
    pub(crate) token_estimate: i64,
}

impl PairwiseSummaryRequest {
    pub(crate) fn coverage_label(&self) -> String {
        let start = self.raw_start_index;
        let end = self.raw_end_exclusive;
        format!("raw[{start}..{end})")
    }

    pub(crate) fn add_to_projected_state(
        self,
        state: &mut RollingPromptState,
        summary_text: String,
    ) {
        state.raw_history_start_index = state.raw_history_start_index.max(self.raw_end_exclusive);
        state.add_pairwise_summary(PairwiseNode::summary_with_text(
            CoverageInterval::new(self.raw_start_index, self.raw_end_exclusive),
            self.level,
            self.token_estimate,
            summary_text,
        ));
    }
}

impl RollingPromptState {
    pub(crate) fn with_pairwise_summaries_from_projection(&self, projected: &Self) -> Self {
        let mut state = self.clone();
        if self.history_version == projected.history_version
            && self.projection_basis_fingerprint == projected.projection_basis_fingerprint
        {
            state.pairwise_summaries = projected.pairwise_summaries.clone();
        }
        state
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum PairwiseRollingPromptBuildOutcome {
    Ready(RollingPromptResult),
    NeedsSummary {
        request: PairwiseSummaryRequest,
        projected_state: RollingPromptState,
    },
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_pairwise_rolling_prompt(
    history: &ContextManager,
    mut projected_state: RollingPromptState,
    groups: Vec<RollingGroup>,
    pairwise_params: PairwiseRollingPromptParams,
    pinned_tokens: i64,
    body_budget: i64,
    target: i64,
    backoff_applied: bool,
    old_raw_start: usize,
    invariant_prefix: &[ResponseItem],
    projected_invariant_prefix: Vec<ResponseItem>,
    summary_mode: PairwiseSummaryMode,
) -> Result<PairwiseRollingPromptBuildOutcome, RollingPromptError> {
    let hot_start = protected_hot_start_index(
        &groups,
        pairwise_params
            .protected_hot_exact_tokens
            .unwrap_or_else(|| default_protected_hot_exact_tokens(body_budget)),
    );
    let cold_groups = &groups[..hot_start];
    let protected_groups = &groups[hot_start..];
    for group in protected_groups {
        if group.item_tokens_exceed_limit() {
            return Err(group.item_limit_error());
        }
    }

    let cold_nodes = cold_groups.iter().map(|group| {
        PairwiseNode::raw(
            CoverageInterval::new(group.raw_start_index, group.raw_end_exclusive),
            group.tokens,
        )
    });
    let settings = PairwiseCompactionSettings {
        max_summary_levels: pairwise_params.max_summary_levels,
        compact_when_level_group_count_gt: pairwise_params.compact_when_level_group_count_gt,
        summary_token_estimate: i64::try_from(pairwise_params.summary_group_token_cap)
            .unwrap_or(i64::MAX),
    };
    let mut tree = match summary_mode {
        PairwiseSummaryMode::Synthetic => PairwiseSummaryTree::from_summaries_and_cold_groups(
            projected_state.pairwise_summaries.clone(),
            cold_nodes,
            settings,
        ),
        PairwiseSummaryMode::RequireReady => {
            match PairwiseSummaryTree::try_from_summaries_and_cold_groups(
                projected_state.pairwise_summaries.clone(),
                cold_nodes,
                settings,
            ) {
                Ok(tree) => tree,
                Err(missing) => {
                    return Ok(PairwiseRollingPromptBuildOutcome::NeedsSummary {
                        request: pairwise_summary_request(
                            missing,
                            cold_groups,
                            pairwise_params.summary_group_token_cap,
                        )?,
                        projected_state,
                    });
                }
            }
        }
    };

    let prompt_body = loop {
        let prompt_body = project_prompt_body(
            &tree,
            cold_groups,
            protected_groups,
            pairwise_params.summary_group_token_cap,
        )?;
        if prompt_body.body_tokens <= body_budget {
            break prompt_body;
        }
        let compacted = match summary_mode {
            PairwiseSummaryMode::Synthetic => tree.compact_one_synthetic(settings),
            PairwiseSummaryMode::RequireReady => {
                match tree.try_compact_one_requiring_summary(settings) {
                    Ok(compacted) => compacted,
                    Err(missing) => {
                        projected_state.pairwise_summaries = tree.summary_nodes();
                        return Ok(PairwiseRollingPromptBuildOutcome::NeedsSummary {
                            request: pairwise_summary_request(
                                missing,
                                cold_groups,
                                pairwise_params.summary_group_token_cap,
                            )?,
                            projected_state,
                        });
                    }
                }
            }
        };
        if !compacted {
            return Err(RollingPromptError::FrontierExceedsBudget {
                frontier_tokens: prompt_body.body_tokens,
                body_budget,
            });
        }
    };

    let new_raw_start = prompt_body
        .first_exact_raw_start
        .unwrap_or_else(|| history.raw_items().len());
    let new_raw_start = old_raw_start.max(new_raw_start);
    let dropped_body_items = history
        .raw_items()
        .iter()
        .enumerate()
        .filter(|(index, _item)| {
            *index >= old_raw_start
                && *index < new_raw_start
                && !is_historical_context_item_at(history.raw_items(), *index, invariant_prefix)
        })
        .count();

    let mut prompt_input = projected_invariant_prefix;
    prompt_input.extend(prompt_body.items);
    let estimated_prompt_tokens = pinned_tokens.saturating_add(prompt_body.body_tokens);
    projected_state.raw_history_start_index = new_raw_start;
    projected_state.pairwise_summaries = tree.summary_nodes();

    Ok(PairwiseRollingPromptBuildOutcome::Ready(
        RollingPromptResult {
            prompt_input,
            estimated_prompt_tokens,
            dropped_body_items,
            raw_history_start_index: new_raw_start,
            target_tokens: target,
            backoff_applied,
            projected_state,
        },
    ))
}

struct ProjectedPairwisePromptBody {
    items: Vec<ResponseItem>,
    body_tokens: i64,
    first_exact_raw_start: Option<usize>,
}

fn project_prompt_body(
    tree: &PairwiseSummaryTree,
    cold_groups: &[RollingGroup],
    protected_groups: &[RollingGroup],
    summary_group_token_cap: usize,
) -> Result<ProjectedPairwisePromptBody, RollingPromptError> {
    let mut items = Vec::new();
    let mut body_tokens = 0i64;
    let mut first_exact_raw_start = None::<usize>;
    for node in tree.visible_nodes() {
        match node.kind {
            PairwiseNodeKind::Summary => {
                let Some(item) = node.summary_response_item(summary_group_token_cap) else {
                    continue;
                };
                let item_tokens = item_token_estimate(&item);
                if item_tokens > MAX_ROLLING_PROMPT_ITEM_TOKENS {
                    return Err(RollingPromptError::ItemExceedsLimit {
                        item_tokens,
                        limit_tokens: MAX_ROLLING_PROMPT_ITEM_TOKENS,
                    });
                }
                body_tokens = body_tokens.saturating_add(item_tokens);
                items.push(item);
            }
            PairwiseNodeKind::Raw => {
                let Some(group) = cold_groups.iter().find(|group| {
                    group.raw_start_index == node.coverage.raw_start_index
                        && group.raw_end_exclusive == node.coverage.raw_end_exclusive
                }) else {
                    continue;
                };
                if group.item_tokens_exceed_limit() {
                    return Err(group.item_limit_error());
                }
                first_exact_raw_start = Some(
                    first_exact_raw_start.map_or(group.raw_start_index, |start| {
                        start.min(group.raw_start_index)
                    }),
                );
                body_tokens = body_tokens.saturating_add(group.tokens);
                items.extend(group.items.iter().cloned());
            }
        }
    }

    for group in protected_groups {
        first_exact_raw_start = Some(
            first_exact_raw_start.map_or(group.raw_start_index, |start| {
                start.min(group.raw_start_index)
            }),
        );
        body_tokens = body_tokens.saturating_add(group.tokens);
        items.extend(group.items.iter().cloned());
    }

    Ok(ProjectedPairwisePromptBody {
        items,
        body_tokens,
        first_exact_raw_start,
    })
}

fn pairwise_summary_request(
    missing: PairwiseMissingSummary,
    cold_groups: &[RollingGroup],
    summary_group_token_cap: usize,
) -> Result<PairwiseSummaryRequest, RollingPromptError> {
    let mut input = Vec::new();
    input.extend(pairwise_node_input_items(
        &missing.first,
        cold_groups,
        summary_group_token_cap,
    )?);
    input.extend(pairwise_node_input_items(
        &missing.second,
        cold_groups,
        summary_group_token_cap,
    )?);
    Ok(PairwiseSummaryRequest {
        input,
        raw_start_index: missing.coverage.raw_start_index,
        raw_end_exclusive: missing.coverage.raw_end_exclusive,
        level: missing.level,
        token_estimate: missing.token_estimate,
    })
}

fn pairwise_node_input_items(
    node: &PairwiseNode,
    cold_groups: &[RollingGroup],
    summary_group_token_cap: usize,
) -> Result<Vec<ResponseItem>, RollingPromptError> {
    match node.kind {
        PairwiseNodeKind::Raw => {
            let Some(group) = cold_groups.iter().find(|group| {
                group.raw_start_index == node.coverage.raw_start_index
                    && group.raw_end_exclusive == node.coverage.raw_end_exclusive
            }) else {
                return Ok(Vec::new());
            };
            if group.item_tokens_exceed_limit() {
                return Err(group.item_limit_error());
            }
            Ok(group.items.clone())
        }
        PairwiseNodeKind::Summary => Ok(node
            .summary_response_item(summary_group_token_cap)
            .into_iter()
            .collect()),
    }
}

fn protected_hot_start_index(groups: &[RollingGroup], protected_hot_exact_tokens: i64) -> usize {
    if groups.is_empty() {
        return 0;
    }
    if protected_hot_exact_tokens <= 0 {
        return groups.len();
    }

    let mut tokens = 0i64;
    let mut index = groups.len();
    while index > 0 && tokens < protected_hot_exact_tokens {
        index -= 1;
        tokens = tokens.saturating_add(groups[index].tokens);
    }
    index
}

fn default_protected_hot_exact_tokens(body_budget: i64) -> i64 {
    body_budget.saturating_mul(2).saturating_div(3).min(160_000)
}
