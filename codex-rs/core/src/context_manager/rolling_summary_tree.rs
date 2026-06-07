#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct CoverageInterval {
    pub(super) raw_start_index: usize,
    pub(super) raw_end_exclusive: usize,
}

impl CoverageInterval {
    pub(super) fn new(raw_start_index: usize, raw_end_exclusive: usize) -> Self {
        assert!(
            raw_start_index < raw_end_exclusive,
            "coverage interval must be non-empty"
        );
        Self {
            raw_start_index,
            raw_end_exclusive,
        }
    }

    fn is_adjacent_to(self, next: Self) -> bool {
        self.raw_end_exclusive == next.raw_start_index
    }

    fn union(self, next: Self) -> Option<Self> {
        self.is_adjacent_to(next).then_some(Self {
            raw_start_index: self.raw_start_index,
            raw_end_exclusive: next.raw_end_exclusive,
        })
    }

    pub(super) fn covers(self, other: Self) -> bool {
        self.raw_start_index <= other.raw_start_index
            && self.raw_end_exclusive >= other.raw_end_exclusive
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PairwiseNodeKind {
    Raw,
    Summary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PairwiseNode {
    pub(super) coverage: CoverageInterval,
    pub(super) level: u8,
    pub(super) kind: PairwiseNodeKind,
    pub(super) token_estimate: i64,
    summary_text: Option<String>,
}

impl PairwiseNode {
    pub(super) fn raw(coverage: CoverageInterval, token_estimate: i64) -> Self {
        Self {
            coverage,
            level: 0,
            kind: PairwiseNodeKind::Raw,
            token_estimate,
            summary_text: None,
        }
    }

    fn summary(coverage: CoverageInterval, level: u8, token_estimate: i64) -> Self {
        Self::summary_with_text(
            coverage,
            level,
            token_estimate,
            format!(
                "summary for raw[{}..{})",
                coverage.raw_start_index, coverage.raw_end_exclusive
            ),
        )
    }

    pub(super) fn summary_with_text(
        coverage: CoverageInterval,
        level: u8,
        token_estimate: i64,
        summary_text: impl Into<String>,
    ) -> Self {
        Self {
            coverage,
            level,
            kind: PairwiseNodeKind::Summary,
            token_estimate,
            summary_text: Some(summary_text.into()),
        }
    }

    pub(super) fn coverage_label(&self) -> String {
        let start = self.coverage.raw_start_index;
        let end = self.coverage.raw_end_exclusive;
        format!("raw[{start}..{end})")
    }

    pub(super) fn summary_response_item(&self, summary_token_cap: usize) -> Option<ResponseItem> {
        if self.kind != PairwiseNodeKind::Summary {
            return None;
        }
        Some(ContextualUserFragment::into(
            RollctxSummaryGroupContext::new(
                self.level,
                self.coverage_label(),
                RollctxSummaryGroupContextBody::from_summary_text(
                    self.summary_text.clone().unwrap_or_default(),
                ),
                summary_token_cap,
            ),
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PairwiseMissingSummary {
    pub(super) first: PairwiseNode,
    pub(super) second: PairwiseNode,
    pub(super) coverage: CoverageInterval,
    pub(super) level: u8,
    pub(super) token_estimate: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PairwiseCompactionSettings {
    pub(super) max_summary_levels: u8,
    pub(super) compact_when_level_group_count_gt: usize,
    pub(super) summary_token_estimate: i64,
}

impl PairwiseCompactionSettings {
    fn threshold(self) -> usize {
        self.compact_when_level_group_count_gt.max(2)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PairwiseSummaryTree {
    levels: Vec<Vec<PairwiseNode>>,
}

impl PairwiseSummaryTree {
    pub(super) fn from_summaries_and_cold_groups(
        summaries: impl IntoIterator<Item = PairwiseNode>,
        cold_groups: impl IntoIterator<Item = PairwiseNode>,
        settings: PairwiseCompactionSettings,
    ) -> Self {
        let mut levels = levels_from_nodes(summaries.into_iter().chain(cold_groups));
        compact_levels(&mut levels, settings);
        Self { levels }
    }

    pub(super) fn try_from_summaries_and_cold_groups(
        summaries: impl IntoIterator<Item = PairwiseNode>,
        cold_groups: impl IntoIterator<Item = PairwiseNode>,
        settings: PairwiseCompactionSettings,
    ) -> Result<Self, PairwiseMissingSummary> {
        let mut levels = levels_from_nodes(summaries.into_iter().chain(cold_groups));
        compact_levels_requiring_summaries(&mut levels, settings)?;
        Ok(Self { levels })
    }

    pub(super) fn compact_one_synthetic(&mut self, settings: PairwiseCompactionSettings) -> bool {
        compact_one_pair_with(
            &mut self.levels,
            settings,
            |coverage, level, token_estimate, _, _| {
                Ok(PairwiseNode::summary(coverage, level, token_estimate))
            },
        )
        .unwrap_or(false)
    }

    pub(super) fn try_compact_one_requiring_summary(
        &mut self,
        settings: PairwiseCompactionSettings,
    ) -> Result<bool, PairwiseMissingSummary> {
        compact_one_pair_with(
            &mut self.levels,
            settings,
            |coverage, level, token_estimate, first, second| {
                Err(PairwiseMissingSummary {
                    first,
                    second,
                    coverage,
                    level,
                    token_estimate,
                })
            },
        )
    }

    #[cfg(test)]
    pub(super) fn from_cold_groups(
        cold_groups: impl IntoIterator<Item = PairwiseNode>,
        settings: PairwiseCompactionSettings,
    ) -> Self {
        Self::from_summaries_and_cold_groups(Vec::new(), cold_groups, settings)
    }

    pub(super) fn visible_nodes(&self) -> Vec<PairwiseNode> {
        let mut nodes = self
            .levels
            .iter()
            .flat_map(|level| level.iter().cloned())
            .collect::<Vec<_>>();
        nodes.sort_by_key(|node| {
            (
                node.coverage.raw_start_index,
                node.coverage.raw_end_exclusive,
                node.level,
            )
        });
        nodes
    }

    #[cfg(test)]
    pub(super) fn assert_exact_once_coverage(&self) {
        let mut visible = self.visible_nodes();
        visible.sort_by_key(|node| node.coverage.raw_start_index);
        for pair in visible.windows(2) {
            assert!(
                pair[0].coverage.raw_end_exclusive <= pair[1].coverage.raw_start_index,
                "visible pairwise nodes must not overlap: {pair:?}"
            );
        }
    }

    pub(super) fn summary_nodes(&self) -> Vec<PairwiseNode> {
        self.visible_nodes()
            .into_iter()
            .filter(|node| node.kind == PairwiseNodeKind::Summary)
            .collect()
    }
}

fn levels_from_nodes(nodes: impl IntoIterator<Item = PairwiseNode>) -> Vec<Vec<PairwiseNode>> {
    let mut nodes = nodes.into_iter().collect::<Vec<_>>();
    nodes.sort_by_key(|node| {
        (
            Reverse(node.level),
            node.coverage.raw_start_index,
            Reverse(node.coverage.raw_end_exclusive - node.coverage.raw_start_index),
            node.coverage.raw_end_exclusive,
        )
    });
    let mut visible = Vec::<PairwiseNode>::new();
    for node in nodes {
        let covered_by_summary = visible.iter().any(|visible_node| {
            visible_node.kind == PairwiseNodeKind::Summary
                && visible_node.level >= node.level
                && visible_node.coverage.covers(node.coverage)
        });
        if !covered_by_summary {
            visible.push(node);
        }
    }

    let mut levels = Vec::<Vec<PairwiseNode>>::new();
    for node in visible {
        let level = usize::from(node.level);
        while levels.len() <= level {
            levels.push(Vec::new());
        }
        levels[level].push(node);
    }
    sort_levels(&mut levels);
    levels
}

fn compact_levels(levels: &mut Vec<Vec<PairwiseNode>>, settings: PairwiseCompactionSettings) {
    let _ = compact_levels_with(levels, settings, |coverage, level, token_estimate, _, _| {
        Ok(PairwiseNode::summary(coverage, level, token_estimate))
    });
}

fn compact_levels_requiring_summaries(
    levels: &mut Vec<Vec<PairwiseNode>>,
    settings: PairwiseCompactionSettings,
) -> Result<(), PairwiseMissingSummary> {
    compact_levels_with(
        levels,
        settings,
        |coverage, level, token_estimate, first, second| {
            Err(PairwiseMissingSummary {
                first,
                second,
                coverage,
                level,
                token_estimate,
            })
        },
    )
}

fn compact_levels_with(
    levels: &mut Vec<Vec<PairwiseNode>>,
    settings: PairwiseCompactionSettings,
    mut build_summary: impl FnMut(
        CoverageInterval,
        u8,
        i64,
        PairwiseNode,
        PairwiseNode,
    ) -> Result<PairwiseNode, PairwiseMissingSummary>,
) -> Result<(), PairwiseMissingSummary> {
    let threshold = settings.threshold();
    let mut level = 0usize;
    while level < levels.len() {
        sort_level(&mut levels[level]);
        let can_promote = level <= usize::from(settings.max_summary_levels);
        while can_promote && levels[level].len() > threshold {
            let Some(pair_start) = oldest_adjacent_pair_index(&levels[level]) else {
                break;
            };
            let first = levels[level][pair_start].clone();
            let second = levels[level][pair_start + 1].clone();
            let Some(coverage) = first.coverage.union(second.coverage) else {
                continue;
            };
            let next_level =
                saturated_summary_level(level, usize::from(settings.max_summary_levels));
            if levels.len() <= next_level {
                levels.push(Vec::new());
            }
            let next_level = u8::try_from(next_level).unwrap_or(u8::MAX);
            let summary = build_summary(
                coverage,
                next_level,
                settings.summary_token_estimate,
                first,
                second,
            )?;
            levels[level].remove(pair_start + 1);
            levels[level].remove(pair_start);
            levels[usize::from(next_level)].push(summary);
            sort_level(&mut levels[usize::from(next_level)]);
        }
        level += 1;
    }
    Ok(())
}

fn compact_one_pair_with(
    levels: &mut Vec<Vec<PairwiseNode>>,
    settings: PairwiseCompactionSettings,
    mut build_summary: impl FnMut(
        CoverageInterval,
        u8,
        i64,
        PairwiseNode,
        PairwiseNode,
    ) -> Result<PairwiseNode, PairwiseMissingSummary>,
) -> Result<bool, PairwiseMissingSummary> {
    let Some((level, pair_start)) = oldest_promotable_pair(levels, settings) else {
        return Ok(false);
    };
    let first = levels[level][pair_start].clone();
    let second = levels[level][pair_start + 1].clone();
    let Some(coverage) = first.coverage.union(second.coverage) else {
        return Ok(false);
    };
    let next_level = saturated_summary_level(level, usize::from(settings.max_summary_levels));
    if levels.len() <= next_level {
        levels.push(Vec::new());
    }
    let next_level_u8 = u8::try_from(next_level).unwrap_or(u8::MAX);
    let summary = build_summary(
        coverage,
        next_level_u8,
        settings.summary_token_estimate,
        first,
        second,
    )?;
    levels[level].remove(pair_start + 1);
    levels[level].remove(pair_start);
    levels[next_level].push(summary);
    sort_levels(levels);
    Ok(true)
}

fn oldest_promotable_pair(
    levels: &mut [Vec<PairwiseNode>],
    settings: PairwiseCompactionSettings,
) -> Option<(usize, usize)> {
    for (level_index, level) in levels.iter_mut().enumerate() {
        if level_index > usize::from(settings.max_summary_levels) {
            continue;
        }
        sort_level(level);
        if let Some(pair_start) = oldest_adjacent_pair_index(level) {
            return Some((level_index, pair_start));
        }
    }
    None
}

fn saturated_summary_level(level: usize, max_summary_level: usize) -> usize {
    level.saturating_add(1).min(max_summary_level)
}

fn oldest_adjacent_pair_index(nodes: &[PairwiseNode]) -> Option<usize> {
    nodes
        .windows(2)
        .position(|pair| pair[0].coverage.is_adjacent_to(pair[1].coverage))
}

fn sort_levels(levels: &mut [Vec<PairwiseNode>]) {
    for level in levels {
        sort_level(level);
    }
}

fn sort_level(level: &mut [PairwiseNode]) {
    level.sort_by_key(|node| {
        (
            node.coverage.raw_start_index,
            node.coverage.raw_end_exclusive,
        )
    });
}

#[cfg(test)]
#[path = "rolling_summary_tree_tests.rs"]
mod tests;
use crate::context::ContextualUserFragment;
use crate::context::RollctxSummaryGroupContext;
use crate::context::RollctxSummaryGroupContextBody;
use codex_protocol::models::ResponseItem;
use std::cmp::Reverse;
