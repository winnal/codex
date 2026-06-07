use super::*;
use pretty_assertions::assert_eq;

fn settings() -> PairwiseCompactionSettings {
    PairwiseCompactionSettings {
        max_summary_levels: 3,
        compact_when_level_group_count_gt: 2,
        summary_token_estimate: 10,
    }
}

fn raw_node(start: usize, end: usize) -> PairwiseNode {
    PairwiseNode::raw(CoverageInterval::new(start, end), 100)
}

fn summary_node(start: usize, end: usize, level: u8) -> PairwiseNode {
    PairwiseNode::summary_with_text(
        CoverageInterval::new(start, end),
        level,
        10,
        format!("summary raw[{start}..{end})"),
    )
}

fn visible_shape(tree: &PairwiseSummaryTree) -> Vec<(usize, usize, u8, PairwiseNodeKind)> {
    tree.visible_nodes()
        .into_iter()
        .map(|node| {
            (
                node.coverage.raw_start_index,
                node.coverage.raw_end_exclusive,
                node.level,
                node.kind,
            )
        })
        .collect()
}

fn summary_count(tree: &PairwiseSummaryTree) -> usize {
    tree.visible_nodes()
        .into_iter()
        .filter(|node| node.kind == PairwiseNodeKind::Summary)
        .count()
}

#[test]
fn existing_parent_summary_dominates_children_and_raw_groups() {
    let tree = PairwiseSummaryTree::from_summaries_and_cold_groups(
        [
            summary_node(0, 2, 1),
            summary_node(2, 4, 1),
            summary_node(0, 4, 2),
        ],
        [
            raw_node(0, 1),
            raw_node(1, 2),
            raw_node(2, 3),
            raw_node(3, 4),
            raw_node(4, 5),
        ],
        settings(),
    );

    assert_eq!(
        visible_shape(&tree),
        vec![
            (0, 4, 2, PairwiseNodeKind::Summary),
            (4, 5, 0, PairwiseNodeKind::Raw),
        ]
    );
    tree.assert_exact_once_coverage();
}

#[test]
fn oldest_eligible_adjacent_cold_pair_compacts_to_next_level() {
    let tree = PairwiseSummaryTree::from_cold_groups(
        [raw_node(0, 1), raw_node(1, 2), raw_node(2, 3)],
        settings(),
    );

    assert_eq!(
        visible_shape(&tree),
        vec![
            (0, 2, 1, PairwiseNodeKind::Summary),
            (2, 3, 0, PairwiseNodeKind::Raw),
        ]
    );
    tree.assert_exact_once_coverage();
}

#[test]
fn non_contiguous_groups_do_not_compact() {
    let tree = PairwiseSummaryTree::from_cold_groups(
        [raw_node(0, 1), raw_node(2, 3), raw_node(4, 5)],
        settings(),
    );

    assert_eq!(
        visible_shape(&tree),
        vec![
            (0, 1, 0, PairwiseNodeKind::Raw),
            (2, 3, 0, PairwiseNodeKind::Raw),
            (4, 5, 0, PairwiseNodeKind::Raw),
        ]
    );
    tree.assert_exact_once_coverage();
}

#[test]
fn levels_merge_oldest_pair_first() {
    let tree = PairwiseSummaryTree::from_cold_groups(
        [
            raw_node(0, 1),
            raw_node(1, 2),
            raw_node(2, 3),
            raw_node(3, 4),
            raw_node(4, 5),
            raw_node(5, 6),
            raw_node(6, 7),
        ],
        settings(),
    );

    assert_eq!(
        visible_shape(&tree),
        vec![
            (0, 4, 2, PairwiseNodeKind::Summary),
            (4, 6, 1, PairwiseNodeKind::Summary),
            (6, 7, 0, PairwiseNodeKind::Raw),
        ]
    );
    tree.assert_exact_once_coverage();
}

#[test]
fn chronological_projection_is_by_coverage_not_level() {
    let tree = PairwiseSummaryTree::from_cold_groups(
        [
            raw_node(0, 1),
            raw_node(1, 2),
            raw_node(2, 3),
            raw_node(3, 4),
            raw_node(4, 5),
            raw_node(5, 6),
            raw_node(6, 7),
            raw_node(7, 8),
        ],
        settings(),
    );

    assert_eq!(
        visible_shape(&tree),
        vec![
            (0, 4, 2, PairwiseNodeKind::Summary),
            (4, 6, 1, PairwiseNodeKind::Summary),
            (6, 7, 0, PairwiseNodeKind::Raw),
            (7, 8, 0, PairwiseNodeKind::Raw),
        ]
    );
    tree.assert_exact_once_coverage();
}

#[test]
fn max_summary_level_bounds_upward_merges() {
    let tree = PairwiseSummaryTree::from_cold_groups(
        [
            raw_node(0, 1),
            raw_node(1, 2),
            raw_node(2, 3),
            raw_node(3, 4),
            raw_node(4, 5),
            raw_node(5, 6),
            raw_node(6, 7),
        ],
        PairwiseCompactionSettings {
            max_summary_levels: 1,
            compact_when_level_group_count_gt: 2,
            summary_token_estimate: 10,
        },
    );

    assert_eq!(
        visible_shape(&tree),
        vec![
            (0, 4, 1, PairwiseNodeKind::Summary),
            (4, 6, 1, PairwiseNodeKind::Summary),
            (6, 7, 0, PairwiseNodeKind::Raw),
        ]
    );
    assert_eq!(summary_count(&tree), 2);
    tree.assert_exact_once_coverage();
}

#[test]
fn max_summary_level_saturates_to_bound_visible_summary_count() {
    let tree = PairwiseSummaryTree::from_cold_groups(
        (0..16).map(|index| raw_node(index, index + 1)),
        PairwiseCompactionSettings {
            max_summary_levels: 1,
            compact_when_level_group_count_gt: 2,
            summary_token_estimate: 10,
        },
    );

    assert_eq!(
        visible_shape(&tree),
        vec![
            (0, 12, 1, PairwiseNodeKind::Summary),
            (12, 14, 1, PairwiseNodeKind::Summary),
            (14, 15, 0, PairwiseNodeKind::Raw),
            (15, 16, 0, PairwiseNodeKind::Raw),
        ]
    );
    assert_eq!(summary_count(&tree), 2);
    tree.assert_exact_once_coverage();
}
