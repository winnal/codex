use super::rolling::RollingPromptState;
use super::rolling_pairwise::PairwiseSummaryRequest;
use super::rolling_summary_tree::CoverageInterval;
use super::rolling_summary_tree::PairwiseNode;
use pretty_assertions::assert_eq;

#[test]
fn pairwise_summary_cache_add_does_not_advance_raw_start() {
    let request = summary_request(0, 2, 1);
    let mut state = RollingPromptState {
        history_version: 7,
        raw_history_start_index: 9,
        projection_basis_fingerprint: 42,
        pairwise_summaries: Vec::new(),
    };

    request.add_to_summary_cache(&mut state, "cached summary".to_string());

    assert_eq!(state.raw_history_start_index, 9);
    assert_eq!(
        state.pairwise_summaries,
        vec![PairwiseNode::summary_with_text(
            CoverageInterval::new(0, 2),
            1,
            256,
            "cached summary"
        )]
    );
}

#[test]
fn pairwise_summary_cache_commit_preserves_projection_cursor() {
    let mut committed = RollingPromptState {
        history_version: 5,
        raw_history_start_index: 11,
        projection_basis_fingerprint: 100,
        pairwise_summaries: vec![PairwiseNode::summary_with_text(
            CoverageInterval::new(8, 10),
            1,
            256,
            "stale",
        )],
    };
    let projected = RollingPromptState {
        history_version: 5,
        raw_history_start_index: 20,
        projection_basis_fingerprint: 100,
        pairwise_summaries: vec![PairwiseNode::summary_with_text(
            CoverageInterval::new(0, 2),
            1,
            256,
            "fresh",
        )],
    };

    committed.commit_pairwise_summary_cache(&projected);

    assert_eq!(committed.raw_history_start_index, 11);
    assert_eq!(committed.projection_basis_fingerprint, 100);
    assert_eq!(
        committed.pairwise_summaries,
        vec![
            PairwiseNode::summary_with_text(CoverageInterval::new(8, 10), 1, 256, "stale",),
            PairwiseNode::summary_with_text(CoverageInterval::new(0, 2), 1, 256, "fresh",)
        ]
    );
}

#[test]
fn pairwise_summary_cache_commit_resets_projection_cursor_on_basis_change() {
    let mut committed = RollingPromptState {
        history_version: 5,
        raw_history_start_index: 11,
        projection_basis_fingerprint: 100,
        pairwise_summaries: Vec::new(),
    };
    let projected = RollingPromptState {
        history_version: 5,
        raw_history_start_index: 0,
        projection_basis_fingerprint: 200,
        pairwise_summaries: vec![PairwiseNode::summary_with_text(
            CoverageInterval::new(0, 2),
            1,
            256,
            "fresh basis",
        )],
    };

    committed.commit_pairwise_summary_cache(&projected);

    assert_eq!(committed.raw_history_start_index, 0);
    assert_eq!(committed.projection_basis_fingerprint, 200);
    assert_eq!(
        committed.pairwise_summaries,
        vec![PairwiseNode::summary_with_text(
            CoverageInterval::new(0, 2),
            1,
            256,
            "fresh basis",
        )]
    );
}

#[test]
fn pairwise_summary_cache_commit_ignores_stale_history_version() {
    let mut committed = RollingPromptState {
        history_version: 5,
        raw_history_start_index: 11,
        projection_basis_fingerprint: 100,
        pairwise_summaries: Vec::new(),
    };
    let projected = RollingPromptState {
        history_version: 6,
        raw_history_start_index: 20,
        projection_basis_fingerprint: 200,
        pairwise_summaries: vec![PairwiseNode::summary_with_text(
            CoverageInterval::new(0, 2),
            1,
            256,
            "stale history",
        )],
    };

    committed.commit_pairwise_summary_cache(&projected);

    assert_eq!(
        committed,
        RollingPromptState {
            history_version: 5,
            raw_history_start_index: 11,
            projection_basis_fingerprint: 100,
            pairwise_summaries: Vec::new(),
        }
    );
}

fn summary_request(
    raw_start_index: usize,
    raw_end_exclusive: usize,
    level: u8,
) -> PairwiseSummaryRequest {
    PairwiseSummaryRequest {
        input: Vec::new(),
        raw_start_index,
        raw_end_exclusive,
        level,
        token_estimate: 256,
    }
}
