use super::ContextManager;
use super::estimate_response_item_token_count;
use super::rolling::RollingPromptError;
use super::rolling::RollingPromptParams;
use super::rolling::RollingPromptState;
use super::rolling::build_rolling_prompt;
use super::rolling_pairwise::PairwiseRollingPromptParams;
use super::rolling_summary_tree::CoverageInterval;
use super::rolling_summary_tree::PairwiseCompactionSettings;
use super::rolling_summary_tree::PairwiseNode;
use super::rolling_summary_tree::PairwiseSummaryTree;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::InputModality;
use codex_utils_output_truncation::TruncationPolicy;
use pretty_assertions::assert_eq;
use std::collections::BTreeMap;

const GROUP_COUNT: usize = 4_096;
const TARGET_TOKENS: i64 = 230_000;
const TOOL_OUTPUT_LIMIT_TOKENS: i64 = 10_000;

#[derive(Debug, Clone)]
struct StressCase {
    threshold: usize,
    max_levels: u8,
    protected_hot_exact_tokens: i64,
    summary_cap: usize,
}

#[derive(Debug)]
struct ShapeReport {
    target_prompt_tokens: i64,
    projected_prompt_tokens: i64,
    hot_exact_tokens: i64,
    exact_cold_tokens: i64,
    summary_tokens: i64,
    summary_group_count: usize,
    summary_levels: BTreeMap<u8, usize>,
    intervals: Vec<VisibleInterval>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisibleKind {
    Exact,
    Summary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VisibleInterval {
    start: usize,
    end: usize,
    kind: VisibleKind,
}

#[test]
fn rolling_pairwise_high_cold_budget_utilization_matrix() {
    let history = stress_history(GROUP_COUNT, 70);
    let raw_tokens = raw_item_tokens(history.raw_items());
    let cases = [
        case(2, 8, 120_000, 256),
        case(4, 8, 160_000, 512),
        case(8, 8, 180_000, 512),
        case(12, 10, 160_000, 512),
        case(16, 10, 160_000, 512),
        case(16, 12, 160_000, 512),
    ];

    let mut reports = Vec::new();
    for case in cases {
        let hot_start = protected_hot_start(&raw_tokens, case.protected_hot_exact_tokens);
        let input_modalities = [InputModality::Text];
        let base_instructions = base_instructions();
        let result = build_rolling_prompt(
            &history,
            &state_with_dense_summaries(&history, hot_start, &case),
            params(&case, TARGET_TOKENS, &base_instructions, &input_modalities),
        )
        .unwrap_or_else(|error| panic!("stress case should fit {case:?}: {error:?}"));
        let report = shape_report(&result.prompt_input, hot_start, TARGET_TOKENS);
        assert!(
            report.projected_prompt_tokens <= report.target_prompt_tokens,
            "prompt exceeded target for {case:?}: {report:#?}"
        );
        assert!(
            report.hot_exact_tokens >= case.protected_hot_exact_tokens,
            "hot suffix under target for {case:?}: {report:#?}"
        );
        assert_exact_once_coverage(&report.intervals, GROUP_COUNT);
        assert_chronological_summary_order(&report.intervals, &case, &report);
        assert!(
            report.summary_group_count > 0,
            "stress case should retain old summary lattice {case:?}: {report:#?}"
        );
        reports.push((case, report));
    }

    let baseline = reports
        .iter()
        .find(|(case, _report)| case.threshold == 2)
        .expect("baseline report")
        .1
        .summary_tokens;
    let aggressive = reports
        .iter()
        .filter(|(case, _report)| case.threshold == 16)
        .map(|(_case, report)| report.summary_tokens)
        .max()
        .expect("aggressive report");
    assert!(
        aggressive > baseline.saturating_mul(3),
        "higher threshold should materially expand cold summary use: baseline={baseline}, aggressive={aggressive}, reports={reports:#?}"
    );
    assert!(
        aggressive >= 45_000,
        "aggressive case should reach the scaled 50k-70k cold-summary range: aggressive={aggressive}, reports={reports:#?}"
    );
}

fn case(
    threshold: usize,
    max_levels: u8,
    protected_hot_exact_tokens: i64,
    summary_cap: usize,
) -> StressCase {
    StressCase {
        threshold,
        max_levels,
        protected_hot_exact_tokens,
        summary_cap,
    }
}

#[test]
fn rolling_pairwise_budget_pressure_catches_up_without_overbudget_prompt() {
    let mut items = stress_items(4_096, 70);
    items.push(user_msg("PRESSURE_NEW_GROUP ".repeat(1_500)));
    let history = history(items);
    let case = StressCase {
        threshold: 16,
        max_levels: 12,
        protected_hot_exact_tokens: 120_000,
        summary_cap: 512,
    };
    let hot_start = protected_hot_start(
        &raw_item_tokens(history.raw_items()),
        case.protected_hot_exact_tokens,
    );
    let input_modalities = [InputModality::Text];
    let base_instructions = base_instructions();
    let result = build_rolling_prompt(
        &history,
        &state_with_dense_summaries(&history, hot_start, &case),
        params(&case, 180_000, &base_instructions, &input_modalities),
    )
    .expect("budget pressure should compact older/lower-level groups enough to fit");

    assert!(
        result.estimated_prompt_tokens <= result.target_tokens,
        "pressure prompt must not exceed target: {result:#?}"
    );
    let report = shape_report(&result.prompt_input, hot_start, result.target_tokens);
    assert!(
        report.summary_group_count < case.threshold * usize::from(case.max_levels + 1),
        "budget pressure should compact the wide lattice before sending: {report:#?}"
    );
}

#[test]
fn rolling_pairwise_stress_preserves_tool_and_steer_boundaries() {
    let items = vec![
        user_msg("COLD_A ".repeat(80)),
        function_call("stress-call"),
        function_output("stress-call", &"TOOL_OUTPUT_STRESS ".repeat(120)),
        user_msg("QUEUED_STEER_STRESS ".repeat(80)),
        user_msg("HOT_A ".repeat(200)),
        user_msg("HOT_B ".repeat(200)),
    ];
    let history = history(items);
    let case = StressCase {
        threshold: 2,
        max_levels: 3,
        protected_hot_exact_tokens: 1,
        summary_cap: 256,
    };
    let input_modalities = [InputModality::Text];
    let base_instructions = base_instructions();
    let result = build_rolling_prompt(
        &history,
        &RollingPromptState::default(),
        params(&case, 2_500, &base_instructions, &input_modalities),
    )
    .expect("tool/steer stress should project");
    let prompt = prompt_text(&result.prompt_input);
    let call_visible = prompt.contains("stress-call");
    let output_visible = prompt.contains("TOOL_OUTPUT_STRESS");
    let steer_visible = prompt.contains("QUEUED_STEER_STRESS");

    assert_eq!(
        (call_visible, output_visible, steer_visible),
        (output_visible, call_visible, call_visible),
        "tool call/output and queued steer must be retained or summarized together: {prompt}"
    );
}

#[test]
fn rolling_pairwise_stress_reports_failure_shape_when_budget_impossible() {
    let history = stress_history(8, 400);
    let case = StressCase {
        threshold: 16,
        max_levels: 12,
        protected_hot_exact_tokens: 100_000,
        summary_cap: 512,
    };
    let input_modalities = [InputModality::Text];
    let base_instructions = base_instructions();
    let error = build_rolling_prompt(
        &history,
        &RollingPromptState::default(),
        params(&case, 500, &base_instructions, &input_modalities),
    )
    .expect_err("uncompactable protected frontier should fail before prompt publication");

    assert!(
        matches!(error, RollingPromptError::FrontierExceedsBudget { .. }),
        "expected frontier budget failure, got {error:?}"
    );
}

fn params<'a>(
    case: &StressCase,
    target_tokens: i64,
    base_instructions: &'a BaseInstructions,
    input_modalities: &'a [InputModality],
) -> RollingPromptParams<'a> {
    RollingPromptParams {
        input_modalities,
        base_instructions,
        invariant_prefix: Vec::new(),
        effective_context_window: Some(target_tokens),
        reserve_percent: Some(0),
        target_tokens: Some(target_tokens),
        target_scale_percent: None,
        tool_output_limit_tokens: TOOL_OUTPUT_LIMIT_TOKENS,
        pairwise_compaction: Some(PairwiseRollingPromptParams {
            protected_hot_exact_tokens: Some(case.protected_hot_exact_tokens),
            summary_group_token_cap: case.summary_cap,
            max_summary_levels: case.max_levels,
            compact_when_level_group_count_gt: case.threshold,
        }),
    }
}

fn state_with_dense_summaries(
    history: &ContextManager,
    hot_start: usize,
    case: &StressCase,
) -> RollingPromptState {
    let input_modalities = [InputModality::Text];
    let base_instructions = base_instructions();
    let initial = build_rolling_prompt(
        history,
        &RollingPromptState::default(),
        params(case, TARGET_TOKENS, &base_instructions, &input_modalities),
    )
    .expect("initial projection should establish fingerprint");
    let mut state = initial.projected_state;
    state.raw_history_start_index = 0;
    state.pairwise_summaries = dense_summary_nodes(hot_start, case);
    state
}

fn base_instructions() -> BaseInstructions {
    BaseInstructions {
        text: "stress base".to_string(),
    }
}

fn dense_summary_nodes(hot_start: usize, case: &StressCase) -> Vec<PairwiseNode> {
    let settings = PairwiseCompactionSettings {
        max_summary_levels: case.max_levels,
        compact_when_level_group_count_gt: case.threshold,
        summary_token_estimate: i64::try_from(case.summary_cap).unwrap_or(i64::MAX),
    };
    let raw_nodes = (0..hot_start).map(|index| {
        PairwiseNode::raw(
            CoverageInterval::new(index, index + 1),
            /*token_estimate*/ 70,
        )
    });
    PairwiseSummaryTree::from_cold_groups(raw_nodes, settings)
        .summary_nodes()
        .into_iter()
        .map(|node| {
            PairwiseNode::summary_with_text(
                node.coverage,
                node.level,
                node.token_estimate,
                summary_text(case.summary_cap, node.level),
            )
        })
        .collect()
}

fn shape_report(
    prompt_input: &[ResponseItem],
    hot_start: usize,
    target_prompt_tokens: i64,
) -> ShapeReport {
    let mut report = ShapeReport {
        target_prompt_tokens,
        projected_prompt_tokens: raw_item_tokens(prompt_input).into_iter().sum(),
        hot_exact_tokens: 0,
        exact_cold_tokens: 0,
        summary_tokens: 0,
        summary_group_count: 0,
        summary_levels: BTreeMap::new(),
        intervals: Vec::new(),
    };

    for item in prompt_input {
        let item_tokens = estimate_response_item_token_count(item);
        let text = item_text(item);
        if let Some((level, start, end)) = parse_summary_fragment(&text) {
            report.summary_tokens = report.summary_tokens.saturating_add(item_tokens);
            report.summary_group_count += 1;
            *report.summary_levels.entry(level).or_default() += 1;
            report.intervals.push(VisibleInterval {
                start,
                end,
                kind: VisibleKind::Summary,
            });
        } else if let Some(index) = parse_group_index(&text) {
            if index >= hot_start {
                report.hot_exact_tokens = report.hot_exact_tokens.saturating_add(item_tokens);
            } else {
                report.exact_cold_tokens = report.exact_cold_tokens.saturating_add(item_tokens);
            }
            report.intervals.push(VisibleInterval {
                start: index,
                end: index + 1,
                kind: VisibleKind::Exact,
            });
        }
    }

    report
}

fn stress_history(group_count: usize, tokens_per_group: usize) -> ContextManager {
    history(stress_items(group_count, tokens_per_group))
}

fn history(items: Vec<ResponseItem>) -> ContextManager {
    let mut history = ContextManager::new();
    history.record_items(items.iter(), TruncationPolicy::Tokens(50_000));
    history
}

fn stress_items(group_count: usize, tokens_per_group: usize) -> Vec<ResponseItem> {
    (0..group_count)
        .map(|index| user_msg(group_text(index, tokens_per_group)))
        .collect()
}

fn group_text(index: usize, tokens: usize) -> String {
    format!("GROUP_{index:04} {}", "g".repeat(tokens.saturating_mul(4)))
}

fn summary_text(tokens: usize, level: u8) -> String {
    format!(
        "SUMMARY_LEVEL_{level} {}",
        "s".repeat(tokens.saturating_mul(4))
    )
}

fn user_msg(text: impl Into<String>) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText { text: text.into() }],
        phase: None,
    }
}

fn function_call(call_id: &str) -> ResponseItem {
    ResponseItem::FunctionCall {
        id: None,
        name: "shell".to_string(),
        namespace: None,
        arguments: "{}".to_string(),
        call_id: call_id.to_string(),
    }
}

fn function_output(call_id: &str, output: &str) -> ResponseItem {
    ResponseItem::FunctionCallOutput {
        call_id: call_id.to_string(),
        output: FunctionCallOutputPayload::from_text(output.to_string()),
    }
}

fn raw_item_tokens(items: &[ResponseItem]) -> Vec<i64> {
    items
        .iter()
        .map(estimate_response_item_token_count)
        .collect()
}

fn protected_hot_start(item_tokens: &[i64], protected_hot_exact_tokens: i64) -> usize {
    let mut tokens = 0i64;
    let mut index = item_tokens.len();
    while index > 0 && tokens < protected_hot_exact_tokens {
        index -= 1;
        tokens = tokens.saturating_add(item_tokens[index]);
    }
    index
}

fn assert_exact_once_coverage(intervals: &[VisibleInterval], group_count: usize) {
    let mut sorted = intervals.to_vec();
    sorted.sort_by_key(|interval| (interval.start, interval.end));
    let mut expected_start = 0usize;
    for interval in sorted {
        assert_eq!(
            interval.start, expected_start,
            "visible interval coverage must be contiguous and exact-once: {interval:?}"
        );
        expected_start = interval.end;
    }
    assert_eq!(expected_start, group_count);
}

fn assert_chronological_summary_order(
    intervals: &[VisibleInterval],
    case: &StressCase,
    report: &ShapeReport,
) {
    let summary_intervals = intervals
        .iter()
        .filter(|interval| interval.kind == VisibleKind::Summary)
        .collect::<Vec<_>>();
    for pair in summary_intervals.windows(2) {
        assert!(
            pair[0].end <= pair[1].start,
            "summary groups should be chronological for {case:?}: {report:#?}"
        );
    }
}

fn parse_summary_fragment(text: &str) -> Option<(u8, usize, usize)> {
    if !text.contains("<rollctx_summary_group") {
        return None;
    }
    let level = attribute_value(text, "level")?.parse().ok()?;
    let covers = attribute_value(text, "covers")?;
    let range = covers
        .strip_prefix("raw[")?
        .strip_suffix(')')?
        .split_once("..")?;
    Some((level, range.0.parse().ok()?, range.1.parse().ok()?))
}

fn attribute_value<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let prefix = format!("{name}=\"");
    let start = text.find(&prefix)? + prefix.len();
    let suffix = &text[start..];
    let end = suffix.find('"')?;
    Some(&suffix[..end])
}

fn parse_group_index(text: &str) -> Option<usize> {
    let start = text.find("GROUP_")? + "GROUP_".len();
    text.get(start..start + 4)?.parse().ok()
}

fn item_text(item: &ResponseItem) -> String {
    match item {
        ResponseItem::Message { content, .. } => content
            .iter()
            .filter_map(|content| match content {
                ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        ResponseItem::FunctionCall { call_id, .. } => call_id.clone(),
        ResponseItem::FunctionCallOutput { output, .. } => output.to_string(),
        _ => serde_json::to_string(item).unwrap_or_default(),
    }
}

fn prompt_text(items: &[ResponseItem]) -> String {
    items.iter().map(item_text).collect::<Vec<_>>().join("\n")
}
