use super::super::rolling_summary_tree::CoverageInterval;
use super::test_support::*;
use super::*;
use codex_protocol::openai_models::default_input_modalities;
use pretty_assertions::assert_eq;

#[test]
fn rolling_prompt_filters_historical_context_and_keeps_current_prefix_once() {
    let history = history(vec![
        developer_msg("<permissions instructions>\nstale"),
        contextual_user_msg("<environment_context>\nstale\n</environment_context>"),
        contextual_user_msg("<user_instructions>\nstale\n</user_instructions>"),
        user_msg("old body"),
        assistant_msg("new body"),
    ]);
    let state = RollingPromptState::default();
    let current_prefix = vec![developer_msg("<permissions instructions>\ncurrent")];

    let prompt_result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: current_prefix.clone(),
            effective_context_window: Some(1_000),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("rolling prompt should fit");

    assert_eq!(
        prompt_result.prompt_input,
        vec![
            current_prefix[0].clone(),
            user_msg("old body"),
            assistant_msg("new body"),
        ]
    );
}

#[test]
fn rolling_prompt_keeps_literal_user_messages_that_only_start_like_context() {
    let literals = vec![
        user_msg("<environment_context> please keep this literal"),
        user_msg("<user_instructions> please keep this literal"),
        user_msg("# AGENTS.md instructions for /tmp should stay literal"),
    ];
    let history = history(literals.clone());
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(1_000),
            reserve_percent: Some(0),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("literal user messages should fit");

    assert_eq!(result.prompt_input, literals);
}

#[test]
fn rolling_prompt_does_not_drop_literal_user_message_equal_to_current_prefix() {
    let literal = user_msg("literal body that matches the current prefix exactly");
    let history = history(vec![literal.clone(), assistant_msg("new")]);
    let state = RollingPromptState::default();
    let current_prefix = vec![literal.clone()];

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: current_prefix.clone(),
            effective_context_window: Some(1_000),
            reserve_percent: Some(0),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("literal user message matching current prefix should fit");

    assert_eq!(
        result.prompt_input,
        vec![current_prefix[0].clone(), literal, assistant_msg("new")]
    );
}

#[test]
fn rolling_prompt_keeps_turn_input_context_and_user_prompt_atomic() {
    let prelude = developer_msg(&"TURN_PRELUDE_CONTEXT ".repeat(120));
    let user = user_msg("USER_PROMPT_SENTINEL");
    let history = history(vec![prelude, user]);
    let state = RollingPromptState::default();

    let error = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(500),
            reserve_percent: Some(0),
            target_tokens: Some(80),
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect_err("turn prelude and user prompt should fail as one newest frontier");

    assert!(matches!(
        error,
        RollingPromptError::FrontierExceedsBudget { .. }
    ));
}

#[test]
fn rolling_prompt_keeps_newest_suffix_and_advances_raw_cursor() {
    let history = history(vec![
        user_msg(&"old ".repeat(200)),
        assistant_msg(&"middle ".repeat(200)),
        user_msg("new"),
    ]);
    let mut state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(120),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("newest frontier should fit");

    assert_eq!(result.prompt_input, vec![user_msg("new")]);
    assert_eq!(result.raw_history_start_index, 2);
    assert_eq!(result.backoff_applied, false);
    assert_eq!(state.raw_history_start_index, 0);
    state.commit_projection(&result.projected_state, history.raw_items().len());
    assert_eq!(state.raw_history_start_index, 2);

    let second = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(10_000),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("larger later budget should still not reintroduce old items");

    assert_eq!(second.prompt_input, vec![user_msg("new")]);
    assert_eq!(state.raw_history_start_index, 2);
}

#[test]
fn rolling_prompt_resets_cursor_after_history_rewrite() {
    let mut history = history(vec![user_msg(&"old ".repeat(200)), user_msg("new")]);
    let mut state = RollingPromptState::default();

    let first = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(120),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("newest frontier should fit");

    assert_eq!(first.prompt_input, vec![user_msg("new")]);
    assert_eq!(state.raw_history_start_index, 0);
    state.commit_projection(&first.projected_state, history.raw_items().len());
    assert_eq!(state.raw_history_start_index, 1);

    history.remove_first_item();
    let second = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(10_000),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("history rewrite should reset cursor coordinates");

    assert_eq!(second.prompt_input, vec![user_msg("new")]);
    assert_eq!(state.raw_history_start_index, 1);
    state.commit_projection(&second.projected_state, history.raw_items().len());
    assert_eq!(state.raw_history_start_index, 0);
}

#[test]
fn rolling_prompt_keeps_adjacent_tool_call_and_output_atomic() {
    let history = history(vec![
        user_msg(&"old ".repeat(200)),
        function_call("call-1"),
        function_output("call-1", "ok"),
    ]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(120),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("call/output frontier should fit");

    assert_eq!(
        result.prompt_input,
        vec![function_call("call-1"), function_output("call-1", "ok")]
    );
}

#[test]
fn rolling_prompt_keeps_tool_followup_and_queued_user_input_atomic() {
    let history = history(vec![
        user_msg(&"old ".repeat(200)),
        function_call("call-1"),
        function_output("call-1", "ok"),
        user_msg("queued steer"),
    ]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(120),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("tool follow-up and queued input frontier should fit");

    assert_eq!(
        result.prompt_input,
        vec![
            function_call("call-1"),
            function_output("call-1", "ok"),
            user_msg("queued steer"),
        ]
    );
}

#[test]
fn rolling_prompt_does_not_detach_later_queued_input_from_tool_followup_under_pressure() {
    let history = history(vec![
        function_call("call-1"),
        function_output("call-1", "ok"),
        user_msg(&"queued-one ".repeat(400)),
        user_msg("queued two"),
    ]);
    let state = RollingPromptState::default();

    let error = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(120),
            reserve_percent: Some(0),
            target_tokens: Some(30),
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect_err("rolling prompt should not detach later queued input from its tool follow-up");

    assert!(matches!(
        error,
        RollingPromptError::FrontierExceedsBudget { .. }
    ));
}

#[test]
fn rolling_prompt_keeps_non_adjacent_parallel_tool_calls_atomic() {
    let history = history(vec![
        user_msg(&"old ".repeat(200)),
        function_call("call-1"),
        function_call("call-2"),
        function_output("call-1", "one"),
        function_output("call-2", "two"),
    ]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(120),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("parallel call/output batch should fit");

    assert_eq!(
        result.prompt_input,
        vec![
            function_call("call-1"),
            function_call("call-2"),
            function_output("call-1", "one"),
            function_output("call-2", "two"),
        ]
    );
    assert_eq!(result.raw_history_start_index, 1);
}

#[test]
fn rolling_prompt_drops_old_in_flight_tool_call_when_newer_body_exists() {
    let history = history(vec![function_call("call-1"), user_msg("new")]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(1_000),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("newer body should fit without stale in-flight call");

    assert_eq!(result.prompt_input, vec![user_msg("new")]);
    assert_eq!(result.raw_history_start_index, 1);
}

#[test]
fn rolling_prompt_keeps_local_shell_call_and_output_atomic() {
    let history = history(vec![
        user_msg(&"old ".repeat(200)),
        local_shell_call("call-1"),
        function_output("call-1", "ok"),
    ]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(120),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("local shell call/output frontier should fit");

    assert_eq!(
        result.prompt_input,
        vec![local_shell_call("call-1"), function_output("call-1", "ok")]
    );
}

#[test]
fn rolling_prompt_keeps_custom_tool_call_and_output_atomic() {
    let history = history(vec![
        user_msg(&"old ".repeat(200)),
        custom_tool_call("call-1"),
        custom_tool_output("call-1", "ok"),
    ]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(120),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("custom call/output frontier should fit");

    assert_eq!(
        result.prompt_input,
        vec![
            custom_tool_call("call-1"),
            custom_tool_output("call-1", "ok")
        ]
    );
}

#[test]
fn rolling_prompt_keeps_tool_search_call_and_output_atomic() {
    let history = history(vec![
        user_msg(&"old ".repeat(200)),
        tool_search_call("call-1"),
        tool_search_output("call-1"),
    ]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(120),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("tool search call/output frontier should fit");

    assert_eq!(
        result.prompt_input,
        vec![tool_search_call("call-1"), tool_search_output("call-1")]
    );
}

#[test]
fn rolling_prompt_keeps_server_tool_search_output_without_matching_call() {
    let output = server_tool_search_output("server-search");
    let history = history(vec![output.clone()]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(1_000),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("server-executed search output is valid model-visible history");

    assert_eq!(result.prompt_input, vec![output]);
}

#[test]
fn rolling_prompt_skips_orphan_tool_outputs() {
    let history = history(vec![
        function_output("missing-function", "orphan"),
        custom_tool_output("missing-custom", "orphan"),
        tool_search_output("missing-search"),
        user_msg("new"),
    ]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(1_000),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("new body should fit without orphan outputs");

    assert_eq!(result.prompt_input, vec![user_msg("new")]);
}

#[test]
fn rolling_prompt_clamps_cursor_to_raw_history_len() {
    let history = history(vec![user_msg("new")]);
    let mut state = RollingPromptState {
        history_version: history.history_version(),
        raw_history_start_index: 99,
        ..RollingPromptState::default()
    };

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(1_000),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("clamped cursor should produce a valid prompt");

    assert_eq!(result.prompt_input, Vec::<ResponseItem>::new());
    assert_eq!(state.raw_history_start_index, 99);
    state.commit_projection(&result.projected_state, history.raw_items().len());
    assert_eq!(state.raw_history_start_index, 1);
}

#[test]
fn rolling_prompt_errors_when_newest_frontier_exceeds_budget() {
    let history = history(vec![user_msg(&"new ".repeat(400))]);
    let state = RollingPromptState::default();

    let error = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(80),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect_err("oversized newest frontier should fail clearly");

    assert!(matches!(
        error,
        RollingPromptError::FrontierExceedsBudget { .. }
    ));
    assert_eq!(state.raw_history_start_index, 0);
}

#[test]
fn rolling_prompt_drops_old_item_that_exceeds_per_item_limit() {
    let newest = user_msg("new");
    let history = history(vec![user_msg(&"old ".repeat(20_000)), newest.clone()]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(100_000),
            reserve_percent: Some(0),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("oversized old item should be ejected");

    assert_eq!(result.prompt_input, vec![newest]);
    assert_eq!(result.raw_history_start_index, 1);
}

#[test]
fn rolling_prompt_accepts_newest_item_above_ten_kb_when_estimated_under_token_limit() {
    let text_above_ten_kb = "x".repeat(12_000);
    let newest = user_msg(&text_above_ten_kb);
    let history = history(vec![newest.clone()]);
    let state = RollingPromptState::default();

    assert!(text_above_ten_kb.len() > 10_000);
    assert!(
        item_token_estimate(&newest) < MAX_ROLLING_PROMPT_ITEM_TOKENS,
        "fixture should be above 10KB but below the 10K-token item cap"
    );

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(20_000),
            reserve_percent: Some(0),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("item below the 10K-token cap should be retained even above 10KB");

    assert_eq!(result.prompt_input, vec![newest]);
}

#[test]
fn rolling_prompt_errors_when_newest_item_exceeds_per_item_limit() {
    let history = history(vec![user_msg(&"new ".repeat(20_000))]);
    let state = RollingPromptState::default();

    let error = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(100_000),
            reserve_percent: Some(0),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect_err("oversized newest item should fail clearly");

    assert!(matches!(error, RollingPromptError::ItemExceedsLimit { .. }));
    assert_eq!(state.raw_history_start_index, 0);
}

#[test]
fn rolling_prompt_applies_backoff_to_target_metadata() {
    let history = history(vec![user_msg("new")]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(1_000),
            reserve_percent: Some(10),
            target_tokens: None,
            target_scale_percent: Some(90),
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("backoff prompt should fit");

    assert_eq!(result.target_tokens, 810);
    assert_eq!(result.backoff_applied, true);
}

#[test]
fn rolling_prompt_backoff_drops_oldest_group_that_fit_before_retry() {
    let old = user_msg(&"old ".repeat(100));
    let newest = user_msg(&"new ".repeat(800));
    let history = history(vec![old.clone(), newest.clone()]);
    let state = RollingPromptState::default();

    let normal = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(1_000),
            reserve_percent: Some(0),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("normal prompt should keep both groups");

    assert_eq!(normal.prompt_input, vec![old, newest.clone()]);
    assert_eq!(normal.raw_history_start_index, 0);

    let retry = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(1_000),
            reserve_percent: Some(0),
            target_tokens: None,
            target_scale_percent: Some(90),
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: None,
        },
    )
    .expect("backoff prompt should keep the newest frontier");

    assert_eq!(retry.prompt_input, vec![newest]);
    assert_eq!(retry.raw_history_start_index, 1);
    assert_eq!(retry.backoff_applied, true);
}

fn pairwise_params() -> PairwiseRollingPromptParams {
    PairwiseRollingPromptParams {
        protected_hot_exact_tokens: Some(1),
        summary_group_token_cap: 1_000,
        max_summary_levels: 3,
        compact_when_level_group_count_gt: 2,
    }
}

fn pairwise_summary(start: usize, end: usize, level: u8, text: &str) -> PairwiseNode {
    PairwiseNode::summary_with_text(
        CoverageInterval::new(start, end),
        level,
        1_000,
        text.to_string(),
    )
}

#[test]
fn rolling_prompt_pairwise_projects_summaries_and_exact_hot_suffix_by_coverage() {
    let history = history(vec![
        user_msg("G0_SENTINEL"),
        user_msg("G1_SENTINEL"),
        user_msg("G2_SENTINEL"),
        user_msg("G3_SENTINEL"),
        user_msg("G4_SENTINEL"),
        user_msg("G5_SENTINEL"),
        user_msg("G6_SENTINEL"),
        user_msg("G7_SENTINEL"),
    ]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(10_000),
            reserve_percent: Some(0),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: Some(pairwise_params()),
        },
    )
    .expect("pairwise prompt should fit");

    let prompt = serde_json::to_string(&result.prompt_input).expect("serialize prompt");
    let summary_0123 = prompt.find("raw[0..4)").expect("S0123 summary");
    let summary_45 = prompt.find("raw[4..6)").expect("S45 summary");
    let exact_g6 = prompt.find("G6_SENTINEL").expect("G6 exact");
    let exact_g7 = prompt.find("G7_SENTINEL").expect("G7 exact");
    assert!(summary_0123 < summary_45);
    assert!(summary_45 < exact_g6);
    assert!(exact_g6 < exact_g7);
    assert!(!prompt.contains("G0_SENTINEL"));
    assert!(!prompt.contains("G1_SENTINEL"));
    assert!(!prompt.contains("G2_SENTINEL"));
    assert!(!prompt.contains("G3_SENTINEL"));
    assert!(!prompt.contains("G4_SENTINEL"));
    assert!(!prompt.contains("G5_SENTINEL"));
}

#[test]
fn rolling_prompt_pairwise_generated_parent_retires_child_summaries() {
    let history = history(vec![
        user_msg("G0_SENTINEL"),
        user_msg("G1_SENTINEL"),
        user_msg("G2_SENTINEL"),
        user_msg("G3_SENTINEL"),
        user_msg("G4_SENTINEL"),
        user_msg("G5_SENTINEL"),
    ]);
    let input_modalities = default_input_modalities();
    let base_instructions = base_instructions();
    let params = RollingPromptParams {
        input_modalities: &input_modalities,
        base_instructions: &base_instructions,
        invariant_prefix: Vec::new(),
        effective_context_window: Some(10_000),
        reserve_percent: Some(0),
        target_tokens: None,
        target_scale_percent: None,
        tool_output_limit_tokens: 10_000,
        pairwise_compaction: Some(pairwise_params()),
    };
    let mut state = RollingPromptState {
        history_version: history.history_version(),
        projection_basis_fingerprint: projection_basis_fingerprint(&params, 10_000, &[]),
        pairwise_summaries: vec![
            pairwise_summary(0, 2, 1, "S01_SENTINEL"),
            pairwise_summary(2, 4, 1, "S23_SENTINEL"),
        ],
        ..RollingPromptState::default()
    };
    PairwiseSummaryRequest {
        input: Vec::new(),
        raw_start_index: 0,
        raw_end_exclusive: 4,
        level: 2,
        token_estimate: 1_000,
    }
    .add_to_projected_state(&mut state, "S0123_SENTINEL".to_string());

    assert_eq!(
        state.pairwise_summaries,
        vec![pairwise_summary(0, 4, 2, "S0123_SENTINEL")]
    );

    let result = build_rolling_prompt(&history, &state, params)
        .expect("canonical parent summary should produce an exact-once prompt");

    let prompt = serde_json::to_string(&result.prompt_input).expect("serialize prompt");
    assert!(prompt.contains("S0123_SENTINEL"));
    assert!(!prompt.contains("S01_SENTINEL"));
    assert!(!prompt.contains("S23_SENTINEL"));
    assert!(!prompt.contains("G0_SENTINEL"));
    assert!(!prompt.contains("G1_SENTINEL"));
    assert!(!prompt.contains("G2_SENTINEL"));
    assert!(!prompt.contains("G3_SENTINEL"));
    assert!(prompt.contains("G4_SENTINEL"));
    assert!(prompt.contains("G5_SENTINEL"));
}

#[test]
fn rolling_prompt_pairwise_retry_scale_keeps_existing_summaries() {
    let history = history(vec![
        user_msg("G0_SENTINEL"),
        user_msg("G1_SENTINEL"),
        user_msg("G2_SENTINEL"),
        user_msg("HOT_SENTINEL"),
    ]);
    let input_modalities = default_input_modalities();
    let base_instructions = base_instructions();
    let initial_params = RollingPromptParams {
        input_modalities: &input_modalities,
        base_instructions: &base_instructions,
        invariant_prefix: Vec::new(),
        effective_context_window: Some(10_000),
        reserve_percent: Some(0),
        target_tokens: None,
        target_scale_percent: None,
        tool_output_limit_tokens: 10_000,
        pairwise_compaction: Some(pairwise_params()),
    };
    let state = RollingPromptState {
        history_version: history.history_version(),
        projection_basis_fingerprint: projection_basis_fingerprint(&initial_params, 10_000, &[]),
        pairwise_summaries: vec![pairwise_summary(0, 2, 1, "S01_SENTINEL")],
        ..RollingPromptState::default()
    };

    let retry_outcome = build_rolling_prompt_with_live_summaries(
        &history,
        &state,
        RollingPromptParams {
            target_scale_percent: Some(90),
            ..initial_params
        },
    )
    .expect("retry projection should fit using existing summary");

    let RollingPromptBuildOutcome::Ready(result) = retry_outcome else {
        panic!("retry projection should not request an already generated summary");
    };
    let prompt = serde_json::to_string(&result.prompt_input).expect("serialize prompt");
    assert!(prompt.contains("S01_SENTINEL"));
    assert!(!prompt.contains("G0_SENTINEL"));
    assert!(!prompt.contains("G1_SENTINEL"));
    assert!(prompt.contains("G2_SENTINEL"));
    assert!(prompt.contains("HOT_SENTINEL"));
    assert_eq!(result.backoff_applied, true);
}

#[test]
fn rolling_prompt_pairwise_summary_state_is_projected_until_committed() {
    let history = history(vec![
        user_msg("G0_SENTINEL"),
        user_msg("G1_SENTINEL"),
        user_msg("G2_SENTINEL"),
        user_msg("G3_SENTINEL"),
    ]);
    let mut state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(10_000),
            reserve_percent: Some(0),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: Some(pairwise_params()),
        },
    )
    .expect("pairwise prompt should fit");

    assert_eq!(state.raw_history_start_index, 0);
    assert!(result.raw_history_start_index > 0);
    let raw_history = serde_json::to_string(history.raw_items()).expect("serialize history");
    assert!(!raw_history.contains("rollctx_summary_group"));

    state.commit_projection(&result.projected_state, history.raw_items().len());
    assert_eq!(
        state.raw_history_start_index,
        result.raw_history_start_index
    );
}

#[test]
fn rolling_prompt_pairwise_budget_pressure_compacts_threshold_pair() {
    let old_a = format!("G0_SENTINEL {}", "alpha ".repeat(600));
    let old_b = format!("G1_SENTINEL {}", "beta ".repeat(600));
    let history = history(vec![
        user_msg(&old_a),
        user_msg(&old_b),
        user_msg("HOT_SENTINEL"),
    ]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(10_000),
            reserve_percent: Some(0),
            target_tokens: Some(500),
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: Some(PairwiseRollingPromptParams {
                summary_group_token_cap: 64,
                ..pairwise_params()
            }),
        },
    )
    .expect("budget pressure should compact the eligible threshold pair");

    let prompt = serde_json::to_string(&result.prompt_input).expect("serialize prompt");
    assert!(prompt.contains("raw[0..2)"));
    assert!(!prompt.contains("G0_SENTINEL"));
    assert!(!prompt.contains("G1_SENTINEL"));
    assert!(prompt.contains("HOT_SENTINEL"));
}

#[test]
fn rolling_prompt_pairwise_keeps_tool_followup_and_queued_steer_exact() {
    let history = history(vec![
        user_msg("COLD_G0_SENTINEL"),
        user_msg("COLD_G1_SENTINEL"),
        user_msg("COLD_G2_SENTINEL"),
        function_call("call-1"),
        function_output("call-1", "TOOL_OUTPUT_SENTINEL"),
        user_msg("QUEUED_STEER_SENTINEL"),
    ]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(10_000),
            reserve_percent: Some(0),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
            pairwise_compaction: Some(pairwise_params()),
        },
    )
    .expect("pairwise prompt should fit");

    let prompt = serde_json::to_string(&result.prompt_input).expect("serialize prompt");
    assert!(prompt.contains("raw[0..2)"));
    assert!(!prompt.contains("COLD_G0_SENTINEL"));
    assert!(!prompt.contains("COLD_G1_SENTINEL"));
    assert!(prompt.contains("COLD_G2_SENTINEL"));
    assert!(prompt.contains("call-1"));
    assert!(prompt.contains("TOOL_OUTPUT_SENTINEL"));
    assert!(prompt.contains("QUEUED_STEER_SENTINEL"));
}
