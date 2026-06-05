use super::test_support::*;
use super::*;
use crate::context::ContextualUserFragment;
use crate::context::ExtensionContextualUserFragment;
use crate::context_manager::updates::is_rolling_invariant_developer_content;
use crate::event_mapping::is_contextual_dev_message_content;
use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::openai_models::default_input_modalities;
use pretty_assertions::assert_eq;

#[test]
fn rolling_prompt_filters_historical_extension_contextual_user_prefix() {
    let old_extension_context =
        user_msg(&ExtensionContextualUserFragment::new("STALE_EXTENSION_CONTEXT").render());
    let current_extension_context =
        user_msg(&ExtensionContextualUserFragment::new("CURRENT_EXTENSION_CONTEXT").render());
    let history = history(vec![
        old_extension_context,
        user_msg(&"old body ".repeat(1_000)),
        user_msg("NEWEST_BODY"),
    ]);
    let state = RollingPromptState::default();
    let current_prefix = vec![current_extension_context];

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: current_prefix,
            effective_context_window: Some(1_000),
            reserve_percent: Some(0),
            target_tokens: Some(500),
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
        },
    )
    .expect("current extension context plus newest body should fit");

    let prompt_text = serde_json::to_string(&result.prompt_input).expect("serialize prompt");
    assert!(
        prompt_text.contains("CURRENT_EXTENSION_CONTEXT"),
        "current extension contextual-user prefix should be retained"
    );
    assert!(
        !prompt_text.contains("STALE_EXTENSION_CONTEXT"),
        "stale extension contextual-user prefix should be filtered"
    );
    assert!(
        prompt_text.contains("NEWEST_BODY"),
        "newest body should remain"
    );
}

#[test]
fn rolling_invariant_developer_marker_is_contextual_dev_fragment() {
    let ResponseItem::Message { content, .. } = rolling_invariant_developer_msg("CURRENT_DEV")
    else {
        panic!("fixture should build a developer message");
    };

    assert!(is_rolling_invariant_developer_content(&content));
    assert!(is_contextual_dev_message_content(&content));
}

#[test]
fn rolling_prompt_does_not_reinclude_generated_context_before_cursor() {
    let prelude = developer_msg("TURN_PRELUDE_CONTEXT");
    let user = user_msg("USER_PROMPT_SENTINEL");
    let newest = assistant_msg("NEWER_SENTINEL");
    let history = history(vec![prelude, user, newest.clone()]);
    let state = RollingPromptState {
        history_version: history.history_version(),
        raw_history_start_index: 1,
        projection_basis_fingerprint: 0,
    };

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
        },
    )
    .expect("newer suffix should fit without rehydrating pre-cursor context");

    assert_eq!(result.prompt_input, vec![newest]);
}

#[test]
fn rolling_prompt_keeps_turn_scoped_developer_context_as_body() {
    let turn_scoped_developer = developer_msg("hook additional context for this turn only");
    let history = history(vec![
        developer_msg("<permissions instructions>\nstale"),
        user_msg(&"old ".repeat(1_000)),
        turn_scoped_developer.clone(),
        user_msg("new"),
    ]);
    let state = RollingPromptState::default();
    let current_prefix = vec![developer_msg("<permissions instructions>\ncurrent")];

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: current_prefix.clone(),
            effective_context_window: Some(1_000),
            reserve_percent: Some(0),
            target_tokens: Some(500),
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
        },
    )
    .expect("turn-scoped developer context should fit");

    assert_eq!(
        result.prompt_input,
        vec![
            current_prefix[0].clone(),
            turn_scoped_developer,
            user_msg("new")
        ]
    );
}

#[test]
fn rolling_prompt_filters_legacy_unmarked_developer_preamble_when_current_dev_changes() {
    let history = history(vec![
        developer_msg("OLD_DEVELOPER_PREFIX"),
        user_msg("OLD_BODY"),
        user_msg("NEWEST_BODY"),
    ]);
    let state = RollingPromptState::default();
    let current_prefix = vec![rolling_invariant_developer_msg("NEW_DEVELOPER_PREFIX")];

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: current_prefix.clone(),
            effective_context_window: Some(10_000),
            reserve_percent: Some(0),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
        },
    )
    .expect("current prefix and body should fit");

    assert_eq!(
        result.prompt_input,
        vec![
            current_prefix[0].clone(),
            user_msg("OLD_BODY"),
            user_msg("NEWEST_BODY")
        ]
    );
}

#[test]
fn rolling_prompt_filters_unmarked_historical_developer_prefix_matching_current_invariant() {
    let old_compact_setup = developer_msg("EXTENSION_DEVELOPER_SETUP");
    let current_prefix = vec![rolling_invariant_developer_msg("EXTENSION_DEVELOPER_SETUP")];
    let history = history(vec![
        old_compact_setup,
        user_msg(&"old ".repeat(1_000)),
        user_msg("NEWEST_BODY"),
    ]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: current_prefix.clone(),
            effective_context_window: Some(1_000),
            reserve_percent: Some(0),
            target_tokens: Some(500),
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
        },
    )
    .expect("current prefix and newest body should fit");

    assert_eq!(
        result.prompt_input,
        vec![current_prefix[0].clone(), user_msg("NEWEST_BODY")]
    );
}

#[test]
fn rolling_prompt_filters_large_unmarked_historical_developer_prefix_before_projection() {
    let developer_setup = "LARGE_EXTENSION_DEVELOPER_SETUP ".repeat(8_000);
    let current_prefix = vec![rolling_invariant_developer_msg(&developer_setup)];
    let projected_current_prefix = current_prefix
        .clone()
        .into_iter()
        .flat_map(|item| project_rolling_message_item(item, MAX_ROLLING_PROMPT_ITEM_TOKENS))
        .collect::<Vec<_>>();
    assert!(
        projected_current_prefix.len() > 1,
        "fixture should force current invariant developer prefix projection"
    );
    let history = history(vec![
        developer_msg(&developer_setup),
        user_msg(&"old ".repeat(1_000)),
        user_msg("NEWEST_BODY"),
    ]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: current_prefix,
            effective_context_window: Some(100_000),
            reserve_percent: Some(0),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
        },
    )
    .expect("current split prefix and newest body should fit");

    assert_eq!(
        &result.prompt_input[..projected_current_prefix.len()],
        projected_current_prefix.as_slice()
    );
    let developer_message_count = result
        .prompt_input
        .iter()
        .filter(|item| matches!(item, ResponseItem::Message { role, .. } if role == "developer"))
        .count();
    assert_eq!(
        developer_message_count,
        projected_current_prefix.len(),
        "historical unmarked developer prefix must not remain after the projected current prefix"
    );
    assert!(
        result.prompt_input.contains(&user_msg("NEWEST_BODY")),
        "newest body should remain"
    );
}

#[test]
fn rolling_prompt_rejects_encrypted_reasoning_when_summary_exceeds_item_cap() {
    let reasoning = ResponseItem::Reasoning {
        id: String::new(),
        summary: vec![ReasoningItemReasoningSummary::SummaryText {
            text: "ROLLCTX_REASONING_SUMMARY ".repeat(30_000),
        }],
        content: None,
        encrypted_content: Some("A".repeat(1_868)),
    };
    let history = history(vec![reasoning]);
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
            target_tokens: Some(50_000),
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
        },
    )
    .expect_err("oversized pass-through reasoning should not bypass the per-item cap");

    assert!(matches!(error, RollingPromptError::ItemExceedsLimit { .. }));
}

#[test]
fn rolling_prompt_keeps_turn_scoped_skill_context_as_body() {
    let skill_context =
        user_msg("<skill>\n<name>demo</name>\n<path>skills/demo/SKILL.md</path>\nbody\n</skill>");
    let history = history(vec![
        contextual_user_msg("<environment_context>\nstale\n</environment_context>"),
        user_msg(&"old ".repeat(1_000)),
        user_msg("new"),
        skill_context.clone(),
    ]);
    let state = RollingPromptState::default();
    let current_prefix = vec![contextual_user_msg(
        "<environment_context>\ncurrent\n</environment_context>",
    )];

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: current_prefix.clone(),
            effective_context_window: Some(1_000),
            reserve_percent: Some(0),
            target_tokens: Some(500),
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
        },
    )
    .expect("turn-scoped skill context should fit");

    assert_eq!(
        result.prompt_input,
        vec![current_prefix[0].clone(), user_msg("new"), skill_context,]
    );
}

#[test]
fn rolling_prompt_filters_marked_historical_raw_developer_prefix() {
    let history = history(vec![
        rolling_invariant_developer_msg("STALE_RAW_DEVELOPER_SENTINEL"),
        user_msg("NEWEST_BODY"),
    ]);
    let state = RollingPromptState::default();
    let current_prefix = vec![rolling_invariant_developer_msg(
        "CURRENT_RAW_DEVELOPER_SENTINEL",
    )];

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: current_prefix,
            effective_context_window: Some(2_000),
            reserve_percent: Some(0),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
        },
    )
    .expect("current developer prefix and newest body should fit");

    let prompt = serde_json::to_string(&result.prompt_input).expect("serialize prompt");
    assert!(prompt.contains("CURRENT_RAW_DEVELOPER_SENTINEL"));
    assert!(
        !prompt.contains("STALE_RAW_DEVELOPER_SENTINEL"),
        "stale marked generated developer prefix must not remain as body"
    );
}

#[test]
fn rolling_prompt_keeps_post_user_generated_context_atomic_with_user_prompt() {
    let history = history(vec![
        user_msg("USER_PROMPT_SENTINEL"),
        contextual_user_msg(&format!(
            "<skill>\n{}\n</skill>",
            "POST_USER_SKILL_CONTEXT ".repeat(120)
        )),
    ]);
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
        },
    )
    .expect_err("post-user generated context and user prompt should fail as one newest frontier");

    assert!(matches!(
        error,
        RollingPromptError::FrontierExceedsBudget { .. }
    ));
}

#[test]
fn rolling_prompt_splits_generated_context_message_under_item_cap() {
    let history = history(vec![
        user_msg("USER_WITH_LARGE_SKILL_CONTEXT"),
        contextual_user_msg(&format!(
            "<skill>\n{}\n</skill>",
            "LARGE_SKILL_CONTEXT ".repeat(8_000)
        )),
    ]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(80_000),
            reserve_percent: Some(0),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
        },
    )
    .expect("generated turn context should split into model-visible bounded items");

    assert!(
        result
            .prompt_input
            .iter()
            .all(|item| item_token_estimate(item) <= MAX_ROLLING_PROMPT_ITEM_TOKENS),
        "all split generated context items must respect the per-item cap"
    );
    assert!(
        serde_json::to_string(&result.prompt_input)
            .expect("serialize prompt")
            .contains("LARGE_SKILL_CONTEXT"),
        "split prompt should preserve generated context text"
    );
}

#[test]
fn rolling_prompt_keeps_multiple_outputs_for_one_custom_tool_call() {
    let history = history(vec![
        custom_tool_call("custom-call-1"),
        custom_tool_output("custom-call-1", "CUSTOM_NOTIFY_OUTPUT"),
        custom_tool_output("custom-call-1", "CUSTOM_FINAL_OUTPUT"),
    ]);
    let state = RollingPromptState::default();

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(2_000),
            reserve_percent: Some(0),
            target_tokens: None,
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
        },
    )
    .expect("custom call with notify and final outputs should fit");

    assert_eq!(
        result.prompt_input,
        vec![
            custom_tool_call("custom-call-1"),
            custom_tool_output("custom-call-1", "CUSTOM_NOTIFY_OUTPUT"),
            custom_tool_output("custom-call-1", "CUSTOM_FINAL_OUTPUT"),
        ]
    );
}

#[test]
fn rolling_prompt_budgets_actual_custom_tool_output_size_used_by_code_mode() {
    let code_mode_output = "x".repeat(20_000);
    let output_item = custom_tool_output("call-1", &code_mode_output);
    let newest = user_msg("new frontier");
    let history = history(vec![
        custom_tool_call("call-1"),
        output_item.clone(),
        newest.clone(),
    ]);
    let state = RollingPromptState::default();

    assert!(
        item_token_estimate(&output_item) > 2_000,
        "fixture should exceed the rolling body budget"
    );
    assert!(
        item_token_estimate(&output_item) < MAX_ROLLING_PROMPT_ITEM_TOKENS,
        "fixture should prove budget accounting without tripping the per-item cap"
    );

    let result = build_rolling_prompt(
        &history,
        &state,
        RollingPromptParams {
            input_modalities: &default_input_modalities(),
            base_instructions: &base_instructions(),
            invariant_prefix: Vec::new(),
            effective_context_window: Some(100_000),
            reserve_percent: Some(0),
            target_tokens: Some(2_000),
            target_scale_percent: None,
            tool_output_limit_tokens: 10_000,
        },
    )
    .expect("rolling prompt should eject oversized old custom output and keep frontier");

    assert_eq!(result.prompt_input, vec![newest]);
    assert_eq!(result.raw_history_start_index, 2);
    assert_eq!(result.dropped_body_items, 2);
}

#[test]
fn rolling_prompt_drops_old_pass_through_items_that_exceed_per_item_limit() {
    for oversized in oversized_pass_through_items() {
        assert!(
            item_token_estimate(&oversized) > MAX_ROLLING_PROMPT_ITEM_TOKENS,
            "fixture should exceed the model-visible item cap: {oversized:?}"
        );
        let newest = user_msg("newest body");
        let history = history(vec![oversized, newest.clone()]);
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
            },
        )
        .expect("oversized old pass-through item should be droppable");

        assert_eq!(result.prompt_input, vec![newest]);
    }
}

#[test]
fn rolling_prompt_errors_when_newest_pass_through_item_exceeds_per_item_limit() {
    for oversized in oversized_pass_through_items() {
        let history = history(vec![oversized]);
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
            },
        )
        .expect_err("oversized newest pass-through item should fail clearly");

        assert!(matches!(error, RollingPromptError::ItemExceedsLimit { .. }));
    }
}
