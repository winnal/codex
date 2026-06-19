use super::*;
use codex_analytics::CompactionTrigger;
use codex_protocol::AgentPath;
use codex_protocol::config_types::AutoCompactTokenLimitScope;
use codex_protocol::items::HookPromptFragment;
use codex_protocol::items::build_hook_prompt_message;
use codex_protocol::models::AgentMessageInputContent;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::protocol::APPS_INSTRUCTIONS_OPEN_TAG;
use codex_protocol::protocol::InterAgentCommunication;
use codex_protocol::protocol::PLUGINS_INSTRUCTIONS_OPEN_TAG;
use pretty_assertions::assert_eq;

fn user(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
        metadata: None,
    }
}

fn assistant(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText {
            text: text.to_string(),
        }],
        phase: None,
        metadata: None,
    }
}

fn agent_message(text: &str) -> ResponseItem {
    ResponseItem::AgentMessage {
        id: None,
        author: "/root/worker".to_string(),
        recipient: "/root".to_string(),
        content: vec![AgentMessageInputContent::InputText {
            text: text.to_string(),
        }],
        metadata: None,
    }
}

fn inter_agent_assistant_msg(text: &str) -> ResponseItem {
    let communication = InterAgentCommunication::new(
        AgentPath::root(),
        AgentPath::root().join("worker").unwrap(),
        Vec::new(),
        text.to_string(),
        /*trigger_turn*/ true,
    );
    ResponseItem::Message {
        id: None,
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText {
            text: serde_json::to_string(&communication).unwrap(),
        }],
        phase: None,
        metadata: None,
    }
}

fn compaction_summary(text: &str) -> ResponseItem {
    ResponseItem::Compaction {
        id: None,
        encrypted_content: text.to_string(),
        metadata: None,
    }
}

fn developer(content: Vec<ContentItem>) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "developer".to_string(),
        content,
        phase: None,
        metadata: None,
    }
}

fn function_call(call_id: &str) -> ResponseItem {
    ResponseItem::FunctionCall {
        id: None,
        name: "tool".to_string(),
        namespace: None,
        arguments: "{}".to_string(),
        call_id: call_id.to_string(),
        metadata: None,
    }
}

fn function_output(call_id: &str, output: &str) -> ResponseItem {
    ResponseItem::FunctionCallOutput {
        id: None,
        call_id: call_id.to_string(),
        output: FunctionCallOutputPayload::from_text(output.to_string()),
        metadata: None,
    }
}

fn plan_input<'a>(
    history_items: &'a [ResponseItem],
    target_tokens: i64,
    effective_replacement_budget: Option<i64>,
) -> ExactTailPlanInput<'a> {
    ExactTailPlanInput {
        history_items,
        target_tokens,
        effective_replacement_budget,
        required_current_context_budget: 0,
        final_replacement_extra_budget_tokens: 0,
        max_model_visible_item_tokens: EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS,
        estimated_summary_scaffold_overhead_tokens: local_summary_scaffold_overhead_tokens(),
        retained_cold_user_message_budget_tokens:
            EXACT_TAIL_LOCAL_RETAINED_COLD_USER_MESSAGE_BUDGET_TOKENS,
        implementation: ExactTailImplementation::Local,
    }
}

fn plan(history: Vec<ResponseItem>, target_tokens: i64) -> ExactTailPlan {
    plan_exact_tail(plan_input(&history, target_tokens, Some(50_000)))
        .expect("exact-tail plan should fit")
}

#[test]
fn planner_keeps_newest_group_hot_and_moves_older_groups_to_cold() {
    let old = user(&"old history ".repeat(100));
    let recent = user(&"recent exact tail ".repeat(100));

    let actual = plan(vec![old.clone(), recent.clone()], 1);

    assert_eq!(actual.cold_history, vec![old]);
    assert_eq!(actual.hot_suffix, vec![recent]);
    assert_eq!(
        actual.coverage,
        ExactTailCoverage {
            cold_covered_groups: vec![0],
            hot_exact_groups: vec![1],
            filtered_stale_groups: Vec::new(),
        }
    );
}

#[test]
fn planner_keeps_whole_groups_until_requested_target_is_reached() {
    let old = user("old history");
    let middle = user("middle history");
    let recent = user("recent history");
    let target = estimate_response_items_token_count(std::slice::from_ref(&recent)) + 1;

    let actual = plan(vec![old.clone(), middle.clone(), recent.clone()], target);

    assert_eq!(actual.cold_history, vec![old]);
    assert_eq!(actual.hot_suffix, vec![middle, recent]);
    assert!(actual.diagnostics.actual_hot_tokens >= target);
}

#[test]
fn planner_fails_when_required_whole_group_to_reach_target_cannot_fit() {
    let old = user("old history");
    let middle = user("middle history");
    let recent = user("recent history");
    let recent_tokens = estimate_response_items_token_count(std::slice::from_ref(&recent));
    let middle_and_recent_tokens =
        estimate_response_items_token_count(&[middle.clone(), recent.clone()]);
    let history = [old, middle, recent];
    let error = plan_exact_tail(ExactTailPlanInput {
        history_items: &history,
        target_tokens: recent_tokens + 1,
        effective_replacement_budget: Some(
            middle_and_recent_tokens
                + EXACT_TAIL_CONSERVATIVE_SUMMARY_BUDGET_TOKENS
                + EXACT_TAIL_MIN_SAFETY_MARGIN_TOKENS
                - 1,
        ),
        required_current_context_budget: 0,
        final_replacement_extra_budget_tokens: 0,
        max_model_visible_item_tokens: EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: 0,
        implementation: ExactTailImplementation::Local,
    })
    .expect_err("required middle group should not fit");

    assert_eq!(error.reason, ExactTailFailReason::MinimumHotSuffixTooLarge);
}

#[test]
fn planner_preserves_visible_hook_prompt_in_hot_suffix() {
    let old = user("old history");
    let hook = build_hook_prompt_message(&[HookPromptFragment::from_single_hook(
        "retry with the exact recent context",
        "hook-run-1",
    )])
    .expect("hook prompt message");

    let actual = plan(vec![old.clone(), hook.clone()], 1);

    assert_eq!(actual.cold_history, vec![old]);
    assert_eq!(actual.hot_suffix, vec![hook]);
    assert_eq!(actual.diagnostics.filtered_context_item_count, 0);
}

#[test]
fn planner_preserves_agent_message_in_its_own_hot_group() {
    let old_user = user("old history");
    let old_assistant = assistant("old answer");
    let delegated_instruction = agent_message("continue with this delegated task exactly");

    let actual = plan(
        vec![
            old_user.clone(),
            old_assistant.clone(),
            delegated_instruction.clone(),
        ],
        1,
    );

    assert_eq!(actual.cold_history, vec![old_user, old_assistant]);
    assert_eq!(actual.hot_suffix, vec![delegated_instruction]);
    assert_eq!(actual.diagnostics.hot_group_count, 1);
}

#[test]
fn planner_preserves_inter_agent_assistant_instruction_in_its_own_hot_group() {
    let old_user = user("old history");
    let old_assistant = assistant("old answer");
    let delegated_instruction = inter_agent_assistant_msg("continue from another agent");

    let actual = plan(
        vec![
            old_user.clone(),
            old_assistant.clone(),
            delegated_instruction.clone(),
        ],
        1,
    );

    assert_eq!(actual.cold_history, vec![old_user, old_assistant]);
    assert_eq!(actual.hot_suffix, vec![delegated_instruction]);
    assert_eq!(actual.diagnostics.hot_group_count, 1);
}

#[test]
fn planner_preserves_function_call_and_output_as_one_hot_group() {
    let old = user(&"old history ".repeat(100));
    let recent = user(&"recent tool turn ".repeat(100));
    let call = function_call("call-1");
    let output = function_output("call-1", &"tool result ".repeat(100));

    let actual = plan(
        vec![old.clone(), recent.clone(), call.clone(), output.clone()],
        1,
    );

    assert_eq!(actual.cold_history, vec![old]);
    assert_eq!(actual.hot_suffix, vec![recent, call, output]);
    assert_eq!(actual.diagnostics.hot_group_count, 1);
}

#[test]
fn planner_uses_summary_scaffold_overhead_in_available_hot_budget() {
    let overhead = 777;
    let history = [user("old"), user("recent")];
    let actual = plan_exact_tail(ExactTailPlanInput {
        history_items: &history,
        target_tokens: 1,
        effective_replacement_budget: Some(50_000),
        required_current_context_budget: 123,
        final_replacement_extra_budget_tokens: 456,
        max_model_visible_item_tokens: EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS,
        estimated_summary_scaffold_overhead_tokens: overhead,
        retained_cold_user_message_budget_tokens: 321,
        implementation: ExactTailImplementation::RemoteLegacy,
    })
    .expect("exact-tail plan should fit");

    assert_eq!(
        actual
            .diagnostics
            .estimated_summary_scaffold_overhead_tokens,
        overhead
    );
    assert_eq!(
        actual.diagnostics.retained_cold_user_message_budget_tokens,
        321
    );
    assert_eq!(
        actual.diagnostics.final_replacement_extra_budget_tokens,
        456
    );
    assert_eq!(
        actual.diagnostics.conservative_cold_summary_budget,
        EXACT_TAIL_CONSERVATIVE_SUMMARY_BUDGET_TOKENS
            + overhead
            + 321
            + EXACT_TAIL_REPLACEMENT_OVERHEAD_MARGIN_TOKENS
    );
    assert_eq!(
        actual.diagnostics.available_for_hot,
        50_000
            - 123
            - actual.diagnostics.conservative_cold_summary_budget
            - actual.diagnostics.safety_margin
    );
}

#[test]
fn minimum_hot_suffix_oversize_fails_before_compaction() {
    let history = [user("old"), user("recent")];
    let actual = plan_exact_tail(ExactTailPlanInput {
        history_items: &history,
        target_tokens: 10_000,
        effective_replacement_budget: Some(1),
        required_current_context_budget: 0,
        final_replacement_extra_budget_tokens: 0,
        max_model_visible_item_tokens: EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: 0,
        implementation: ExactTailImplementation::Local,
    })
    .expect_err("minimum hot suffix cannot fit");

    assert_eq!(actual.reason, ExactTailFailReason::MinimumHotSuffixTooLarge);
}

#[test]
fn all_hot_history_without_cold_prefix_fails_before_compaction() {
    let history = [user("recent only")];
    let actual = plan_exact_tail(ExactTailPlanInput {
        history_items: &history,
        target_tokens: 10_000,
        effective_replacement_budget: Some(50_000),
        required_current_context_budget: 0,
        final_replacement_extra_budget_tokens: 0,
        max_model_visible_item_tokens: EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: 0,
        implementation: ExactTailImplementation::Local,
    })
    .expect_err("exact-tail compaction needs cold history to summarize");

    assert_eq!(actual.reason, ExactTailFailReason::NoColdPrefix);
}

#[test]
fn unavailable_replacement_budget_fails_before_compaction() {
    let history = [user("old"), user("recent")];
    let actual = plan_exact_tail(ExactTailPlanInput {
        history_items: &history,
        target_tokens: 1,
        effective_replacement_budget: None,
        required_current_context_budget: 0,
        final_replacement_extra_budget_tokens: 0,
        max_model_visible_item_tokens: EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: 0,
        implementation: ExactTailImplementation::Local,
    })
    .expect_err("exact-tail requires a replacement budget");

    assert_eq!(actual.reason, ExactTailFailReason::BudgetUnavailable);
}

#[test]
fn planner_filters_stale_contextual_developer_wrapper() {
    let old = user("old history");
    let stale_context = developer(vec![ContentItem::InputText {
        text: "<token_budget>\n1000 tokens remain\n</token_budget>".to_string(),
    }]);
    let recent = user("recent history");

    let actual = plan(vec![old.clone(), stale_context, recent.clone()], 1);

    assert_eq!(actual.cold_history, vec![old]);
    assert_eq!(actual.hot_suffix, vec![recent]);
    assert_eq!(actual.diagnostics.filtered_context_item_count, 1);
    assert_eq!(actual.coverage.filtered_stale_groups, vec![1]);
}

#[test]
fn planner_filters_stale_app_and_plugin_developer_wrappers() {
    let old = user("old history");
    let stale_apps_context = developer(vec![ContentItem::InputText {
        text: format!("{APPS_INSTRUCTIONS_OPEN_TAG}\n## Apps (Connectors)"),
    }]);
    let stale_plugins_context = developer(vec![ContentItem::InputText {
        text: format!("{PLUGINS_INSTRUCTIONS_OPEN_TAG}\n## Plugins"),
    }]);
    let recent = user("recent history");

    let actual = plan(
        vec![
            old.clone(),
            stale_apps_context,
            stale_plugins_context,
            recent.clone(),
        ],
        1,
    );

    assert_eq!(actual.cold_history, vec![old]);
    assert_eq!(actual.hot_suffix, vec![recent]);
    assert_eq!(actual.diagnostics.filtered_context_item_count, 2);
    assert_eq!(actual.coverage.filtered_stale_groups, vec![1, 2]);
}

#[test]
fn replacement_fit_ignores_oversize_stale_context_wrapper_item() {
    let actual = plan(vec![user("old history"), user("recent history")], 1);
    let stale_context = developer(vec![ContentItem::InputText {
        text: format!("<token_budget>\n{}\n</token_budget>", "ctx ".repeat(15_000)),
    }]);
    let stale_context_tokens =
        estimate_response_items_token_count(std::slice::from_ref(&stale_context));
    assert!(stale_context_tokens > EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS);

    let final_tokens =
        check_replacement_fits(&actual, &[stale_context], /*actual_summary_tokens*/ 0)
            .expect("stale context wrapper should not fail the exact-tail item cap");

    assert!(final_tokens < actual.diagnostics.effective_replacement_budget);
}

#[test]
fn planner_filters_mixed_app_plugin_and_persistent_developer_message() {
    let old = user("old history");
    let mixed_developer = developer(vec![
        ContentItem::InputText {
            text: format!("{APPS_INSTRUCTIONS_OPEN_TAG}\n## Apps (Connectors)"),
        },
        ContentItem::InputText {
            text: format!("{PLUGINS_INSTRUCTIONS_OPEN_TAG}\n## Plugins"),
        },
        ContentItem::InputText {
            text: "Persistent developer instruction must not be silently dropped.".to_string(),
        },
    ]);
    let recent = user("recent history");

    let actual = plan(vec![old.clone(), mixed_developer, recent.clone()], 1);

    assert_eq!(actual.cold_history, vec![old]);
    assert_eq!(actual.hot_suffix, vec![recent]);
    assert_eq!(actual.diagnostics.filtered_context_item_count, 1);
    assert_eq!(
        actual.coverage.filtered_stale_groups,
        vec![1],
        "old mixed initial-context bundles should be superseded by freshly built current context"
    );
}

#[test]
fn planner_filters_oversize_mixed_developer_context_bundle() {
    let old = user("old history");
    let mixed_developer = developer(vec![
        ContentItem::InputText {
            text: format!("<token_budget>\n{}\n</token_budget>", "ctx ".repeat(15_000)),
        },
        ContentItem::InputText {
            text: "Persistent developer instruction must not be silently dropped.".to_string(),
        },
    ]);
    let recent = user("recent history");
    let mixed_developer_tokens =
        estimate_response_items_token_count(std::slice::from_ref(&mixed_developer));
    assert!(mixed_developer_tokens > EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS);

    let actual = plan(vec![old.clone(), mixed_developer, recent.clone()], 1);

    assert_eq!(actual.cold_history, vec![old]);
    assert_eq!(actual.hot_suffix, vec![recent]);
    assert_eq!(actual.diagnostics.filtered_context_item_count, 1);
}

#[test]
fn oversize_hot_user_item_fails_closed() {
    let history = [user("old"), user(&"hot user ".repeat(40_000))];
    let actual = plan_exact_tail(ExactTailPlanInput {
        history_items: &history,
        target_tokens: 1,
        effective_replacement_budget: Some(200_000),
        required_current_context_budget: 0,
        final_replacement_extra_budget_tokens: 0,
        max_model_visible_item_tokens: EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: 0,
        implementation: ExactTailImplementation::Local,
    })
    .expect_err("hot user item should exceed the per-item cap");

    assert_eq!(actual.reason, ExactTailFailReason::ModelVisibleItemTooLarge);
}

#[test]
fn oversize_hot_developer_item_fails_closed() {
    let history = [
        user("old"),
        user("recent"),
        developer(vec![ContentItem::InputText {
            text: "hot developer ".repeat(40_000),
        }]),
    ];
    let actual = plan_exact_tail(ExactTailPlanInput {
        history_items: &history,
        target_tokens: 1,
        effective_replacement_budget: Some(200_000),
        required_current_context_budget: 0,
        final_replacement_extra_budget_tokens: 0,
        max_model_visible_item_tokens: EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: 0,
        implementation: ExactTailImplementation::Local,
    })
    .expect_err("hot developer item should exceed the per-item cap");

    assert_eq!(actual.reason, ExactTailFailReason::ModelVisibleItemTooLarge);
}

#[test]
fn oversize_hot_assistant_item_fails_closed() {
    let history = [
        user("old"),
        user("recent"),
        assistant(&"hot assistant ".repeat(40_000)),
    ];
    let actual = plan_exact_tail(ExactTailPlanInput {
        history_items: &history,
        target_tokens: 1,
        effective_replacement_budget: Some(200_000),
        required_current_context_budget: 0,
        final_replacement_extra_budget_tokens: 0,
        max_model_visible_item_tokens: EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: 0,
        implementation: ExactTailImplementation::Local,
    })
    .expect_err("hot assistant item should exceed the per-item cap");

    assert_eq!(actual.reason, ExactTailFailReason::ModelVisibleItemTooLarge);
}

#[test]
fn oversize_hot_tool_output_item_fails_closed() {
    let history = [
        user("old"),
        user("recent"),
        function_call("call-1"),
        function_output("call-1", &"hot tool output ".repeat(40_000)),
    ];
    let actual = plan_exact_tail(ExactTailPlanInput {
        history_items: &history,
        target_tokens: 1,
        effective_replacement_budget: Some(200_000),
        required_current_context_budget: 0,
        final_replacement_extra_budget_tokens: 0,
        max_model_visible_item_tokens: EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: 0,
        implementation: ExactTailImplementation::Local,
    })
    .expect_err("hot tool output should exceed the per-item cap");

    assert_eq!(actual.reason, ExactTailFailReason::ModelVisibleItemTooLarge);
    assert!(
        actual
            .into_codex_err()
            .to_string()
            .contains("function_call_output call_id=call-1"),
        "error should identify the oversized item"
    );
}

#[test]
fn configured_larger_item_cap_allows_read_thread_sized_tool_output() {
    let output = function_output("call-1", &"read_thread summary payload ".repeat(2_400));
    let output_tokens = estimate_response_items_token_count(std::slice::from_ref(&output));
    assert!(output_tokens > EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS);
    assert!(output_tokens < 20_000);
    let history = [
        user("old"),
        user("recent"),
        function_call("call-1"),
        output.clone(),
    ];

    let default_error = plan_exact_tail(ExactTailPlanInput {
        history_items: &history,
        target_tokens: 1,
        effective_replacement_budget: Some(200_000),
        required_current_context_budget: 0,
        final_replacement_extra_budget_tokens: 0,
        max_model_visible_item_tokens: EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: 0,
        implementation: ExactTailImplementation::Local,
    })
    .expect_err("default item cap should reject the read_thread-sized tool output");
    assert_eq!(
        default_error.reason,
        ExactTailFailReason::ModelVisibleItemTooLarge
    );

    let actual = plan_exact_tail(ExactTailPlanInput {
        history_items: &history,
        target_tokens: 1,
        effective_replacement_budget: Some(200_000),
        required_current_context_budget: 0,
        final_replacement_extra_budget_tokens: 0,
        max_model_visible_item_tokens: 20_000,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: 0,
        implementation: ExactTailImplementation::Local,
    })
    .expect("configured 20k item cap should allow the read_thread-sized tool output");

    assert_eq!(
        actual.hot_suffix,
        vec![user("recent"), function_call("call-1"), output]
    );
}

#[test]
fn cold_input_too_large_fails_closed_without_pruning() {
    let actual = plan(vec![user("old"), user("recent")], 1);
    let error = check_cold_input_fits(
        &actual,
        &[user(&"oversized cold input ".repeat(10))],
        Some(1),
    )
    .expect_err("oversized cold input should fail closed");

    assert_eq!(error.reason, ExactTailFailReason::ColdInputTooLarge);
}

#[test]
fn cold_input_oversize_item_fails_closed_without_requesting_compaction() {
    let actual = plan(vec![user("old"), user("recent")], 1);
    let error = check_cold_input_fits(
        &actual,
        &[user(&"oversized cold input item ".repeat(40_000))],
        Some(200_000),
    )
    .expect_err("oversized cold input item should fail the per-item cap");

    assert_eq!(error.reason, ExactTailFailReason::ModelVisibleItemTooLarge);
}

#[test]
fn replacement_fit_includes_final_prompt_reserve() {
    let replacement_history = (0..2_000)
        .map(|index| user(&format!("summary fragment {index}")))
        .collect::<Vec<_>>();
    let replacement_tokens = estimate_response_items_token_count(&replacement_history);
    let final_extra_tokens = 1_000;
    let effective_budget = replacement_tokens + final_extra_tokens - 1;
    let history = [user("old"), user("recent")];
    let actual = plan_exact_tail(ExactTailPlanInput {
        history_items: &history,
        target_tokens: 1,
        effective_replacement_budget: Some(effective_budget),
        required_current_context_budget: 0,
        final_replacement_extra_budget_tokens: final_extra_tokens,
        max_model_visible_item_tokens: EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: 0,
        implementation: ExactTailImplementation::Local,
    })
    .expect("exact-tail plan should fit before final replacement reserve is applied");
    assert!(replacement_tokens <= effective_budget);

    let error = check_replacement_fits(
        &actual,
        &replacement_history,
        /*actual_summary_tokens*/ replacement_tokens,
    )
    .expect_err("final prompt reserve should make the replacement exceed the budget");

    assert_eq!(error.reason, ExactTailFailReason::ReplacementTooLarge);
}

#[test]
fn replacement_fit_rejects_oversize_model_visible_item_even_when_total_fits() {
    let replacement_history = vec![user(&"oversized replacement item ".repeat(40_000))];
    let replacement_tokens = estimate_response_items_token_count(&replacement_history);
    let history = [user("old"), user("recent")];
    let actual = plan_exact_tail(ExactTailPlanInput {
        history_items: &history,
        target_tokens: 1,
        effective_replacement_budget: Some(replacement_tokens + 100_000),
        required_current_context_budget: 0,
        final_replacement_extra_budget_tokens: 0,
        max_model_visible_item_tokens: EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: 0,
        implementation: ExactTailImplementation::Local,
    })
    .expect("exact-tail plan should fit before final replacement item cap is applied");

    let error = check_replacement_fits(
        &actual,
        &replacement_history,
        /*actual_summary_tokens*/ replacement_tokens,
    )
    .expect_err("oversize replacement item should fail the model-visible item cap");

    assert_eq!(error.reason, ExactTailFailReason::ModelVisibleItemTooLarge);
}

#[test]
fn post_summary_oversize_fails_without_dropping_hot_suffix() {
    let old = user(&"old ".repeat(100));
    let recent = user(&"recent ".repeat(100));
    let actual = plan(vec![old, recent.clone()], 1);
    let oversized_replacement = vec![user(&"oversized summary ".repeat(20_000))];

    let error = check_replacement_fits(
        &actual,
        &oversized_replacement,
        /*actual_summary_tokens*/ 100_000,
    )
    .expect_err("replacement should exceed fit budget");

    assert_eq!(error.reason, ExactTailFailReason::ModelVisibleItemTooLarge);
    assert_eq!(actual.hot_suffix, vec![recent]);
}

#[test]
fn empty_local_summary_fails_closed() {
    let actual = plan(vec![user("old"), user("recent")], 1);
    let error =
        ensure_non_empty_local_summary(&actual, "  \n").expect_err("empty summary should fail");

    assert_eq!(error.reason, ExactTailFailReason::NoUsableColdSummary);
}

#[test]
fn empty_remote_replacement_fails_closed() {
    let actual = plan(vec![user("old"), user("recent")], 1);
    let error = ensure_replacement_has_cold_summary(&actual, &[])
        .expect_err("empty remote replacement should fail");

    assert_eq!(error.reason, ExactTailFailReason::NoUsableColdSummary);
}

#[test]
fn non_empty_remote_replacement_is_usable_cold_summary() {
    let actual = plan(vec![user("old"), user("recent")], 1);

    ensure_replacement_has_cold_summary(&actual, &[compaction_summary("summary")])
        .expect("summary text should be usable");
}

#[test]
fn retained_remote_user_text_is_not_a_usable_cold_summary() {
    let actual = plan(vec![user("old"), user("recent")], 1);
    let error = ensure_replacement_has_cold_summary(&actual, &[user("retained cold text")])
        .expect_err("retained user text is not a summary-bearing remote compact output");

    assert_eq!(error.reason, ExactTailFailReason::NoUsableColdSummary);
}

#[test]
fn mid_turn_replacement_inserts_initial_context_before_last_protected_user_group() {
    let summary = user(&format!("{SUMMARY_PREFIX}\ncold summary"));
    let initial_context = developer(vec![ContentItem::InputText {
        text: "<token_budget>\n900 tokens remain\n</token_budget>".to_string(),
    }]);
    let first_hot_user = user("first protected hot user");
    let first_hot_assistant = assistant("first protected hot assistant");
    let second_hot_user = user("second protected hot user");
    let second_hot_assistant = assistant("second protected hot assistant");

    let actual = append_hot_suffix_to_replacement(
        vec![summary.clone()],
        vec![initial_context.clone()],
        vec![
            first_hot_user.clone(),
            first_hot_assistant.clone(),
            second_hot_user.clone(),
            second_hot_assistant.clone(),
        ],
        InitialContextInjection::BeforeLastUserMessage,
    );

    assert_eq!(
        actual,
        vec![
            summary,
            first_hot_user,
            first_hot_assistant,
            initial_context,
            second_hot_user,
            second_hot_assistant,
        ]
    );
}

#[test]
fn mid_turn_replacement_inserts_initial_context_before_agent_message_boundary() {
    let summary = user(&format!("{SUMMARY_PREFIX}\ncold summary"));
    let initial_context = developer(vec![ContentItem::InputText {
        text: "<token_budget>\n900 tokens remain\n</token_budget>".to_string(),
    }]);
    let delegated_instruction = agent_message("continue from exact hot suffix");

    let actual = append_hot_suffix_to_replacement(
        vec![summary.clone()],
        vec![initial_context.clone()],
        vec![delegated_instruction.clone()],
        InitialContextInjection::BeforeLastUserMessage,
    );

    assert_eq!(
        actual,
        vec![summary, initial_context, delegated_instruction]
    );
}

#[test]
fn mid_turn_replacement_inserts_initial_context_before_inter_agent_instruction_boundary() {
    let summary = user(&format!("{SUMMARY_PREFIX}\ncold summary"));
    let initial_context = developer(vec![ContentItem::InputText {
        text: "<token_budget>\n900 tokens remain\n</token_budget>".to_string(),
    }]);
    let delegated_instruction = inter_agent_assistant_msg("continue from exact hot suffix");

    let actual = append_hot_suffix_to_replacement(
        vec![summary.clone()],
        vec![initial_context.clone()],
        vec![delegated_instruction.clone()],
        InitialContextInjection::BeforeLastUserMessage,
    );

    assert_eq!(
        actual,
        vec![summary, initial_context, delegated_instruction]
    );
}

#[test]
fn body_after_prefix_auto_replacement_budget_uses_full_context_window() {
    let actual = exact_tail_replacement_budget(
        Some(200_000),
        Some(100),
        AutoCompactTokenLimitScope::BodyAfterPrefix,
        CompactionTrigger::Auto,
    );

    assert_eq!(actual, Some(200_000));
}

#[test]
fn total_scope_auto_replacement_budget_clamps_to_auto_limit() {
    let actual = exact_tail_replacement_budget(
        Some(200_000),
        Some(100),
        AutoCompactTokenLimitScope::Total,
        CompactionTrigger::Auto,
    );

    assert_eq!(actual, Some(100));
}

#[test]
fn remote_legacy_summary_scaffold_overhead_is_zero() {
    assert_eq!(remote_legacy_summary_scaffold_overhead_tokens(), 0);
}
