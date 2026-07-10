use super::*;
use crate::context::world_state::WorldState;
use crate::context_manager::ContextManager;
use crate::context_manager::model_visible_tool_output_item_token_limit;
use crate::tools::exact_tail_continuity::derive_exact_tail_tool_surface_hint;
use codex_analytics::CompactionTrigger;
use codex_protocol::AgentPath;
use codex_protocol::config_types::AutoCompactTokenLimitScope;
use codex_protocol::items::HookPromptFragment;
use codex_protocol::items::build_hook_prompt_message;
use codex_protocol::models::AgentMessageInputContent;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::protocol::APPS_INSTRUCTIONS_OPEN_TAG;
use codex_protocol::protocol::ExactTailModelVisibleItemKind;
use codex_protocol::protocol::InterAgentCommunication;
use codex_protocol::protocol::PLUGINS_INSTRUCTIONS_OPEN_TAG;
use codex_utils_output_truncation::TruncationPolicy;
use codex_utils_output_truncation::approx_bytes_for_tokens;
use pretty_assertions::assert_eq;
use std::sync::Arc;

fn mid_turn_injection() -> InitialContextInjection {
    InitialContextInjection::BeforeLastUserMessage(Arc::new(WorldState::default()))
}

#[test]
fn item_id_synthesis_cannot_mutate_exact_hot_suffix_after_verification() {
    let mut plan = plan(vec![user("old"), user("recent")], 1);
    let error = ensure_hot_suffix_item_ids_are_stable(&plan, /*item_ids_enabled*/ true)
        .expect_err("missing hot item IDs must fail before compaction");
    assert_eq!(error.reason, ExactTailFailReason::HotSuffixItemIdMissing);

    for item in &mut plan.hot_suffix {
        item.set_id(Some("msg_existing".to_string()));
    }
    ensure_hot_suffix_item_ids_are_stable(&plan, /*item_ids_enabled*/ true)
        .expect("existing hot item IDs remain stable during installation");
}

fn user(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
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
        internal_chat_message_metadata_passthrough: None,
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
        internal_chat_message_metadata_passthrough: None,
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
        internal_chat_message_metadata_passthrough: None,
    }
}

fn compaction_summary(text: &str) -> ResponseItem {
    ResponseItem::Compaction {
        id: None,
        encrypted_content: text.to_string(),
        internal_chat_message_metadata_passthrough: None,
    }
}

fn developer(content: Vec<ContentItem>) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "developer".to_string(),
        content,
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn function_call(call_id: &str) -> ResponseItem {
    ResponseItem::FunctionCall {
        id: None,
        name: "tool".to_string(),
        namespace: None,
        arguments: "{}".to_string(),
        call_id: call_id.to_string(),
        internal_chat_message_metadata_passthrough: None,
    }
}

fn function_output(call_id: &str, output: &str) -> ResponseItem {
    ResponseItem::FunctionCallOutput {
        id: None,
        call_id: call_id.to_string(),
        output: FunctionCallOutputPayload::from_text(output.to_string()),
        internal_chat_message_metadata_passthrough: None,
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

fn prepared_for_hot_suffix(
    hot_suffix: Vec<ResponseItem>,
    initial_context: Vec<ResponseItem>,
) -> PreparedExactTailPlan {
    let mut plan = plan(vec![user("old"), user("placeholder hot")], 1);
    plan.hot_suffix = hot_suffix;
    plan.tool_surface_hint = derive_exact_tail_tool_surface_hint(&plan.hot_suffix);
    PreparedExactTailPlan {
        plan,
        initial_context,
    }
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
fn exact_tail_normalizes_legacy_tool_outputs_only_for_exact_preserve_policy() {
    let old_policy = TruncationPolicy::Bytes(approx_bytes_for_tokens(25_000));
    let new_policy = TruncationPolicy::Bytes(approx_bytes_for_tokens(20_000));
    let item = ResponseItem::FunctionCallOutput {
        id: None,
        call_id: "legacy-large-output".to_string(),
        output: FunctionCallOutputPayload {
            body: FunctionCallOutputBody::Text("legacy exact-tail output ".repeat(25_000)),
            success: Some(true),
        },
        internal_chat_message_metadata_passthrough: None,
    };
    let mut standard_history = ContextManager::new();
    standard_history.record_items([&item], old_policy);
    let mut exact_history = standard_history.clone();
    let new_item_token_limit = model_visible_tool_output_item_token_limit(20_000);
    assert!(
        estimate_response_items_token_count(standard_history.raw_items()) > new_item_token_limit,
        "test setup should simulate a legacy item that no longer fits the lowered config"
    );

    let standard_normalized = normalize_tool_outputs_for_exact_tail_policy(
        &mut standard_history,
        CompactionHistoryPolicy::Standard,
        new_policy,
    );
    let exact_normalized = normalize_tool_outputs_for_exact_tail_policy(
        &mut exact_history,
        CompactionHistoryPolicy::PreserveRecentExact {
            target_tokens: 1,
            max_model_visible_item_tokens: new_item_token_limit,
        },
        new_policy,
    );

    assert_eq!(standard_normalized, 0);
    assert!(
        estimate_response_items_token_count(standard_history.raw_items()) > new_item_token_limit,
        "standard compaction should not normalize exact-tail-only compatibility"
    );
    assert_eq!(exact_normalized, 1);
    assert!(
        estimate_response_items_token_count(exact_history.raw_items()) <= new_item_token_limit,
        "exact-tail source history should be normalized to the active item envelope"
    );
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
fn planner_splits_long_turn_into_atomic_hot_groups() {
    let old = user("old history");
    let prompt = user("long running task");
    let step_one = assistant("step one");
    let step_two = assistant("step two");

    let actual = plan(
        vec![
            old.clone(),
            prompt.clone(),
            step_one.clone(),
            step_two.clone(),
        ],
        1,
    );

    assert_eq!(actual.cold_history, vec![old, prompt, step_one]);
    assert_eq!(actual.hot_suffix, vec![step_two]);
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

    assert_eq!(actual.cold_history, vec![old, recent]);
    assert_eq!(actual.hot_suffix, vec![call, output]);
    assert_eq!(actual.diagnostics.hot_group_count, 1);
}

#[test]
fn planner_preserves_interleaved_function_call_dependencies_as_one_hot_group() {
    let old = user("old history");
    let call_one = function_call("call-1");
    let call_two = function_call("call-2");
    let output_one = function_output("call-1", "tool result one");
    let output_two = function_output("call-2", "tool result two");

    let actual = plan(
        vec![
            old.clone(),
            call_one.clone(),
            call_two.clone(),
            output_one.clone(),
            output_two.clone(),
        ],
        1,
    );

    assert_eq!(actual.cold_history, vec![old]);
    assert_eq!(
        actual.hot_suffix,
        vec![call_one, call_two, output_one, output_two]
    );
    assert_eq!(actual.diagnostics.hot_group_count, 1);
}

#[test]
fn planner_target_overshoots_only_by_one_atomic_group() {
    let old = user("old history");
    let target = 150;
    let hot_groups = (0..5)
        .map(|index| {
            assistant(&format!(
                "granular response segment {index} {}",
                "x ".repeat(80)
            ))
        })
        .collect::<Vec<_>>();
    let history = std::iter::once(old.clone())
        .chain(hot_groups.iter().cloned())
        .collect::<Vec<_>>();

    let actual = plan(history, target);
    let newest_group_tokens =
        estimate_response_items_token_count(std::slice::from_ref(hot_groups.last().unwrap()));

    assert_eq!(actual.cold_history.first(), Some(&old));
    assert!(actual.diagnostics.actual_hot_tokens >= target);
    assert!(
        actual.diagnostics.actual_hot_tokens < target + newest_group_tokens,
        "granular grouping should bound target overshoot to one atomic group"
    );
}

#[test]
fn planner_treats_prior_compaction_summary_as_cold_prefix() {
    let old = user("old history");
    let summary = compaction_summary("prior exact-tail summary");
    let hot = user("recent exact tail");

    let actual = plan(vec![old.clone(), summary.clone(), hot.clone()], 10_000);

    assert_eq!(actual.cold_history, vec![old, summary]);
    assert_eq!(actual.hot_suffix, vec![hot]);
    assert!(actual.diagnostics.actual_hot_tokens < actual.diagnostics.requested_hot_tokens);
}

#[test]
fn planner_does_not_preserve_newest_compaction_summary_as_hot_suffix() {
    let old = user("old history");
    let summary = compaction_summary("prior exact-tail summary");

    let actual = plan(vec![old.clone(), summary.clone()], 10_000);

    assert_eq!(actual.cold_history, vec![old, summary]);
    assert!(actual.hot_suffix.is_empty());
    assert_eq!(actual.diagnostics.hot_group_count, 0);
}

#[test]
fn planner_does_not_apply_hot_budget_guard_to_newest_compaction_summary() {
    let old = user("old history");
    let summary = compaction_summary(&"prior exact-tail summary ".repeat(20_000));

    let actual = plan(vec![old.clone(), summary.clone()], 10_000);
    let summary_tokens = estimate_response_items_token_count(std::slice::from_ref(&summary));

    assert!(summary_tokens > actual.diagnostics.available_for_hot);
    assert_eq!(actual.cold_history, vec![old, summary]);
    assert!(actual.hot_suffix.is_empty());
}

#[test]
fn planner_treats_prior_local_summary_message_as_cold_prefix() {
    let old = user("old history");
    let summary = user(&format!("{SUMMARY_PREFIX}\nprior local exact-tail summary"));
    let hot = user("recent exact tail");

    let actual = plan(vec![old.clone(), summary.clone(), hot.clone()], 10_000);

    assert_eq!(actual.cold_history, vec![old, summary]);
    assert_eq!(actual.hot_suffix, vec![hot]);
    assert!(actual.diagnostics.actual_hot_tokens < actual.diagnostics.requested_hot_tokens);
}

#[test]
fn planner_does_not_preserve_newest_local_summary_message_as_hot_suffix() {
    let old = user("old history");
    let summary = user(&format!("{SUMMARY_PREFIX}\nprior local exact-tail summary"));

    let actual = plan(vec![old.clone(), summary.clone()], 10_000);

    assert_eq!(actual.cold_history, vec![old, summary]);
    assert!(actual.hot_suffix.is_empty());
    assert_eq!(actual.diagnostics.hot_group_count, 0);
}

#[test]
fn planner_reserves_post_summary_cold_context_when_target_is_too_large() {
    let old = user("old history");
    let summary = user(&format!("{SUMMARY_PREFIX}\nprior exact-tail summary"));
    let post_summary_cold = assistant(&"new cold material after summary ".repeat(40));
    let hot_one = assistant(&"recent exact tail one ".repeat(40));
    let hot_two = assistant(&"recent exact tail two ".repeat(40));

    let actual = plan(
        vec![
            old.clone(),
            summary.clone(),
            post_summary_cold.clone(),
            hot_one.clone(),
            hot_two.clone(),
        ],
        10_000,
    );

    assert_eq!(actual.cold_history, vec![old, summary, post_summary_cold]);
    assert_eq!(actual.hot_suffix, vec![hot_one, hot_two]);
    assert!(actual.diagnostics.actual_hot_tokens < actual.diagnostics.requested_hot_tokens);
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
fn oversize_hot_function_call_item_is_preserved_exactly() {
    let arguments = format!("{{\"command\":\"echo ok{}\"}}", "ё".repeat(120_000));
    let function_call = ResponseItem::FunctionCall {
        id: None,
        name: "shell_command".to_string(),
        namespace: None,
        arguments,
        call_id: "call-oversized".to_string(),
        internal_chat_message_metadata_passthrough: None,
    };
    let function_output = function_output("call-oversized", "failed to parse function arguments");
    let history = [user("old"), user("recent"), function_call, function_output];

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
    .expect("model-authored tool calls should preserve exactly when total budget fits");

    assert_eq!(
        actual.hot_suffix,
        vec![history[2].clone(), history[3].clone()]
    );
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
    assert_eq!(
        actual.model_visible_item_limit,
        Some(ExactTailModelVisibleItemLimit {
            item_kind: ExactTailModelVisibleItemKind::FunctionCallOutput,
            item_tokens: estimate_response_items_token_count(std::slice::from_ref(&history[3])),
            max_item_tokens: EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS,
        })
    );
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
    let output = function_output("call-1", &"x".repeat(84_000));
    let output_tokens = estimate_response_items_token_count(std::slice::from_ref(&output));
    assert!(output_tokens > EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS);
    assert!(output_tokens > 20_000);
    let configured_item_cap = model_visible_tool_output_item_token_limit(20_000);
    assert!(output_tokens <= configured_item_cap);
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
        max_model_visible_item_tokens: configured_item_cap,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: 0,
        implementation: ExactTailImplementation::Local,
    })
    .expect("configured 20k body cap should allow the read_thread-sized tool output");

    assert_eq!(actual.hot_suffix, vec![function_call("call-1"), output]);
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
fn cold_tool_output_fails_before_standard_remote_v2_trim_could_rescue_it() {
    let call_id = "cold-tool";
    let oversized_output = function_output(call_id, &"cold tool output ".repeat(20_000));
    let history = [
        user("cold user"),
        function_call(call_id),
        oversized_output,
        user("recent hot"),
    ];
    let actual = plan_exact_tail(ExactTailPlanInput {
        history_items: &history,
        target_tokens: 1,
        effective_replacement_budget: Some(200_000),
        required_current_context_budget: 0,
        final_replacement_extra_budget_tokens: 0,
        max_model_visible_item_tokens: 120_000,
        estimated_summary_scaffold_overhead_tokens: 0,
        retained_cold_user_message_budget_tokens: 0,
        implementation: ExactTailImplementation::RemoteV2,
    })
    .expect("large cold tool output should be cold, not the minimum hot group");
    assert_eq!(actual.hot_suffix, vec![user("recent hot")]);
    assert!(
        actual.diagnostics.cold_tokens > 30_000,
        "test setup must exceed the cold input window before standard trimming"
    );

    let error = check_cold_input_fits(&actual, &actual.cold_history, Some(30_000))
        .expect_err("untrimmed exact-tail cold tool output must fail before v2 compaction");

    assert_eq!(error.reason, ExactTailFailReason::ColdInputTooLarge);

    let trimmed_cold_history = vec![
        user("cold user"),
        function_call(call_id),
        function_output(
            call_id,
            "Output exceeded the available model context and was truncated",
        ),
    ];
    check_cold_input_fits(&actual, &trimmed_cold_history, Some(30_000))
        .expect("standard remote-v2 trim sentinel would fit, proving the ordering matters");
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
fn hot_suffix_proof_accepts_strict_tail_match() {
    let hot_user = user("protected hot user");
    let hot_assistant = assistant("protected hot assistant");
    let prepared = prepared_for_hot_suffix(vec![hot_user.clone(), hot_assistant.clone()], vec![]);
    let replacement = vec![compaction_summary("summary"), hot_user, hot_assistant];

    let proof = verify_exact_hot_suffix_preserved(&prepared, &replacement)
        .expect("strict hot suffix tail should verify");

    assert_eq!(
        proof,
        ExactTailHotSuffixProof {
            exact_match: true,
            planned_item_count: 2,
            installed_item_count: 2,
        }
    );
}

#[test]
fn hot_suffix_proof_rejects_mid_tail_initial_context_insertion() {
    let hot_user_one = user("first protected hot user");
    let hot_assistant_one = assistant("first protected hot assistant");
    let hot_user_two = user("second protected hot user");
    let hot_assistant_two = assistant("second protected hot assistant");
    let initial_context = developer(vec![ContentItem::InputText {
        text: "<token_budget>\n900 tokens remain\n</token_budget>".to_string(),
    }]);
    let prepared = prepared_for_hot_suffix(
        vec![
            hot_user_one.clone(),
            hot_assistant_one.clone(),
            hot_user_two.clone(),
            hot_assistant_two.clone(),
        ],
        vec![initial_context.clone()],
    );
    let replacement = vec![
        compaction_summary("summary"),
        hot_user_one,
        hot_assistant_one,
        initial_context,
        hot_user_two,
        hot_assistant_two,
    ];

    let error = verify_exact_hot_suffix_preserved(&prepared, &replacement)
        .expect_err("current context inside the planned hot suffix must not verify as exact");

    assert_eq!(error.reason, ExactTailFailReason::HotSuffixMismatch);
}

#[test]
fn hot_suffix_proof_rejects_mutated_hot_item() {
    let prepared = prepared_for_hot_suffix(vec![user("hot"), assistant("final")], vec![]);
    let replacement = vec![
        compaction_summary("summary"),
        user("hot mutated"),
        assistant("final"),
    ];

    let error = verify_exact_hot_suffix_preserved(&prepared, &replacement)
        .expect_err("mutated hot suffix should fail proof");

    assert_eq!(error.reason, ExactTailFailReason::HotSuffixMismatch);
}

#[test]
fn hot_suffix_proof_rejects_truncated_hot_suffix() {
    let prepared = prepared_for_hot_suffix(vec![user("hot"), assistant("final")], vec![]);
    let replacement = vec![compaction_summary("summary"), user("hot")];

    let error = verify_exact_hot_suffix_preserved(&prepared, &replacement)
        .expect_err("truncated hot suffix should fail proof");

    assert_eq!(error.reason, ExactTailFailReason::HotSuffixMismatch);
}

#[test]
fn hot_suffix_proof_rejects_reordered_hot_suffix() {
    let prepared = prepared_for_hot_suffix(vec![user("hot"), assistant("final")], vec![]);
    let replacement = vec![
        compaction_summary("summary"),
        assistant("final"),
        user("hot"),
    ];

    let error = verify_exact_hot_suffix_preserved(&prepared, &replacement)
        .expect_err("reordered hot suffix should fail proof");

    assert_eq!(error.reason, ExactTailFailReason::HotSuffixMismatch);
}

#[test]
fn hot_suffix_proof_rejects_wrong_source_suffix() {
    let prepared =
        prepared_for_hot_suffix(vec![user("source hot"), assistant("source final")], vec![]);
    let replacement = vec![
        compaction_summary("summary"),
        user("other hot"),
        assistant("other final"),
    ];

    let error = verify_exact_hot_suffix_preserved(&prepared, &replacement)
        .expect_err("wrong-source hot suffix should fail proof");

    assert_eq!(error.reason, ExactTailFailReason::HotSuffixMismatch);
}

#[test]
fn mid_turn_replacement_inserts_initial_context_before_protected_hot_suffix() {
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
        &mid_turn_injection(),
    );

    assert_eq!(
        actual,
        vec![
            summary,
            initial_context,
            first_hot_user,
            first_hot_assistant,
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
        &mid_turn_injection(),
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
        &mid_turn_injection(),
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
