use super::compact_exact_tail_support::*;
use codex_protocol::config_types::CompactExactTailStrategy;
use core_test_support::responses;
use core_test_support::test_codex::TestCodexBuilder;
use pretty_assertions::assert_eq;

const TURN_STATE_HEADER: &str = "x-codex-turn-state";
const BETA_FEATURES_HEADER: &str = "x-codex-beta-features";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_semantic_transcript_manual_uses_v2_summary_only_when_default_v2_disabled()
-> Result<()> {
    let server = start_mock_server().await;
    let cold_user = format!("SEM_ROUTE_COLD_USER {}", "semantic cold user ".repeat(100));
    let cold_assistant = format!(
        "SEM_ROUTE_COLD_ASSISTANT {}",
        "semantic cold assistant ".repeat(100)
    );
    let hot_user = format!("SEM_ROUTE_HOT_USER {}", "semantic hot user ".repeat(100));
    let hot_assistant = format!(
        "SEM_ROUTE_HOT_ASSISTANT {}",
        "semantic hot assistant ".repeat(100)
    );
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("semantic-route-cold-assistant", &cold_assistant),
                ev_completed("semantic-route-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message("semantic-route-hot-assistant", &hot_assistant),
                ev_completed("semantic-route-hot-response"),
            ]),
            sse(vec![
                ev_compaction_item("SEM_ROUTE_EXACT_TAIL_SUMMARY"),
                ev_completed("semantic-route-compact-response"),
            ]),
            sse(vec![
                ev_assistant_message(
                    "semantic-route-follow-up-assistant",
                    "SEM_ROUTE_FOLLOW_UP_DONE",
                ),
                ev_completed("semantic-route-follow-up-response"),
            ]),
        ],
    )
    .await;
    let mut builder = semantic_builder().with_config(|config| {
        let _ = config.features.disable(Feature::RemoteCompactionV2);
    });
    let test = builder.build(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    submit_turn(&test, &cold_user).await?;
    submit_turn(&test, &hot_user).await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        Duration::from_secs(90),
    )
    .await;
    submit_turn(&test, "SEM_ROUTE_FOLLOW_UP_USER").await?;

    let requests = response_mock.requests();
    assert_eq!(
        requests.len(),
        4,
        "expected cold, hot, v2 semantic compact, and follow-up requests"
    );
    let compact_request = &requests[2];
    let compact_body = compact_request.body_json().to_string();
    assert!(
        compact_request
            .header(BETA_FEATURES_HEADER)
            .as_deref()
            .is_some_and(header_contains_remote_v2),
        "semantic transcript v2 compact request must advertise remote_compaction_v2 even when default routing is disabled"
    );
    assert!(
        body_contains_text(&compact_body, "SEM_ROUTE_COLD_USER"),
        "semantic transcript compact request should include cold user text; body: {compact_body}"
    );
    assert!(
        body_contains_text(&compact_body, "SEM_ROUTE_COLD_ASSISTANT"),
        "semantic transcript compact request should include cold assistant text; body: {compact_body}"
    );
    assert!(
        body_contains_text(&compact_body, "SEM_ROUTE_HOT_USER"),
        "older same-turn user text should remain cold; body: {compact_body}"
    );
    assert!(
        !body_contains_text(&compact_body, "SEM_ROUTE_HOT_ASSISTANT"),
        "exact hot assistant must stay out of semantic transcript compact request; body: {compact_body}"
    );
    assert_single_trailing_compaction_trigger(compact_request);
    assert!(
        !body_contains_text(&compact_body, SUMMARIZATION_PROMPT),
        "semantic transcript should not route through the legacy prompt-shaped compactor; body: {compact_body}"
    );

    let follow_up_input = requests[3].input();
    assert_ordered_input_texts(
        &follow_up_input,
        &[
            "SEM_ROUTE_COLD_USER",
            "SEM_ROUTE_HOT_USER",
            "SEM_ROUTE_EXACT_TAIL_SUMMARY",
            "SEM_ROUTE_HOT_ASSISTANT",
            "SEM_ROUTE_FOLLOW_UP_USER",
        ],
    );
    assert!(
        !input_contains_text(&follow_up_input, "SEM_ROUTE_COLD_ASSISTANT"),
        "cold assistant should exist only inside the v2 summary, not as duplicated exact retained prefix"
    );
    assert_eq!(
        input_text_match_count(&follow_up_input, "SEM_ROUTE_COLD_USER"),
        1,
        "retained cold user anchor should appear once"
    );
    assert_eq!(
        input_text_match_count(&follow_up_input, "SEM_ROUTE_HOT_USER"),
        1,
        "retained older same-turn user anchor should appear once"
    );
    let diagnostics = exact_tail_diagnostics_from_rollout(&rollout_path)?;
    assert_eq!(diagnostics.len(), 1);
    let diagnostic = &diagnostics[0];
    assert_eq!(
        diagnostic.get("route").and_then(Value::as_str),
        Some("semantic_transcript")
    );
    assert!(
        diagnostic
            .get("semantic_transcript_item_count")
            .and_then(Value::as_u64)
            .is_some()
    );
    assert_eq!(
        diagnostic
            .get("retained_cold_message_count")
            .and_then(Value::as_u64),
        Some(2)
    );
    assert_eq!(
        diagnostic
            .get("hot_suffix_exact_match")
            .and_then(Value::as_bool),
        Some(true)
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_semantic_transcript_rehydrates_tool_surface_after_compact() -> Result<()> {
    let server = start_mock_server().await;
    let tool_name = "semantic_continuity_probe";
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("semantic-continuity-cold", "SEM_CONTINUITY_COLD"),
                ev_completed("semantic-continuity-cold-response"),
            ]),
            sse(vec![
                ev_response_created("semantic-continuity-search-response"),
                ev_tool_search_call(
                    "semantic-continuity-search-call",
                    &serde_json::json!({
                        "query": "semantic continuity probe",
                        "limit": 4,
                    }),
                ),
                ev_completed("semantic-continuity-search-response"),
            ]),
            sse(vec![
                ev_assistant_message("semantic-continuity-hot", "SEM_CONTINUITY_HOT"),
                ev_completed("semantic-continuity-hot-response"),
            ]),
            sse(vec![
                ev_compaction_item("SEM_CONTINUITY_SUMMARY"),
                ev_completed("semantic-continuity-compact-response"),
            ]),
            sse(vec![
                ev_assistant_message("semantic-continuity-follow", "SEM_CONTINUITY_FOLLOW"),
                ev_completed("semantic-continuity-follow-response"),
            ]),
        ],
    )
    .await;
    let tool_description = "Semantic continuity probe. ".repeat(80);
    let dynamic_tool = deferred_dynamic_namespace_tool("codex_app", tool_name, &tool_description);
    let mut builder = semantic_builder().with_config(|config| {
        configure_search_capable_model(config);
        config.compact_preserve_recent_tokens = Some(100);
    });
    let base_test = builder.build(&server).await?;
    let new_thread = base_test
        .thread_manager
        .start_thread_with_tools(base_test.config.clone(), vec![dynamic_tool])
        .await?;
    let rollout_path = new_thread
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");
    let mut test = base_test;
    test.codex = new_thread.thread;
    test.session_configured = new_thread.session_configured;

    submit_turn(&test, "SEM_CONTINUITY_COLD_USER").await?;
    submit_turn(&test, "SEM_CONTINUITY_HOT_USER").await?;
    manual_compact_and_wait(&test).await?;
    submit_turn(&test, "SEM_CONTINUITY_FOLLOW_USER").await?;

    let follow_body = response_mock
        .requests()
        .last()
        .expect("follow-up request")
        .body_json();
    assert!(
        namespace_child_tool(&follow_body, "codex_app", tool_name).is_some(),
        "semantic transcript follow-up should rehydrate current dynamic tool spec: {follow_body}"
    );
    let diagnostics = exact_tail_tool_surface_diagnostics_from_rollout(&rollout_path)?;
    assert_eq!(diagnostics.len(), 1);
    let diagnostic = &diagnostics[0];
    assert_eq!(
        diagnostic["route"],
        serde_json::json!("semantic_transcript")
    );
    assert_eq!(diagnostic["rehydrated_tool_count"], serde_json::json!(1));
    assert_eq!(diagnostic["missing_hot_tool_count"], serde_json::json!(0));

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_semantic_transcript_auto_uses_v2_summary_only_request() -> Result<()> {
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("semantic-auto-cold", "SEM_AUTO_COLD_ASSISTANT"),
                ev_completed_with_tokens("semantic-auto-cold-response", /*total_tokens*/ 50),
            ]),
            sse(vec![
                ev_assistant_message("semantic-auto-hot", "SEM_AUTO_HOT_ASSISTANT"),
                ev_completed_with_tokens("semantic-auto-hot-response", /*total_tokens*/ 500),
            ]),
            sse(vec![
                ev_compaction_item("SEM_AUTO_EXACT_TAIL_SUMMARY"),
                ev_completed("semantic-auto-compact-response"),
            ]),
            sse(vec![
                ev_assistant_message("semantic-auto-follow-up", "SEM_AUTO_FOLLOW_UP_DONE"),
                ev_completed("semantic-auto-follow-up-response"),
            ]),
        ],
    )
    .await;
    let mut builder = semantic_builder_with_remote_v2().with_config(|config| {
        config.model_auto_compact_token_limit = Some(100);
        config.model_auto_compact_token_limit_scope = AutoCompactTokenLimitScope::BodyAfterPrefix;
    });
    let test = builder.build(&server).await?;

    submit_turn(&test, "SEM_AUTO_COLD_USER").await?;
    submit_turn(&test, "SEM_AUTO_HOT_USER").await?;
    submit_turn(&test, "SEM_AUTO_FOLLOW_UP_USER").await?;

    let requests = response_mock.requests();
    assert_eq!(
        requests.len(),
        4,
        "expected cold, hot, v2 semantic compact, and follow-up requests"
    );
    let compact_body = requests[2].body_json().to_string();
    assert!(
        body_contains_text(&compact_body, "SEM_AUTO_COLD_ASSISTANT"),
        "semantic auto compact request should include cold assistant transcript; body: {compact_body}"
    );
    assert_single_trailing_compaction_trigger(&requests[2]);
    let follow_up_input = requests[3].input();
    assert_ordered_input_texts(
        &follow_up_input,
        &[
            "SEM_AUTO_EXACT_TAIL_SUMMARY",
            "SEM_AUTO_HOT_ASSISTANT",
            "SEM_AUTO_FOLLOW_UP_USER",
        ],
    );
    assert!(
        !input_contains_text(&follow_up_input, "SEM_AUTO_COLD_ASSISTANT"),
        "cold assistant should be represented by the v2 summary only, not leaked as exact prefix"
    );
    assert_eq!(
        input_text_match_count(&follow_up_input, "SEM_AUTO_COLD_USER"),
        1,
        "retained auto cold user anchor should appear once"
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_semantic_transcript_auto_preserves_turn_state_and_resets_continuation()
-> Result<()> {
    let server = start_mock_server().await;
    let response_mock = responses::mount_response_sequence(
        &server,
        vec![
            responses::sse_response(sse(vec![
                ev_assistant_message("semantic-state-cold", "SEM_STATE_COLD_ASSISTANT"),
                ev_completed_with_tokens("semantic-state-cold-response", /*total_tokens*/ 50),
            ])),
            responses::sse_response(sse(vec![
                ev_function_call("semantic-state-call-before", DUMMY_FUNCTION_NAME, "{}"),
                ev_completed_with_tokens(
                    "semantic-state-first-response",
                    /*total_tokens*/ 500,
                ),
            ]))
            .insert_header(TURN_STATE_HEADER, "sampling-state"),
            responses::sse_response(sse(vec![
                ev_compaction_item("SEM_STATE_EXACT_TAIL_SUMMARY"),
                ev_completed("semantic-state-compact-response"),
            ]))
            .insert_header(TURN_STATE_HEADER, "compact-state"),
            responses::sse_response(sse(vec![
                ev_function_call("semantic-state-call-after", DUMMY_FUNCTION_NAME, "{}"),
                ev_completed_with_tokens("semantic-state-after-response", /*total_tokens*/ 80),
            ]))
            .insert_header(TURN_STATE_HEADER, "continuation-state"),
            responses::sse_response(sse(vec![
                ev_assistant_message("semantic-state-final", "SEM_STATE_FINAL"),
                ev_completed_with_tokens("semantic-state-final-response", /*total_tokens*/ 80),
            ])),
        ],
    )
    .await;
    let mut builder = semantic_builder().with_config(|config| {
        config.model_auto_compact_token_limit = Some(200);
        config.model_auto_compact_token_limit_scope = AutoCompactTokenLimitScope::BodyAfterPrefix;
    });
    let test = builder.build(&server).await?;

    submit_turn(&test, "SEM_STATE_COLD_USER").await?;
    submit_turn(&test, "SEM_STATE_HOT_TOOL_USER").await?;

    let requests = response_mock.requests();
    assert_eq!(requests.len(), 5);
    assert_eq!(requests[0].header(TURN_STATE_HEADER), None);
    assert_eq!(requests[1].header(TURN_STATE_HEADER), None);

    let compact_request = &requests[2];
    let compact_body = compact_request.body_json();
    assert_eq!(
        compact_request.header(TURN_STATE_HEADER).as_deref(),
        Some("sampling-state")
    );
    assert_eq!(
        compact_body.get("previous_response_id"),
        None,
        "semantic v2 compaction must not continue the sampling Responses chain"
    );
    assert!(
        compact_request
            .header(BETA_FEATURES_HEADER)
            .as_deref()
            .is_some_and(header_contains_remote_v2),
        "semantic v2 compaction must advertise the remote_compaction_v2 capability"
    );
    assert!(
        compact_body
            .get("prompt_cache_key")
            .and_then(Value::as_str)
            .is_some_and(|key| key.starts_with("semantic-transcript-v2-compaction:")),
        "semantic v2 compaction must use an isolated prompt cache key; body: {compact_body}"
    );
    assert_single_trailing_compaction_trigger(compact_request);

    let post_compaction_body = requests[3].body_json();
    assert_eq!(
        requests[3].header(TURN_STATE_HEADER).as_deref(),
        Some("sampling-state")
    );
    assert_eq!(
        requests[4].header(TURN_STATE_HEADER).as_deref(),
        Some("sampling-state")
    );
    assert_eq!(
        post_compaction_body.get("previous_response_id"),
        None,
        "post-compaction sampling must reset stale continuation state"
    );
    assert!(
        !post_compaction_body
            .get("prompt_cache_key")
            .and_then(Value::as_str)
            .is_some_and(|key| key.starts_with("semantic-transcript-v2-compaction:")),
        "post-compaction sampling must not inherit the isolated compact cache key"
    );
    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_semantic_transcript_respects_raised_effective_item_cap() -> Result<()> {
    let server = start_mock_server().await;
    let cold_user = format!(
        "SEM_RAISED_COLD_START {} SEM_RAISED_COLD_END",
        "word ".repeat(12_000)
    );
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("semantic-raised-cold", "SEM_RAISED_COLD_ASSISTANT"),
                ev_completed("semantic-raised-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message("semantic-raised-hot", "SEM_RAISED_HOT_ASSISTANT"),
                ev_completed("semantic-raised-hot-response"),
            ]),
            sse(vec![
                ev_compaction_item("SEM_RAISED_EXACT_TAIL_SUMMARY"),
                ev_completed("semantic-raised-compact-response"),
            ]),
            sse(vec![
                ev_assistant_message("semantic-raised-follow-up", "SEM_RAISED_FOLLOW_UP_DONE"),
                ev_completed("semantic-raised-follow-up-response"),
            ]),
        ],
    )
    .await;
    // Reviewed cap exception: semantic transcript should honor the active
    // user-configured exact-tail/model-visible item envelope. The default
    // safety cap remains 10k, but an explicit tool_output_token_limit override
    // raises the effective cap for already-bounded model-visible items.
    let mut builder = semantic_builder().with_config(|config| {
        config.tool_output_token_limit = Some(20_000);
    });
    let test = builder.build(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    submit_turn(&test, &cold_user).await?;
    submit_turn(&test, "SEM_RAISED_HOT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        Duration::from_secs(90),
    )
    .await;
    submit_turn(&test, "SEM_RAISED_FOLLOW_UP_USER").await?;

    let requests = response_mock.requests();
    assert_eq!(requests.len(), 4);
    let compact_body = requests[2].body_json().to_string();
    assert!(
        body_contains_text(&compact_body, "SEM_RAISED_COLD_END"),
        "raised cap should preserve the cold user anchor in the semantic transcript request; body: {compact_body}"
    );
    assert!(
        !compact_body.contains("tokens truncated"),
        "raised cap should avoid fixed-10k truncation marker in the semantic transcript request; body: {compact_body}"
    );
    assert_single_trailing_compaction_trigger(&requests[2]);
    let follow_up_input = requests[3].input();
    assert!(
        input_contains_text(&follow_up_input, "SEM_RAISED_COLD_END"),
        "raised cap should retain the full cold user anchor in the installed prefix"
    );
    assert!(
        !input_contains_text(&follow_up_input, "tokens truncated"),
        "raised cap should avoid fixed-10k truncation marker in installed prefix"
    );

    let diagnostics = exact_tail_diagnostics_from_rollout(&rollout_path)?;
    let diagnostic = diagnostics.last().expect("semantic diagnostic");
    assert_eq!(
        diagnostic.get("route").and_then(Value::as_str),
        Some("semantic_transcript")
    );
    assert!(
        diagnostic
            .get("max_model_visible_item_tokens")
            .and_then(Value::as_i64)
            .is_some_and(|tokens| tokens >= 20_000),
        "semantic diagnostic should report the raised effective item cap: {diagnostic:#?}"
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_semantic_transcript_recompacts_prior_retained_prefix_and_summary() -> Result<()>
{
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("semantic-cycle-cold", "SEM_CYCLE_COLD_ASSISTANT"),
                ev_completed("semantic-cycle-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message("semantic-cycle-hot", "SEM_CYCLE_HOT_ASSISTANT"),
                ev_completed("semantic-cycle-hot-response"),
            ]),
            sse(vec![
                ev_compaction_item("SEM_CYCLE_SUMMARY_ONE"),
                ev_completed("semantic-cycle-compact-one-response"),
            ]),
            sse(vec![
                ev_assistant_message(
                    "semantic-cycle-after-first",
                    "SEM_CYCLE_AFTER_FIRST_ASSISTANT",
                ),
                ev_completed("semantic-cycle-after-first-response"),
            ]),
            sse(vec![
                ev_compaction_item("SEM_CYCLE_SUMMARY_TWO"),
                ev_completed("semantic-cycle-compact-two-response"),
            ]),
            sse(vec![
                ev_assistant_message("semantic-cycle-follow-up", "SEM_CYCLE_FOLLOW_UP_DONE"),
                ev_completed("semantic-cycle-follow-up-response"),
            ]),
        ],
    )
    .await;
    let mut builder = semantic_builder();
    let test = builder.build(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    submit_turn(&test, "SEM_CYCLE_COLD_USER").await?;
    submit_turn(&test, "SEM_CYCLE_HOT_USER").await?;
    manual_compact_and_wait(&test).await?;
    submit_turn(&test, "SEM_CYCLE_AFTER_FIRST_USER").await?;
    manual_compact_and_wait(&test).await?;
    submit_turn(&test, "SEM_CYCLE_FOLLOW_UP_USER").await?;

    let requests = response_mock.requests();
    assert_eq!(
        requests.len(),
        6,
        "expected cold, hot, first compact, after-first, second compact, and follow-up"
    );
    let second_compact_body = requests[4].body_json().to_string();
    assert!(
        body_contains_text(&second_compact_body, "SEM_CYCLE_SUMMARY_ONE"),
        "second semantic transcript should include the prior summary; body: {second_compact_body}"
    );
    assert!(
        body_contains_text(&second_compact_body, "SEM_CYCLE_COLD_USER"),
        "second semantic transcript should include the prior retained cold prefix; body: {second_compact_body}"
    );
    assert!(
        body_contains_text(&second_compact_body, "SEM_CYCLE_AFTER_FIRST_USER"),
        "second semantic transcript should include newly cold user text; body: {second_compact_body}"
    );
    assert_single_trailing_compaction_trigger(&requests[4]);

    let replacement = replacement_history_from_rollout(&rollout_path)?;
    assert!(
        input_contains_text(&replacement, "SEM_CYCLE_SUMMARY_TWO"),
        "second replacement should install the new summary: {replacement:#?}"
    );
    assert!(
        !input_contains_text(&replacement, "SEM_CYCLE_SUMMARY_ONE"),
        "second replacement should not permanently retain the prior summary outside the new summary: {replacement:#?}"
    );
    assert_eq!(
        input_text_match_count(&replacement, "SEM_CYCLE_COLD_USER"),
        1
    );
    assert_ordered_input_texts(
        &requests[5].input(),
        &[
            "SEM_CYCLE_COLD_USER",
            "SEM_CYCLE_AFTER_FIRST_USER",
            "SEM_CYCLE_SUMMARY_TWO",
            "SEM_CYCLE_AFTER_FIRST_ASSISTANT",
            "SEM_CYCLE_FOLLOW_UP_USER",
        ],
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_semantic_transcript_rejects_empty_summary_with_old_retained_summary()
-> Result<()> {
    let server = start_mock_server().await;
    mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("semantic-empty-cold", "SEM_EMPTY_COLD_ASSISTANT"),
                ev_completed("semantic-empty-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message("semantic-empty-hot", "SEM_EMPTY_HOT_ASSISTANT"),
                ev_completed("semantic-empty-hot-response"),
            ]),
            sse(vec![
                ev_compaction_item(""),
                ev_completed("semantic-empty-compact-response"),
            ]),
        ],
    )
    .await;
    let mut builder = semantic_builder();
    let test = builder.build(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    submit_turn(
        &test,
        &format!("{SUMMARY_PREFIX}\nSEM_EMPTY_OLD_RETAINED_SUMMARY"),
    )
    .await?;
    submit_turn(&test, "SEM_EMPTY_HOT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::Error(_)),
        Duration::from_secs(90),
    )
    .await;

    assert!(
        replacement_history_from_rollout(&rollout_path).is_err(),
        "failed semantic compaction should not install replacement history"
    );
    let diagnostics = exact_tail_diagnostics_from_rollout(&rollout_path)?;
    let diagnostic = diagnostics.last().expect("semantic failure diagnostic");
    assert_eq!(
        diagnostic.get("route").and_then(Value::as_str),
        Some("semantic_transcript")
    );
    assert_eq!(
        diagnostic.get("fit_result").and_then(Value::as_str),
        Some("failure")
    );
    assert_eq!(
        diagnostic.get("failure_reason").and_then(Value::as_str),
        Some("ExactTailNoUsableColdSummary")
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_semantic_transcript_accepts_bounded_v2_summary_with_retained_budgets()
-> Result<()> {
    for retained_budget in [32_000, 64_000] {
        run_bounded_v2_summary_success(retained_budget).await?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_semantic_transcript_replacement_failure_reports_summary_tokens() -> Result<()> {
    let server = start_mock_server().await;
    mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("semantic-fail-cold", "SEM_FAIL_COLD_ASSISTANT"),
                ev_completed("semantic-fail-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message("semantic-fail-hot", "SEM_FAIL_HOT_ASSISTANT"),
                ev_completed("semantic-fail-hot-response"),
            ]),
            sse(vec![
                ev_compaction_item(&format!(
                    "SEM_FAIL_LARGE_SUMMARY {}",
                    "summary ".repeat(60_000)
                )),
                ev_completed("semantic-fail-compact-response"),
            ]),
        ],
    )
    .await;
    let mut builder = semantic_builder().with_config(|config| {
        config.model_context_window = Some(80_000);
    });
    let test = builder.build(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    submit_turn(&test, "SEM_FAIL_COLD_USER").await?;
    submit_turn(&test, "SEM_FAIL_HOT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::Error(_)),
        Duration::from_secs(90),
    )
    .await;

    assert!(
        replacement_history_from_rollout(&rollout_path).is_err(),
        "oversized semantic summary must not install replacement history"
    );

    let diagnostics = exact_tail_diagnostics_from_rollout(&rollout_path)?;
    let diagnostic = diagnostics.last().expect("semantic failure diagnostic");
    assert_eq!(
        diagnostic.get("route").and_then(Value::as_str),
        Some("semantic_transcript")
    );
    assert_eq!(
        diagnostic.get("fit_result").and_then(Value::as_str),
        Some("failure")
    );
    assert_eq!(
        diagnostic.get("failure_reason").and_then(Value::as_str),
        Some("ExactTailModelVisibleItemTooLarge")
    );
    assert!(
        diagnostic
            .get("summary_tokens")
            .and_then(Value::as_i64)
            .is_some_and(|tokens| tokens > 10_000),
        "failure diagnostic should report returned summary tokens: {diagnostic:#?}"
    );
    let replacement_tokens = diagnostic
        .get("replacement_tokens_estimate")
        .and_then(Value::as_i64)
        .expect("replacement estimate");
    let final_replacement_tokens = diagnostic
        .get("final_replacement_tokens_estimate")
        .and_then(Value::as_i64)
        .expect("final replacement estimate");
    let final_extra_tokens = diagnostic
        .get("final_replacement_extra_budget_tokens")
        .and_then(Value::as_i64)
        .expect("final replacement extra budget");
    let effective_budget = diagnostic
        .get("effective_replacement_budget")
        .and_then(Value::as_i64)
        .expect("effective replacement budget");
    assert_eq!(
        final_replacement_tokens,
        replacement_tokens + final_extra_tokens
    );
    assert!(
        replacement_tokens > effective_budget,
        "failure diagnostic should report replacement estimate: {diagnostic:#?}"
    );
    assert!(
        final_replacement_tokens > effective_budget,
        "failure diagnostic should report final replacement estimate: {diagnostic:#?}"
    );

    shutdown_codex(&test).await?;
    Ok(())
}

fn semantic_builder() -> TestCodexBuilder {
    test_codex()
        .with_auth(CodexAuth::from_api_key("dummy"))
        .with_config(|config| {
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.compact_preserve_recent_tokens = Some(1);
            config.compact_exact_tail_strategy = CompactExactTailStrategy::SemanticTranscript;
            config.compact_exact_tail_semantic_transcript_retained_message_token_budget = 32_000;
        })
}

fn semantic_builder_with_remote_v2() -> TestCodexBuilder {
    semantic_builder().with_config(explicitly_enable_remote_compaction_v2)
}

async fn manual_compact_and_wait(test: &TestCodex) -> Result<()> {
    test.codex.submit(Op::Compact).await?;
    wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        Duration::from_secs(90),
    )
    .await;
    Ok(())
}

async fn submit_turn(test: &TestCodex, prompt: &str) -> Result<()> {
    test.submit_turn_with_completion_timeout(prompt, Duration::from_secs(90))
        .await
}

fn header_contains_remote_v2(header: &str) -> bool {
    header
        .split(',')
        .any(|feature| feature == "remote_compaction_v2")
}

fn assert_single_trailing_compaction_trigger(
    request: &core_test_support::responses::ResponsesRequest,
) {
    let input = request.input();
    assert_eq!(
        input
            .iter()
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("compaction_trigger"))
            .count(),
        1,
        "semantic v2 compaction input should contain exactly one compaction_trigger"
    );
    assert_eq!(
        input
            .last()
            .and_then(|item| item.get("type"))
            .and_then(Value::as_str),
        Some("compaction_trigger"),
        "semantic v2 compaction input should end with compaction_trigger"
    );
}

async fn run_bounded_v2_summary_success(retained_budget: i64) -> Result<()> {
    let server = start_mock_server().await;
    let marker = format!("SEM_LARGE_SUMMARY_{retained_budget}");
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message(
                    &format!("semantic-large-cold-{retained_budget}"),
                    &format!("SEM_LARGE_COLD_ASSISTANT_{retained_budget}"),
                ),
                ev_completed(&format!("semantic-large-cold-response-{retained_budget}")),
            ]),
            sse(vec![
                ev_assistant_message(
                    &format!("semantic-large-hot-{retained_budget}"),
                    &format!("SEM_LARGE_HOT_ASSISTANT_{retained_budget}"),
                ),
                ev_completed(&format!("semantic-large-hot-response-{retained_budget}")),
            ]),
            sse(vec![
                ev_compaction_item(&format!("{marker} {}", "summary ".repeat(800))),
                ev_completed(&format!(
                    "semantic-large-compact-response-{retained_budget}"
                )),
            ]),
            sse(vec![
                ev_assistant_message(
                    &format!("semantic-large-follow-up-{retained_budget}"),
                    &format!("SEM_LARGE_FOLLOW_UP_DONE_{retained_budget}"),
                ),
                ev_completed(&format!(
                    "semantic-large-follow-up-response-{retained_budget}"
                )),
            ]),
        ],
    )
    .await;
    let mut builder = semantic_builder().with_config(move |config| {
        config.compact_exact_tail_semantic_transcript_retained_message_token_budget =
            retained_budget;
    });
    let test = builder.build(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    submit_turn(&test, &format!("SEM_LARGE_COLD_USER_{retained_budget}")).await?;
    submit_turn(&test, &format!("SEM_LARGE_HOT_USER_{retained_budget}")).await?;
    manual_compact_and_wait(&test).await?;
    submit_turn(
        &test,
        &format!("SEM_LARGE_FOLLOW_UP_USER_{retained_budget}"),
    )
    .await?;

    let requests = response_mock.requests();
    assert_eq!(requests.len(), 4);
    let follow_up_input = requests[3].input();
    assert!(
        input_contains_text(&follow_up_input, &marker),
        "large returned summary should be installed for retained budget {retained_budget}: {follow_up_input:#?}"
    );
    assert!(
        input_contains_text(
            &follow_up_input,
            &format!("SEM_LARGE_HOT_ASSISTANT_{retained_budget}")
        ),
        "exact hot suffix should remain installed for retained budget {retained_budget}: {follow_up_input:#?}"
    );

    let diagnostics = exact_tail_diagnostics_from_rollout(&rollout_path)?;
    let diagnostic = diagnostics.last().expect("semantic diagnostic");
    assert_eq!(
        diagnostic.get("route").and_then(Value::as_str),
        Some("semantic_transcript")
    );
    assert_eq!(
        diagnostic.get("fit_result").and_then(Value::as_str),
        Some("success")
    );
    assert!(
        diagnostic
            .get("summary_tokens")
            .and_then(Value::as_i64)
            .is_some_and(|tokens| tokens > 0),
        "bounded summary success should report returned summary tokens: {diagnostic:#?}"
    );

    shutdown_codex(&test).await?;
    Ok(())
}
