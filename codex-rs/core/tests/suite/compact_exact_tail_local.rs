use super::compact_exact_tail_support::*;
use pretty_assertions::assert_eq;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_local_manual_compact_excludes_newest_atomic_hot_suffix_from_compaction_request()
-> Result<()> {
    let server = start_mock_server().await;
    let cold_user = format!("COLD_USER {}", "cold context ".repeat(100));
    let cold_assistant = format!("COLD_ASSISTANT {}", "cold answer ".repeat(100));
    let hot_user = format!("HOT_USER {}", "hot context ".repeat(100));
    let hot_assistant = format!("HOT_ASSISTANT {}", "hot answer ".repeat(100));
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("cold-assistant-message", &cold_assistant),
                ev_completed("cold-response"),
            ]),
            sse(vec![
                ev_assistant_message("hot-assistant-message", &hot_assistant),
                ev_completed("hot-response"),
            ]),
            sse(vec![
                ev_assistant_message("compact-assistant-message", "EXACT_TAIL_SUMMARY"),
                ev_completed("compact-response"),
            ]),
            sse(vec![
                ev_assistant_message("follow-up-assistant-message", "FOLLOW_UP_DONE"),
                ev_completed("follow-up-response"),
            ]),
        ],
    )
    .await;
    let provider = local_compaction_provider(&server);
    let mut builder = test_codex().with_config(move |config| {
        config.model_provider = provider;
        set_test_compact_prompt(config);
        config.model_context_window = Some(200_000);
        config.compact_preserve_recent_tokens = Some(1);
    });
    builder = builder.with_workspace_setup(|cwd, fs| async move {
        fs.write_file(
            &codex_utils_path_uri::PathUri::from_path(cwd.join("AGENTS.md"))?,
            b"LOCAL_MANUAL_CONTEXT_MARKER".to_vec(),
            /*sandbox*/ None,
        )
        .await?;
        Ok(())
    });
    let test = builder.build(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    test.submit_turn(&cold_user).await?;
    test.submit_turn(&hot_user).await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    test.submit_turn("FOLLOW_UP_USER").await?;

    let requests = response_mock.requests();
    assert_eq!(
        requests.len(),
        4,
        "expected cold turn, hot turn, compact, follow-up"
    );
    let compact_input = requests[2].input();
    assert!(
        input_contains_text(&compact_input, "COLD_USER"),
        "cold user should be sent to the compactor; compact input: {compact_input:#?}"
    );
    assert!(
        input_contains_text(&compact_input, "COLD_ASSISTANT"),
        "cold assistant output should be sent to the compactor; compact input: {compact_input:#?}"
    );
    assert!(
        input_contains_text(&compact_input, SUMMARIZATION_PROMPT),
        "compact request should include the summarization prompt"
    );
    assert!(
        input_contains_text(&compact_input, "HOT_USER"),
        "older same-turn user text should remain cold when the target only selects the newest atomic group; compact input: {compact_input:#?}"
    );
    assert!(
        !input_contains_text(&compact_input, "HOT_ASSISTANT"),
        "newest atomic hot assistant text must be excluded from the compaction request; compact input: {compact_input:#?}"
    );

    let replacement_history = replacement_history_from_rollout(&rollout_path)?;
    let expected_summary = summary_with_prefix("EXACT_TAIL_SUMMARY");
    assert_ordered_input_texts(
        &replacement_history,
        &["HOT_USER", expected_summary.as_str(), "HOT_ASSISTANT"],
    );
    assert!(
        !input_contains_text(&replacement_history, SUMMARIZATION_PROMPT),
        "replacement history must not persist the compact prompt"
    );
    assert!(
        !input_contains_text(&replacement_history, "LOCAL_MANUAL_CONTEXT_MARKER"),
        "manual/pre-turn exact-tail replacement must not persist reinjectable initial context"
    );

    let follow_up_input = requests[3].input();
    assert_eq!(
        input_text_match_count(&follow_up_input, "LOCAL_MANUAL_CONTEXT_MARKER"),
        1,
        "next normal request should reinject initial context exactly once"
    );
    assert_ordered_input_texts(
        &follow_up_input,
        &[
            "HOT_USER",
            expected_summary.as_str(),
            "HOT_ASSISTANT",
            "FOLLOW_UP_USER",
        ],
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_local_backend_context_window_error_fails_without_pruning() -> Result<()> {
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("local-backend-cold-assistant", "LOCAL_BACKEND_COLD"),
                ev_completed("local-backend-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message("local-backend-hot-assistant", "LOCAL_BACKEND_HOT"),
                ev_completed("local-backend-hot-response"),
            ]),
            sse_failed(
                "local-backend-compact-failed",
                "context_length_exceeded",
                "Your input exceeds the context window of this model. Please adjust your input and try again.",
            ),
        ],
    )
    .await;
    let provider = local_compaction_provider(&server);
    let mut builder = test_codex().with_config(move |config| {
        config.model_provider = provider;
        set_test_compact_prompt(config);
        config.model_context_window = Some(200_000);
        config.compact_preserve_recent_tokens = Some(1);
    });
    let test = builder.build(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    test.submit_turn("LOCAL_BACKEND_COLD_USER").await?;
    test.submit_turn("LOCAL_BACKEND_HOT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    let error_message = wait_for_event_match(&test.codex, |event| match event {
        EventMsg::Error(err) => Some(err.message.clone()),
        _ => None,
    })
    .await;

    assert!(
        error_message.contains("BackendContextExceededDespiteLocalFit"),
        "expected exact-tail backend context-window failure, got {error_message}"
    );
    assert_eq!(
        response_mock.requests().len(),
        3,
        "exact-tail backend context-window failure should not retry by pruning cold input"
    );
    assert!(
        replacement_history_from_rollout(&rollout_path).is_err(),
        "failed exact-tail backend context-window error must not install compacted history"
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_local_manual_reserves_reinjected_initial_context_before_compacting()
-> Result<()> {
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![sse(vec![
            ev_assistant_message("local-context-cold-assistant", "LOCAL_CONTEXT_COLD"),
            ev_completed("local-context-cold-response"),
        ])],
    )
    .await;
    let provider = local_compaction_provider(&server);
    let large_context = format!("LOCAL_LARGE_INITIAL_CONTEXT {}", "context ".repeat(6_000));
    let mut builder = test_codex()
        .with_config(move |config| {
            config.model_provider = provider;
            set_test_compact_prompt(config);
            config.model_context_window = Some(35_000);
            config.compact_preserve_recent_tokens = Some(1);
        })
        .with_workspace_setup(move |cwd, fs| async move {
            fs.write_file(
                &codex_utils_path_uri::PathUri::from_path(cwd.join("AGENTS.md"))?,
                large_context.into_bytes(),
                /*sandbox*/ None,
            )
            .await?;
            Ok(())
        });
    let test = builder.build(&server).await?;

    test.submit_turn("LOCAL_CONTEXT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    let error_message = wait_for_event_match(&test.codex, |event| match event {
        EventMsg::Error(err) => Some(err.message.clone()),
        _ => None,
    })
    .await;

    assert!(
        error_message.contains("ExactTailMinimumHotSuffixTooLarge"),
        "expected exact-tail current-context budget failure, got {error_message}"
    );
    let requests = response_mock.requests();
    assert!(
        requests
            .iter()
            .all(|request| !input_contains_text(&request.input(), SUMMARIZATION_PROMPT)),
        "exact-tail should reserve reinjected initial context and fail before issuing a local compaction request"
    );
    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_local_manual_rejects_oversize_reinjected_initial_context_item() -> Result<()> {
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![sse(vec![
            ev_assistant_message("local-context-item-cold-assistant", "LOCAL_ITEM_COLD"),
            ev_completed("local-context-item-cold-response"),
        ])],
    )
    .await;
    let provider = local_compaction_provider(&server);
    let oversized_context = format!(
        "LOCAL_OVERSIZE_INITIAL_CONTEXT {}",
        "oversize context ".repeat(40_000)
    );
    let mut builder = test_codex().with_config(move |config| {
        config.model_provider = provider;
        config.developer_instructions = Some(oversized_context);
        set_test_compact_prompt(config);
        config.model_context_window = Some(200_000);
        config.compact_preserve_recent_tokens = Some(1);
    });
    let test = builder.build(&server).await?;

    test.submit_turn("LOCAL_CONTEXT_ITEM_USER").await?;
    test.codex.submit(Op::Compact).await?;
    let error_message = wait_for_event_match(&test.codex, |event| match event {
        EventMsg::Error(err) => Some(err.message.clone()),
        _ => None,
    })
    .await;

    assert!(
        error_message.contains("ExactTailModelVisibleItemTooLarge"),
        "expected exact-tail current-context item cap failure, got {error_message}"
    );
    let requests = response_mock.requests();
    assert!(
        requests
            .iter()
            .all(|request| !input_contains_text(&request.input(), SUMMARIZATION_PROMPT)),
        "oversize reinjected context should fail before issuing a local compaction request"
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_local_empty_summary_fails_without_installing_history() -> Result<()> {
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("empty-summary-cold-assistant", "EMPTY_SUMMARY_COLD"),
                ev_completed("empty-summary-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message("empty-summary-hot-assistant", "EMPTY_SUMMARY_HOT"),
                ev_completed("empty-summary-hot-response"),
            ]),
            sse(vec![
                ev_assistant_message("empty-summary-compact-assistant", ""),
                ev_completed("empty-summary-compact-response"),
            ]),
        ],
    )
    .await;
    let provider = local_compaction_provider(&server);
    let mut builder = test_codex().with_config(move |config| {
        config.model_provider = provider;
        set_test_compact_prompt(config);
        config.model_context_window = Some(200_000);
        config.compact_preserve_recent_tokens = Some(1);
    });
    let test = builder.build(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    test.submit_turn("EMPTY_SUMMARY_COLD_USER").await?;
    test.submit_turn("EMPTY_SUMMARY_HOT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    let error_message = wait_for_event_match(&test.codex, |event| match event {
        EventMsg::Error(err) => Some(err.message.clone()),
        _ => None,
    })
    .await;

    assert!(
        error_message.contains("ExactTailNoUsableColdSummary"),
        "expected exact-tail empty summary error, got {error_message}"
    );
    assert_eq!(
        response_mock.requests().len(),
        3,
        "empty summary should fail after the local compactor returns"
    );
    assert!(
        replacement_history_from_rollout(&rollout_path).is_err(),
        "failed exact-tail summary must not install compacted history"
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_auto_compact_body_after_prefix_uses_full_context_window_budget() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let first_turn = sse(vec![
        ev_assistant_message("exact-tail-prefix-m1", FIRST_REPLY),
        ev_completed_with_usage(
            "exact-tail-prefix-r1",
            /*input_tokens*/ 600,
            /*output_tokens*/ 50,
        ),
    ]);
    let second_turn = sse(vec![
        ev_assistant_message("exact-tail-prefix-m2", SECOND_LARGE_REPLY),
        ev_completed_with_usage(
            "exact-tail-prefix-r2",
            /*input_tokens*/ 700,
            /*output_tokens*/ 50,
        ),
    ]);
    let auto_compact_turn = sse(vec![
        ev_assistant_message("exact-tail-prefix-m3", AUTO_SUMMARY_TEXT),
        ev_completed_with_tokens("exact-tail-prefix-r3", /*total_tokens*/ 20),
    ]);
    let third_turn = sse(vec![
        ev_assistant_message("exact-tail-prefix-m4", FINAL_REPLY),
        ev_completed_with_usage(
            "exact-tail-prefix-r4",
            /*input_tokens*/ 750,
            /*output_tokens*/ 20,
        ),
    ]);
    let request_log = mount_sse_sequence(
        &server,
        vec![first_turn, second_turn, auto_compact_turn, third_turn],
    )
    .await;

    let model_provider = local_compaction_provider(&server);
    let test = test_codex()
        .with_config(move |config| {
            config.model_provider = model_provider;
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.model_auto_compact_token_limit = Some(100);
            config.model_auto_compact_token_limit_scope =
                AutoCompactTokenLimitScope::BodyAfterPrefix;
            config.compact_preserve_recent_tokens = Some(1);
        })
        .build(&server)
        .await
        .expect("build codex");

    for user in ["EXACT_TAIL_BODY_PREFIX_ONE", "EXACT_TAIL_BODY_PREFIX_TWO"] {
        test.submit_turn(user).await.expect("submit turn");
    }
    assert_eq!(
        request_log.requests().len(),
        2,
        "first two turns should establish a prefix and growth without compacting"
    );

    test.submit_turn("EXACT_TAIL_BODY_PREFIX_THREE")
        .await
        .expect("submit third turn");

    let requests = request_log.requests();
    assert_eq!(
        requests.len(),
        4,
        "third turn should include exact-tail pre-turn compaction plus the post-compaction request"
    );
    let compact_body = requests[2].body_json().to_string();
    assert!(
        body_contains_text(&compact_body, SUMMARIZATION_PROMPT),
        "exact-tail body-after-prefix compaction should reach the compactor instead of failing against the body budget"
    );
    assert!(
        !body_contains_text(&compact_body, SECOND_LARGE_REPLY),
        "the protected newest atomic hot suffix should not be sent to the compactor"
    );

    shutdown_codex(&test).await?;
    Ok(())
}
