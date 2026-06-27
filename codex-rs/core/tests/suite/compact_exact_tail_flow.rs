use super::compact_exact_tail_support::*;
use pretty_assertions::assert_eq;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_mid_turn_compaction_preserves_hot_tool_turn_outside_compactor() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let cold_turn = sse(vec![
        ev_assistant_message(
            "exact-tail-mid-cold-assistant",
            "EXACT_TAIL_MID_COLD_ASSISTANT",
        ),
        ev_completed_with_usage(
            "exact-tail-mid-cold-response",
            /*input_tokens*/ 100,
            /*output_tokens*/ 50,
        ),
    ]);
    let tool_turn = sse(vec![
        ev_function_call(DUMMY_CALL_ID, DUMMY_FUNCTION_NAME, "{}"),
        ev_completed_with_usage(
            "exact-tail-mid-tool-response",
            /*input_tokens*/ 500,
            /*output_tokens*/ 50,
        ),
    ]);
    let auto_compact_turn = sse(vec![
        ev_assistant_message("exact-tail-mid-summary", "EXACT_TAIL_MID_SUMMARY"),
        ev_completed_with_tokens("exact-tail-mid-compact-response", /*total_tokens*/ 20),
    ]);
    let post_auto_compact_turn = sse(vec![
        ev_assistant_message("exact-tail-mid-final", "EXACT_TAIL_MID_FINAL"),
        ev_completed_with_tokens("exact-tail-mid-final-response", /*total_tokens*/ 30),
    ]);
    let request_log = mount_sse_sequence(
        &server,
        vec![
            cold_turn,
            tool_turn,
            auto_compact_turn,
            post_auto_compact_turn,
        ],
    )
    .await;
    let provider = local_compaction_provider(&server);
    let test = test_codex()
        .with_config(move |config| {
            config.model_provider = provider;
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.model_auto_compact_token_limit = Some(200);
            config.model_auto_compact_token_limit_scope =
                AutoCompactTokenLimitScope::BodyAfterPrefix;
            config.compact_preserve_recent_tokens = Some(1);
        })
        .with_workspace_setup(|cwd, fs| async move {
            fs.write_file(
                &codex_utils_path_uri::PathUri::from_host_native_path(cwd.join("AGENTS.md"))?,
                b"EXACT_TAIL_MID_CONTEXT_MARKER".to_vec(),
                /*sandbox*/ None,
            )
            .await?;
            Ok(())
        })
        .build(&server)
        .await
        .expect("build codex");

    test.submit_turn_with_completion_timeout("EXACT_TAIL_MID_COLD_USER", Duration::from_secs(90))
        .await
        .expect("submit cold turn");
    test.submit_turn_with_completion_timeout("EXACT_TAIL_MID_HOT_USER", Duration::from_secs(90))
        .await
        .expect("submit hot tool turn");

    let requests = request_log.requests();
    assert_eq!(
        requests.len(),
        4,
        "expected cold turn, tool turn, exact-tail mid-turn compact, and continuation"
    );
    let compact_input = requests[2].input();
    assert!(
        input_contains_text(&compact_input, "EXACT_TAIL_MID_COLD_USER"),
        "cold prefix should be sent to mid-turn exact-tail compactor"
    );
    assert!(
        input_contains_text(&compact_input, "EXACT_TAIL_MID_HOT_USER"),
        "older same-turn user text should remain cold when the protected suffix is the tool dependency group"
    );
    assert!(
        input_item_index_by_type_and_call_id(&compact_input, "function_call", DUMMY_CALL_ID)
            .is_none(),
        "hot tool call should be preserved exactly outside the compactor"
    );
    assert!(
        input_item_index_by_type_and_call_id(&compact_input, "function_call_output", DUMMY_CALL_ID)
            .is_none(),
        "hot tool output should be preserved exactly outside the compactor"
    );

    let continuation_input = requests[3].input();
    assert!(
        input_contains_text(&continuation_input, "EXACT_TAIL_MID_SUMMARY"),
        "continuation should include the generated cold summary"
    );
    assert!(
        input_contains_text(&continuation_input, "EXACT_TAIL_MID_CONTEXT_MARKER"),
        "continuation should include reinjected initial context"
    );
    assert_ordered_input_texts(
        &continuation_input,
        &["EXACT_TAIL_MID_HOT_USER", "EXACT_TAIL_MID_SUMMARY"],
    );
    let function_call_index =
        input_item_index_by_type_and_call_id(&continuation_input, "function_call", DUMMY_CALL_ID)
            .expect("continuation should preserve function call with original call id");
    let function_output_index = input_item_index_by_type_and_call_id(
        &continuation_input,
        "function_call_output",
        DUMMY_CALL_ID,
    )
    .expect("continuation should preserve function output with original call id");
    assert!(
        function_call_index < function_output_index,
        "function call should precede function output in continuation input"
    );
    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_replacement_history_survives_resume() -> Result<()> {
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("resume-cold-assistant", "RESUME_COLD_ASSISTANT"),
                ev_completed("resume-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message("resume-hot-assistant", "RESUME_HOT_ASSISTANT"),
                ev_completed("resume-hot-response"),
            ]),
            sse(vec![
                ev_assistant_message("resume-compact-assistant", "RESUME_EXACT_TAIL_SUMMARY"),
                ev_completed("resume-compact-response"),
            ]),
            sse(vec![
                ev_assistant_message("resume-follow-up-assistant", "RESUME_FOLLOW_UP_DONE"),
                ev_completed("resume-follow-up-response"),
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
    let initial = builder.build(&server).await?;
    let home = initial.home.clone();
    let rollout_path = initial
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    initial
        .submit_turn_with_completion_timeout("RESUME_COLD_USER", Duration::from_secs(90))
        .await?;
    initial
        .submit_turn_with_completion_timeout("RESUME_HOT_USER", Duration::from_secs(90))
        .await?;
    initial.codex.submit(Op::Compact).await?;
    wait_for_event_with_timeout(
        &initial.codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        Duration::from_secs(90),
    )
    .await;
    initial.codex.submit(Op::Shutdown).await?;
    wait_for_event_with_timeout(
        &initial.codex,
        |event| matches!(event, EventMsg::ShutdownComplete),
        Duration::from_secs(90),
    )
    .await;

    let provider = local_compaction_provider(&server);
    let mut resumed_builder = test_codex().with_config(move |config| {
        config.model_provider = provider;
        set_test_compact_prompt(config);
        config.model_context_window = Some(200_000);
        config.compact_preserve_recent_tokens = Some(1);
    });
    let resumed = resumed_builder.resume(&server, home, rollout_path).await?;
    resumed
        .submit_turn_with_completion_timeout("RESUME_FOLLOW_UP_USER", Duration::from_secs(90))
        .await?;

    let requests = response_mock.requests();
    assert_eq!(
        requests.len(),
        4,
        "expected cold, hot, compact, and resumed follow-up requests"
    );
    let follow_up_body = requests[3].body_json().to_string();
    assert!(
        body_contains_text(&follow_up_body, "RESUME_EXACT_TAIL_SUMMARY")
            && body_contains_text(&follow_up_body, "RESUME_HOT_USER")
            && body_contains_text(&follow_up_body, "RESUME_HOT_ASSISTANT")
            && body_contains_text(&follow_up_body, "RESUME_FOLLOW_UP_USER"),
        "resumed follow-up should include the cold summary and exact hot suffix"
    );
    assert!(
        !body_contains_text(&follow_up_body, "RESUME_COLD_ASSISTANT"),
        "resumed follow-up should not replay cold assistant text outside the summary"
    );

    shutdown_codex(&resumed).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_manual_compact_twice_preserves_newest_suffix_each_time() -> Result<()> {
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("twice-cold-assistant", "TWICE_COLD_ASSISTANT"),
                ev_completed("twice-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message("twice-hot-assistant", "TWICE_HOT_ASSISTANT"),
                ev_completed("twice-hot-response"),
            ]),
            sse(vec![
                ev_assistant_message("twice-compact-one", "TWICE_EXACT_TAIL_SUMMARY_ONE"),
                ev_completed("twice-compact-one-response"),
            ]),
            sse(vec![
                ev_assistant_message("twice-after-one-assistant", "TWICE_AFTER_ONE_ASSISTANT"),
                ev_completed("twice-after-one-response"),
            ]),
            sse(vec![
                ev_assistant_message("twice-compact-two", "TWICE_EXACT_TAIL_SUMMARY_TWO"),
                ev_completed("twice-compact-two-response"),
            ]),
            sse(vec![
                ev_assistant_message("twice-final-assistant", "TWICE_FINAL_DONE"),
                ev_completed("twice-final-response"),
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

    test.submit_turn_with_completion_timeout("TWICE_COLD_USER", Duration::from_secs(90))
        .await?;
    test.submit_turn_with_completion_timeout("TWICE_HOT_USER", Duration::from_secs(90))
        .await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        Duration::from_secs(90),
    )
    .await;
    test.submit_turn_with_completion_timeout("TWICE_AFTER_ONE_USER", Duration::from_secs(90))
        .await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        Duration::from_secs(90),
    )
    .await;
    test.submit_turn_with_completion_timeout("TWICE_FINAL_USER", Duration::from_secs(90))
        .await?;

    let requests = response_mock.requests();
    assert_eq!(
        requests.len(),
        6,
        "expected cold, hot, first compact, after-one, second compact, and final requests"
    );
    let second_compact_body = requests[4].body_json().to_string();
    assert!(
        body_contains_text(&second_compact_body, "TWICE_EXACT_TAIL_SUMMARY_ONE")
            && body_contains_text(&second_compact_body, "TWICE_HOT_ASSISTANT")
            && body_contains_text(&second_compact_body, "TWICE_AFTER_ONE_USER"),
        "second compact should summarize the prior summary plus older atomic suffix material"
    );
    assert!(
        !body_contains_text(&second_compact_body, "TWICE_AFTER_ONE_ASSISTANT"),
        "second compact should exclude only the newest atomic exact suffix from the compactor"
    );

    let final_body = requests[5].body_json().to_string();
    assert!(
        body_contains_text(&final_body, "TWICE_EXACT_TAIL_SUMMARY_TWO")
            && body_contains_text(&final_body, "TWICE_AFTER_ONE_ASSISTANT")
            && body_contains_text(&final_body, "TWICE_FINAL_USER"),
        "final request should contain second cold summary plus the newest atomic exact suffix"
    );
    assert!(
        !body_contains_text(&final_body, "TWICE_HOT_ASSISTANT"),
        "final request should not replay older assistant text outside the second summary"
    );
    assert!(
        !body_contains_text(&final_body, "TWICE_EXACT_TAIL_SUMMARY_ONE"),
        "final request should not retain the prior summary separately after the second summary replaces it"
    );

    shutdown_codex(&test).await?;
    Ok(())
}
