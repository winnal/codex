use super::rolling_prompt_support::turn_metadata;
use super::rolling_prompt_support::wait_for_turn_complete;
use anyhow::Result;
use codex_protocol::config_types::AutoCompactTokenLimitScope;
use codex_protocol::config_types::PromptRetentionMode;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use core_test_support::responses;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::submit_thread_settings;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_mode_skips_pre_turn_auto_compaction_for_next_turn() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                responses::ev_assistant_message("msg-1", "seed turn done"),
                responses::ev_completed_with_tokens("resp-1", /*total_tokens*/ 100_000_000),
            ]),
            sse(vec![
                responses::ev_assistant_message("msg-2", "second turn done"),
                responses::ev_completed("resp-2"),
            ]),
        ],
    )
    .await;

    let test = test_codex()
        .with_config(|config| {
            config.prompt_retention = PromptRetentionMode::Rolling;
            config.rolling_context_reserve_percent = Some(0);
            config.rolling_context_target_tokens = Some(50_000);
            config.model_auto_compact_token_limit = Some(1);
            config.model_auto_compact_token_limit_scope = AutoCompactTokenLimitScope::Total;
        })
        .build(&server)
        .await?;

    let first_turn_id = test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "seed rolling history".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&test.codex, &first_turn_id).await?;

    let second_turn_id = test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "SECOND_TURN_SENTINEL".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&test.codex, &second_turn_id).await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 2);
    let first_metadata = turn_metadata(&captured[0])?;
    let second_metadata = turn_metadata(&captured[1])?;
    assert_eq!(first_metadata["request_kind"].as_str(), Some("turn"));
    assert_eq!(second_metadata["request_kind"].as_str(), Some("turn"));
    assert_eq!(
        second_metadata.get("compaction"),
        None,
        "rolling pre-turn sampling must not issue a summary compaction request"
    );
    assert!(
        captured[1].body_contains_text("SECOND_TURN_SENTINEL"),
        "second turn should still be sampled as a regular rolling turn"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_mode_skips_model_downshift_pre_turn_compaction() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                responses::ev_assistant_message("msg-1", "large model turn done"),
                responses::ev_completed_with_tokens("resp-1", /*total_tokens*/ 100_000_000),
            ]),
            sse(vec![
                responses::ev_assistant_message("msg-2", "smaller model turn done"),
                responses::ev_completed("resp-2"),
            ]),
        ],
    )
    .await;

    let test = test_codex()
        .with_model("gpt-5.4")
        .with_config(|config| {
            config.prompt_retention = PromptRetentionMode::Rolling;
            config.rolling_context_reserve_percent = Some(0);
            config.rolling_context_target_tokens = Some(50_000);
            config.model_auto_compact_token_limit = Some(1);
            config.model_auto_compact_token_limit_scope = AutoCompactTokenLimitScope::Total;
        })
        .build(&server)
        .await?;

    let first_turn_id = test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "seed large-model rolling history".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&test.codex, &first_turn_id).await?;

    submit_thread_settings(
        &test.codex,
        codex_protocol::protocol::ThreadSettingsOverrides {
            model: Some("gpt-5.2".to_string()),
            ..Default::default()
        },
    )
    .await?;

    let second_turn_id = test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "DOWNSHIFT_SECOND_TURN_SENTINEL".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&test.codex, &second_turn_id).await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 2);
    let second_metadata = turn_metadata(&captured[1])?;
    assert_eq!(second_metadata["request_kind"].as_str(), Some("turn"));
    assert_eq!(
        second_metadata.get("compaction"),
        None,
        "rolling model downshift must not issue a summary compaction request"
    );
    assert!(
        captured[1].body_contains_text("DOWNSHIFT_SECOND_TURN_SENTINEL"),
        "downshifted rolling turn should still sample the new user input"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_mode_manual_compact_still_issues_compaction_request() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let first_request = mount_sse_once(
        &server,
        sse(vec![
            responses::ev_assistant_message("msg-1", "first turn done"),
            responses::ev_completed("resp-1"),
        ]),
    )
    .await;
    let compact_request =
        responses::mount_compact_user_history_with_summary_once(&server, "manual compact summary")
            .await;

    let test = test_codex()
        .with_config(|config| {
            config.prompt_retention = PromptRetentionMode::Rolling;
            config.model_auto_compact_token_limit = Some(1);
            config.model_auto_compact_token_limit_scope = AutoCompactTokenLimitScope::Total;
        })
        .build(&server)
        .await?;

    let turn_id = test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "first regular rolling turn".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&test.codex, &turn_id).await?;

    let compact_turn_id = test.codex.submit(Op::Compact).await?;
    wait_for_turn_complete(&test.codex, &compact_turn_id).await?;

    let first_metadata = turn_metadata(&first_request.single_request())?;
    let compact_request = compact_request.single_request();
    let compact_metadata = turn_metadata(&compact_request)?;
    assert_eq!(first_metadata["request_kind"].as_str(), Some("turn"));
    assert_eq!(
        compact_metadata["request_kind"].as_str(),
        Some("compaction")
    );
    assert_eq!(
        compact_metadata["compaction"]["trigger"].as_str(),
        Some("manual")
    );
    assert_eq!(
        compact_metadata["compaction"]["reason"].as_str(),
        Some("user_requested")
    );
    assert!(
        matches!(
            compact_metadata["compaction"]["implementation"].as_str(),
            Some("responses_compact") | Some("responses_compaction_v2")
        ),
        "manual compact should use a compact implementation, got {compact_metadata}"
    );
    assert_eq!(
        compact_metadata["compaction"]["phase"].as_str(),
        Some("standalone_turn")
    );
    assert_eq!(
        compact_metadata["compaction"]["strategy"].as_str(),
        Some("memento")
    );

    Ok(())
}
