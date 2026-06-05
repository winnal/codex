use super::rolling_prompt_support::turn_metadata;
use super::rolling_prompt_support::wait_for_turn_complete;
use anyhow::Result;
use codex_protocol::config_types::AutoCompactTokenLimitScope;
use codex_protocol::config_types::PromptRetentionMode;
use codex_protocol::models::PermissionProfile;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use codex_utils_output_truncation::approx_token_count;
use core_test_support::responses;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::mount_sse_once_match;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::json;
use wiremock::matchers::body_string_contains;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_mode_skips_mid_turn_auto_compaction_for_tool_followup() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let initial_request = mount_sse_once(
        &server,
        sse(vec![
            responses::ev_shell_command_call("rollctx-shell", "echo rolling-followup"),
            responses::ev_completed_with_tokens("resp-1", /*total_tokens*/ 100_000_000),
        ]),
    )
    .await;
    let follow_up_request = mount_sse_once(
        &server,
        sse(vec![
            responses::ev_assistant_message("msg-2", "rolling followup done"),
            responses::ev_completed("resp-2"),
        ]),
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

    let turn_id = test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "run a command before continuing".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&test.codex, &turn_id).await?;

    let initial_metadata = turn_metadata(&initial_request.single_request())?;
    let follow_up = follow_up_request.single_request();
    let follow_up_metadata = turn_metadata(&follow_up)?;

    assert_eq!(initial_metadata["request_kind"].as_str(), Some("turn"));
    assert_eq!(follow_up_metadata["request_kind"].as_str(), Some("turn"));
    assert_eq!(
        follow_up_metadata.get("compaction"),
        None,
        "rolling follow-up request must not be a summary compaction request"
    );
    let output = follow_up
        .function_call_output_text("rollctx-shell")
        .expect("follow-up request should include shell output");
    assert!(
        output.contains("rolling-followup\n"),
        "follow-up request should preserve shell output, got {output:?}"
    );

    Ok(())
}

#[cfg_attr(windows, ignore = "uses POSIX seq command")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_mode_followup_uses_resolved_tool_output_token_limit() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let call_id = "rollctx-shell-token-limit";
    let args = json!({
        "command": "seq 1 150",
        "timeout_ms": 5_000,
    });
    mount_sse_once(
        &server,
        sse(vec![
            responses::ev_response_created("resp-1"),
            responses::ev_function_call(call_id, "shell_command", &serde_json::to_string(&args)?),
            responses::ev_completed_with_tokens("resp-1", /*total_tokens*/ 100_000_000),
        ]),
    )
    .await;
    let follow_up_request = mount_sse_once(
        &server,
        sse(vec![
            responses::ev_assistant_message("msg-2", "rolling truncated followup done"),
            responses::ev_completed("resp-2"),
        ]),
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
            config.tool_output_token_limit = Some(50);
        })
        .build(&server)
        .await?;

    test.submit_turn_with_permission_profile(
        "run the shell tool under rolling retention",
        PermissionProfile::Disabled,
    )
    .await?;

    let follow_up = follow_up_request.single_request();
    let follow_up_metadata = turn_metadata(&follow_up)?;
    assert_eq!(follow_up_metadata["request_kind"].as_str(), Some("turn"));
    assert_eq!(
        follow_up_metadata.get("compaction"),
        None,
        "rolling follow-up request must not be a summary compaction request"
    );

    let output = follow_up
        .function_call_output_text(call_id)
        .expect("follow-up request should include shell output");
    assert!(
        output.starts_with("Exit code: 0\nWall time: "),
        "tool output should keep the standard shell envelope, got {output:?}"
    );
    assert!(
        output.contains("Total output lines: 150\nOutput:\n1\n2\n3\n"),
        "tool output should keep the beginning of the command output, got {output:?}"
    );
    assert!(
        output.contains("tokens truncated"),
        "tool output should be truncated under the resolved tool output limit, got {output:?}"
    );
    assert!(
        output.ends_with("146\n147\n148\n149\n150\n"),
        "tool output should keep the end of the command output, got {output:?}"
    );
    let output_tokens = approx_token_count(&output);
    assert!(
        output_tokens <= 100,
        "rolling prompt should project the configured small tool output limit, got {output_tokens} tokens in {output:?}"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_mode_retries_context_window_error_without_compaction() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut seed_events = Vec::new();
    for index in 0..120 {
        seed_events.push(responses::ev_assistant_message(
            &format!("seed-msg-{index}"),
            &format!("BACKOFF_DROP_SENTINEL_{index:03} {}", "pad ".repeat(500)),
        ));
    }
    seed_events.push(responses::ev_completed("seed-resp"));
    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(seed_events),
            responses::sse_failed(
                "overflow-resp",
                "context_length_exceeded",
                "Your input exceeds the context window of this model. Please adjust your input and try again.",
            ),
            sse(vec![
                responses::ev_assistant_message("retry-msg", "retry succeeded"),
                responses::ev_completed("retry-resp"),
            ]),
            sse(vec![
                responses::ev_assistant_message("cursor-msg", "cursor check succeeded"),
                responses::ev_completed("cursor-resp"),
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
            config.include_permissions_instructions = false;
            config.include_apps_instructions = false;
            config.include_collaboration_mode_instructions = false;
            config.include_skill_instructions = false;
            config.include_environment_context = false;
        })
        .build(&server)
        .await?;

    let seed_turn_id = test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "seed history before overflow".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&test.codex, &seed_turn_id).await?;

    let rolling_turn_id = test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "trigger rolling retry after context overflow".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&test.codex, &rolling_turn_id).await?;

    let cursor_check_turn_id = test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "check rolling cursor after retry".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&test.codex, &cursor_check_turn_id).await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 4);
    let overflow_metadata = turn_metadata(&captured[1])?;
    let retry_metadata = turn_metadata(&captured[2])?;
    let cursor_metadata = turn_metadata(&captured[3])?;
    let overflow_retained_old_groups = captured[1]
        .body_json()
        .to_string()
        .matches("BACKOFF_DROP_SENTINEL_")
        .count();
    let retry_retained_old_groups = captured[2]
        .body_json()
        .to_string()
        .matches("BACKOFF_DROP_SENTINEL_")
        .count();
    let cursor_retained_old_groups = captured[3]
        .body_json()
        .to_string()
        .matches("BACKOFF_DROP_SENTINEL_")
        .count();

    assert_eq!(overflow_metadata["request_kind"].as_str(), Some("turn"));
    assert_eq!(retry_metadata["request_kind"].as_str(), Some("turn"));
    assert!(
        overflow_retained_old_groups > 0,
        "overflow request should include old groups before backoff"
    );
    assert!(
        retry_retained_old_groups < overflow_retained_old_groups,
        "90% backoff should drop older groups: overflow={overflow_retained_old_groups}, retry={retry_retained_old_groups}"
    );
    assert_eq!(
        retry_metadata.get("compaction"),
        None,
        "rolling context-window retry must not be a summary compaction request"
    );
    assert!(
        captured[2].body_contains_text("trigger rolling retry after context overflow"),
        "retry request should still keep the active frontier"
    );
    assert_eq!(cursor_metadata["request_kind"].as_str(), Some("turn"));
    assert!(
        captured[3].body_contains_text("check rolling cursor after retry"),
        "later request should include the new user input"
    );
    assert!(
        cursor_retained_old_groups <= retry_retained_old_groups,
        "later request must not re-include groups ejected by the committed retry cursor: retry={retry_retained_old_groups}, later={cursor_retained_old_groups}"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_mode_request_body_uses_current_prefix_and_newest_suffix() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let old_body = format!("OLD_BODY_SENTINEL {}", "old ".repeat(20_000));
    let test = test_codex()
        .with_config(|config| {
            config.prompt_retention = PromptRetentionMode::Rolling;
            config.rolling_context_reserve_percent = Some(0);
            config.rolling_context_target_tokens = Some(50_000);
            config.developer_instructions = Some("CURRENT_DEV_SENTINEL".to_string());
            config.include_permissions_instructions = false;
            config.include_apps_instructions = false;
            config.include_collaboration_mode_instructions = false;
            config.include_skill_instructions = false;
            config.include_environment_context = false;
        })
        .build(&server)
        .await?;
    let seed_request = mount_sse_once_match(
        &server,
        body_string_contains("seed old response"),
        sse(vec![
            responses::ev_assistant_message("seed-msg", &old_body),
            responses::ev_completed("seed-resp"),
        ]),
    )
    .await;
    let rolling_request = mount_sse_once_match(
        &server,
        body_string_contains("NEWEST_BODY_SENTINEL"),
        sse(vec![
            responses::ev_assistant_message("roll-msg", "rolled"),
            responses::ev_completed("roll-resp"),
        ]),
    )
    .await;

    let seed_turn_id = test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "seed old response".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&test.codex, &seed_turn_id).await?;

    let rolling_turn_id = test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "NEWEST_BODY_SENTINEL".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&test.codex, &rolling_turn_id).await?;

    seed_request.single_request();
    let rolling_request = rolling_request.single_request();
    assert!(
        rolling_request.body_contains_text("CURRENT_DEV_SENTINEL"),
        "rolling request should include freshly rendered current developer prefix"
    );
    assert!(
        rolling_request.body_contains_text("NEWEST_BODY_SENTINEL"),
        "rolling request should keep newest body suffix"
    );
    assert!(
        !rolling_request.body_contains_text("OLD_BODY_SENTINEL"),
        "rolling request should eject old oversized body prefix"
    );

    Ok(())
}
