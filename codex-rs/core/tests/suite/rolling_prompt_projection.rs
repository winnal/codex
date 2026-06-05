use super::rolling_prompt_support::turn_metadata;
use super::rolling_prompt_support::wait_for_turn_complete;
use anyhow::Context;
use anyhow::Result;
use codex_protocol::config_types::AutoCompactTokenLimitScope;
use codex_protocol::config_types::CollaborationMode;
use codex_protocol::config_types::ModeKind;
use codex_protocol::config_types::PromptRetentionMode;
use codex_protocol::config_types::Settings;
use codex_protocol::models::PermissionProfile;
use codex_protocol::protocol::AdditionalContextEntry;
use codex_protocol::protocol::AdditionalContextKind;
use codex_protocol::protocol::AskForApproval;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use codex_utils_output_truncation::approx_token_count;
use core_test_support::responses;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use core_test_support::test_codex::turn_permission_fields;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::collections::BTreeMap;

#[cfg_attr(windows, ignore = "uses POSIX yes/head commands")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_mode_followup_projects_twenty_k_tool_output_to_item_cap() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let call_id = "rollctx-shell-twenty-k-token-limit";
    let args = json!({
        "command": "yes 'ROLLCTX20K token token token token token token token token token token' | head -n 12000",
        "timeout_ms": 10_000,
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
            responses::ev_assistant_message("msg-2", "rolling 20k projected followup done"),
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
            config.tool_output_token_limit = Some(20_000);
        })
        .build(&server)
        .await?;

    test.submit_turn_with_permission_profile(
        "run the shell tool under rolling retention with a large output",
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
        output.contains("ROLLCTX20K"),
        "follow-up request should preserve the projected shell output sentinel"
    );
    assert!(
        output.contains("tokens truncated"),
        "large output should be truncated in the rolling prompt projection"
    );
    assert!(
        approx_token_count(&output) <= 10_000,
        "rolling prompt projection should keep output text within the per-item cap"
    );

    Ok(())
}

#[cfg_attr(windows, ignore = "uses POSIX yes/head commands")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_mode_followup_projects_default_tool_output_limit_to_item_cap() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let call_id = "rollctx-shell-default-token-limit";
    let args = json!({
        "command": "yes 'ROLLCTXDEFAULT token token token token token token token token token token' | head -n 12000",
        "timeout_ms": 10_000,
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
            responses::ev_assistant_message("msg-2", "rolling default projected followup done"),
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
        })
        .build(&server)
        .await?;

    test.submit_turn_with_permission_profile(
        "run the shell tool under rolling retention with default output limits",
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
        output.contains("ROLLCTXDEFAULT"),
        "follow-up request should preserve the projected shell output sentinel"
    );
    assert!(
        output.contains("tokens truncated"),
        "default output limit should still truncate oversized rolling output"
    );
    assert!(
        approx_token_count(&output) <= 10_000,
        "rolling prompt projection should keep default-capped output within the per-item cap"
    );

    Ok(())
}

#[cfg_attr(windows, ignore = "uses POSIX sleep command")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_mode_defers_queued_user_input_and_context_until_tool_followup_preserves_output()
-> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let call_id = "rollctx-pending-input-tool-output";
    let args = json!({
        "command": "python3 -c 'import time; time.sleep(0.5); print(\"ROLLCTX_PENDING_TOOL_OUTPUT \" * 20000)'",
        "timeout_ms": 10_000,
    });
    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                responses::ev_response_created("resp-1"),
                responses::ev_function_call(
                    call_id,
                    "shell_command",
                    &serde_json::to_string(&args)?,
                ),
                responses::ev_completed_with_tokens("resp-1", /*total_tokens*/ 100_000_000),
            ]),
            sse(vec![
                responses::ev_assistant_message("msg-2", "rolling tool continuation done"),
                responses::ev_completed("resp-2"),
            ]),
            sse(vec![
                responses::ev_assistant_message("msg-3", "rolling queued prompt done"),
                responses::ev_completed("resp-3"),
            ]),
        ],
    )
    .await;

    let test = test_codex()
        .with_model("gpt-5.4")
        .with_config(|config| {
            config.base_instructions = Some("base".to_string());
            config.prompt_retention = PromptRetentionMode::Rolling;
            config.rolling_context_reserve_percent = Some(0);
            config.rolling_context_target_tokens = Some(12_000);
            config.model_auto_compact_token_limit = Some(1);
            config.model_auto_compact_token_limit_scope = AutoCompactTokenLimitScope::Total;
            config.tool_output_token_limit = Some(20_000);
            config.include_permissions_instructions = false;
            config.include_apps_instructions = false;
            config.include_collaboration_mode_instructions = false;
            config.include_skill_instructions = false;
            config.include_environment_context = false;
        })
        .build(&server)
        .await?;

    let (sandbox_policy, permission_profile) =
        turn_permission_fields(PermissionProfile::Disabled, test.config.cwd.as_path());
    let session_model = test.session_configured.model.clone();
    let first_turn_id = test
        .codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "run a delayed large shell output under rolling retention".into(),
                text_elements: Vec::new(),
            }],
            environments: None,
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: codex_protocol::protocol::ThreadSettingsOverrides {
                cwd: Some(test.config.cwd.to_path_buf()),
                approval_policy: Some(AskForApproval::Never),
                sandbox_policy: Some(sandbox_policy),
                permission_profile,
                collaboration_mode: Some(CollaborationMode {
                    mode: ModeKind::Default,
                    settings: Settings {
                        model: session_model,
                        reasoning_effort: None,
                        developer_instructions: None,
                    },
                }),
                ..Default::default()
            },
        })
        .await?;

    wait_for_event(
        &test.codex,
        |event| matches!(event, EventMsg::ExecCommandBegin(begin) if begin.call_id == call_id),
    )
    .await;
    test.codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: format!("ROLLCTX_QUEUED_USER_PROMPT {}", "queued ".repeat(2_000)),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: BTreeMap::from([(
                "queued_context".to_string(),
                AdditionalContextEntry {
                    value: "ROLLCTX_QUEUED_ADDITIONAL_CONTEXT".to_string(),
                    kind: AdditionalContextKind::Untrusted,
                },
            )]),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&test.codex, &first_turn_id).await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 3);
    let tool_follow_up = &captured[1];
    let output = tool_follow_up
        .function_call_output_text(call_id)
        .context("tool continuation should include the shell output before queued input")?;
    assert!(
        output.contains("ROLLCTX_PENDING_TOOL_OUTPUT"),
        "tool continuation should preserve the projected shell output"
    );
    assert!(
        !tool_follow_up.body_contains_text("ROLLCTX_QUEUED_USER_PROMPT"),
        "rolling tool continuation should not drain newer user input first"
    );
    assert!(
        !tool_follow_up.body_contains_text("ROLLCTX_QUEUED_ADDITIONAL_CONTEXT"),
        "rolling tool continuation should not detach queued additional_context from queued user input"
    );
    assert!(
        captured[2].body_contains_text("ROLLCTX_QUEUED_USER_PROMPT"),
        "queued input should be processed after the tool continuation"
    );
    assert!(
        captured[2].body_contains_text("ROLLCTX_QUEUED_ADDITIONAL_CONTEXT"),
        "queued additional_context should be processed with its queued input"
    );

    Ok(())
}
