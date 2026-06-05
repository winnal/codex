use super::rolling_prompt_support::turn_metadata;
use super::rolling_prompt_support::wait_for_turn_complete;
use anyhow::Context;
use anyhow::Result;
use codex_config::types::McpServerConfig;
use codex_config::types::McpServerTransportConfig;
use codex_features::Feature;
use codex_protocol::config_types::AutoCompactTokenLimitScope;
use codex_protocol::config_types::PromptRetentionMode;
use codex_protocol::models::PermissionProfile;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use codex_utils_output_truncation::approx_token_count;
use core_test_support::responses;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::stdio_server_bin;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_mcp_server;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn rolling_mode_mcp_followup_projects_tool_output_to_item_cap() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let call_id = "rollctx-rmcp-large-output";
    let server_name = "rmcp";
    let namespace = format!("mcp__{server_name}");
    let large_msg = "ROLLCTX_MCP_OUTPUT ".repeat(4_000);
    let args_json = json!({ "message": large_msg });

    mount_sse_once(
        &server,
        sse(vec![
            responses::ev_response_created("resp-1"),
            responses::ev_function_call_with_namespace(
                call_id,
                &namespace,
                "echo",
                &args_json.to_string(),
            ),
            responses::ev_completed_with_tokens("resp-1", /*total_tokens*/ 100_000_000),
        ]),
    )
    .await;
    let follow_up_request = mount_sse_once(
        &server,
        sse(vec![
            responses::ev_assistant_message("msg-2", "rolling mcp projected followup done"),
            responses::ev_completed("resp-2"),
        ]),
    )
    .await;

    let rmcp_test_server_bin = stdio_server_bin()?;
    let mut builder = test_codex().with_config(move |config| {
        config.prompt_retention = PromptRetentionMode::Rolling;
        config.rolling_context_reserve_percent = Some(0);
        config.rolling_context_target_tokens = Some(50_000);
        config.model_auto_compact_token_limit = Some(1);
        config.model_auto_compact_token_limit_scope = AutoCompactTokenLimitScope::Total;
        config.tool_output_token_limit = Some(20_000);
        let mut servers = config.mcp_servers.get().clone();
        servers.insert(
            server_name.to_string(),
            McpServerConfig {
                transport: McpServerTransportConfig::Stdio {
                    command: rmcp_test_server_bin,
                    args: Vec::new(),
                    env: None,
                    env_vars: Vec::new(),
                    cwd: None,
                },
                environment_id: "local".to_string(),
                enabled: true,
                required: false,
                supports_parallel_tool_calls: false,
                disabled_reason: None,
                startup_timeout_sec: Some(Duration::from_secs(10)),
                tool_timeout_sec: None,
                default_tools_approval_mode: None,
                enabled_tools: None,
                disabled_tools: None,
                scopes: None,
                oauth: None,
                oauth_resource: None,
                tools: HashMap::new(),
            },
        );
        config
            .mcp_servers
            .set(servers)
            .expect("test mcp servers should accept any configuration");
    });
    let fixture = builder.build(&server).await?;
    wait_for_mcp_server(&fixture.codex, server_name).await?;

    let turn_id = fixture
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "call the rmcp echo tool with a large message under rolling retention"
                    .to_string(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    tokio::time::timeout(
        Duration::from_secs(30),
        wait_for_turn_complete(&fixture.codex, &turn_id),
    )
    .await
    .context("rolling MCP turn should complete within the test timeout")??;

    let mut follow_up = None;
    for _ in 0..50 {
        if let Some(request) = follow_up_request.last_request() {
            follow_up = Some(request);
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let follow_up = follow_up.expect("rolling MCP follow-up request should be captured");
    let follow_up_metadata = turn_metadata(&follow_up)?;
    assert_eq!(follow_up_metadata["request_kind"].as_str(), Some("turn"));
    assert_eq!(
        follow_up_metadata.get("compaction"),
        None,
        "rolling MCP follow-up request must not be a summary compaction request"
    );
    let output = follow_up
        .function_call_output_text(call_id)
        .context("follow-up request should include MCP tool output")?;
    assert!(
        output.contains("ROLLCTX_MCP_OUTPUT"),
        "follow-up request should preserve the projected MCP output sentinel"
    );
    assert!(
        output.contains("tokens truncated"),
        "large MCP output should be truncated in the rolling prompt projection"
    );
    assert!(
        approx_token_count(&output) <= 10_000,
        "rolling prompt projection should keep MCP output within the per-item cap"
    );

    Ok(())
}

#[cfg_attr(
    windows,
    ignore = "code mode exec is not covered on Windows in this suite"
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_mode_code_mode_followup_projects_custom_tool_output_to_item_cap() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let call_id = "rollctx-code-mode-large-output";
    let code = r#"notify("ROLLCTX_CODE_NOTIFY"); text("ROLLCTX_CODE_MODE_OUTPUT ".repeat(30000));"#;
    mount_sse_once(
        &server,
        sse(vec![
            responses::ev_response_created("resp-1"),
            responses::ev_custom_tool_call(call_id, "exec", code),
            responses::ev_completed_with_tokens("resp-1", /*total_tokens*/ 100_000_000),
        ]),
    )
    .await;
    let follow_up_request = mount_sse_once(
        &server,
        sse(vec![
            responses::ev_assistant_message("msg-2", "rolling code mode projected followup done"),
            responses::ev_completed("resp-2"),
        ]),
    )
    .await;

    let test = test_codex()
        .with_model("test-gpt-5.1-codex")
        .with_config(|config| {
            config
                .features
                .enable(Feature::CodeMode)
                .expect("test config should allow feature update");
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
        "run code mode under rolling retention with a large output",
        PermissionProfile::Disabled,
    )
    .await?;

    let follow_up = follow_up_request.single_request();
    let follow_up_metadata = turn_metadata(&follow_up)?;
    assert_eq!(follow_up_metadata["request_kind"].as_str(), Some("turn"));
    assert_eq!(
        follow_up_metadata.get("compaction"),
        None,
        "rolling code-mode follow-up request must not be a summary compaction request"
    );
    let output_items = follow_up
        .inputs_of_type("custom_tool_call_output")
        .into_iter()
        .filter(|item| item.get("call_id").and_then(Value::as_str) == Some(call_id))
        .collect::<Vec<_>>();
    assert_eq!(
        output_items.len(),
        2,
        "rolling code-mode follow-up should retain notify and final custom outputs"
    );
    let output = output_items
        .iter()
        .map(|output_item| match output_item.get("output") {
            Some(Value::String(text)) => text.clone(),
            Some(Value::Object(object)) => object
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            Some(Value::Array(items)) => items
                .iter()
                .filter_map(|item| item.get("text").and_then(Value::as_str))
                .collect::<String>(),
            _ => String::new(),
        })
        .collect::<String>();
    assert!(
        output.contains("ROLLCTX_CODE_NOTIFY"),
        "follow-up request should preserve the code-mode notify output sentinel"
    );
    assert!(
        output.contains("ROLLCTX_CODE_MODE_OUTPUT"),
        "follow-up request should preserve the projected code-mode output sentinel"
    );
    assert!(
        output.contains("tokens truncated"),
        "large code-mode output should be truncated in the rolling prompt projection"
    );
    assert!(
        approx_token_count(&output) <= 10_000,
        "rolling prompt projection should keep code-mode output within the per-item cap"
    );

    Ok(())
}
