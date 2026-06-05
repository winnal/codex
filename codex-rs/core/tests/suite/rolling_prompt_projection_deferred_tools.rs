use super::rolling_prompt_support::turn_metadata;
use super::rolling_prompt_support::wait_for_turn_complete;
use anyhow::Result;
use codex_login::CodexAuth;
use codex_protocol::config_types::AutoCompactTokenLimitScope;
use codex_protocol::config_types::PromptRetentionMode;
use codex_protocol::dynamic_tools::DynamicToolSpec;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use codex_utils_output_truncation::approx_token_count;
use core_test_support::apps_test_server::configure_search_capable_model;
use core_test_support::responses;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_mode_tool_search_followup_projects_output_to_item_cap() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let call_id = "rollctx-tool-search-large-output";
    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                responses::ev_response_created("resp-1"),
                responses::ev_tool_search_call(
                    call_id,
                    &json!({
                        "query": "oversized deferred tool",
                        "limit": 8,
                    }),
                ),
                responses::ev_completed_with_tokens("resp-1", /*total_tokens*/ 100_000_000),
            ]),
            sse(vec![
                responses::ev_assistant_message("msg-2", "rolling tool search followup done"),
                responses::ev_completed("resp-2"),
            ]),
        ],
    )
    .await;

    let input_schema = json!({
        "type": "object",
        "properties": {
            "mode": { "type": "string" },
        },
        "required": ["mode"],
        "additionalProperties": false,
    });
    let dynamic_tool = DynamicToolSpec {
        namespace: Some("codex_app".to_string()),
        name: "oversized_dynamic_tool".to_string(),
        description: format!(
            "Oversized deferred tool for rolling projection. {}",
            "ROLLCTX_TOOL_SEARCH ".repeat(20_000)
        ),
        input_schema,
        defer_loading: true,
    };

    let mut builder = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_config(|config| {
            configure_search_capable_model(config);
            config.prompt_retention = PromptRetentionMode::Rolling;
            config.rolling_context_reserve_percent = Some(0);
            config.rolling_context_target_tokens = Some(50_000);
            config.model_auto_compact_token_limit = Some(1);
            config.model_auto_compact_token_limit_scope = AutoCompactTokenLimitScope::Total;
            config.tool_output_token_limit = Some(20_000);
        });
    let mut test = builder.build(&server).await?;
    let new_thread = test
        .thread_manager
        .start_thread_with_tools(test.config.clone(), vec![dynamic_tool])
        .await?;
    test.codex = new_thread.thread;
    test.session_configured = new_thread.session_configured;

    let turn_id = test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "Find the oversized deferred tool under rolling retention".to_string(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&test.codex, &turn_id).await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 2);
    let follow_up = &captured[1];
    let follow_up_metadata = turn_metadata(follow_up)?;
    assert_eq!(follow_up_metadata["request_kind"].as_str(), Some("turn"));
    assert_eq!(
        follow_up_metadata.get("compaction"),
        None,
        "rolling tool-search follow-up request must not be a summary compaction request"
    );
    let output = follow_up.tool_search_output(call_id);
    let tools_json = serde_json::to_string(
        output
            .get("tools")
            .and_then(Value::as_array)
            .expect("tool_search_output should carry a tools array"),
    )?;
    assert!(
        tools_json.contains("original_tool_count"),
        "large tool-search output should be summarized in rolling projection"
    );
    assert!(
        tools_json.contains("tokens truncated"),
        "large tool-search output should be visibly truncated in rolling projection"
    );
    assert!(
        approx_token_count(&tools_json) <= 10_000,
        "rolling prompt projection should keep tool-search output within the per-item cap"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_mode_next_turn_projects_web_search_call_to_item_cap() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let web_search_id = "rollctx-web-search-large-action";
    let query = format!(
        "ROLLCTX_WEB_SEARCH_QUERY {}",
        "web search query ".repeat(30_000)
    );
    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                responses::ev_response_created("resp-1"),
                responses::ev_web_search_call_done(web_search_id, "completed", &query),
                responses::ev_completed("resp-1"),
            ]),
            sse(vec![
                responses::ev_assistant_message("msg-2", "rolling web search projection done"),
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
            config.tool_output_token_limit = Some(20_000);
        })
        .build(&server)
        .await?;

    let first_turn_id = test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "perform a large web search".into(),
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
                text: "inspect the previous web search".into(),
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
    let second_request = &captured[1];
    let metadata = turn_metadata(second_request)?;
    assert_eq!(metadata["request_kind"].as_str(), Some("turn"));
    assert_eq!(
        metadata.get("compaction"),
        None,
        "rolling web-search projection request must not be a summary compaction request"
    );
    let web_search_items = second_request.inputs_of_type("web_search_call");
    assert_eq!(web_search_items.len(), 1);
    let projected_query = web_search_items[0]
        .get("action")
        .and_then(|action| action.get("query"))
        .and_then(Value::as_str)
        .expect("web search action should retain a projected query");
    assert!(
        projected_query.contains("ROLLCTX_WEB_SEARCH_QUERY"),
        "rolling projection should preserve the web-search query sentinel"
    );
    assert!(
        projected_query.contains("tokens truncated"),
        "rolling projection should truncate oversized web-search action"
    );
    assert!(
        approx_token_count(projected_query) <= 10_000,
        "rolling projection should keep web-search action within the per-item cap"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_mode_next_turn_projects_image_generation_result_to_empty() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let image_id = "rollctx-image-large-result";
    let revised_prompt = format!(
        "ROLLCTX_IMAGE_REVISED_PROMPT {}",
        "image prompt ".repeat(30_000)
    );
    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                responses::ev_response_created("resp-1"),
                responses::ev_image_generation_call(
                    image_id,
                    "completed",
                    &revised_prompt,
                    &"ROLLCTX_IMAGE_RESULT ".repeat(30_000),
                ),
                responses::ev_completed("resp-1"),
            ]),
            sse(vec![
                responses::ev_assistant_message("msg-2", "rolling image projection done"),
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
            config.tool_output_token_limit = Some(20_000);
        })
        .build(&server)
        .await?;

    let first_turn_id = test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "generate a large image result".into(),
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
                text: "inspect the previous image call".into(),
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
    let image_items = captured[1].inputs_of_type("image_generation_call");
    assert_eq!(image_items.len(), 1);
    assert_eq!(
        image_items[0].get("id").and_then(Value::as_str),
        Some(image_id)
    );
    assert_eq!(
        image_items[0].get("result").and_then(Value::as_str),
        Some("")
    );
    let revised_prompt = image_items[0]
        .get("revised_prompt")
        .and_then(Value::as_str)
        .expect("image item should retain a revised prompt");
    assert!(
        revised_prompt.contains("ROLLCTX_IMAGE_REVISED_PROMPT"),
        "rolling projection should preserve the revised prompt sentinel"
    );
    assert!(
        revised_prompt.contains("tokens truncated"),
        "rolling projection should truncate oversized revised prompts"
    );
    assert!(
        approx_token_count(revised_prompt) <= 10_000,
        "rolling projection should keep revised prompt within the per-item cap"
    );
    assert!(
        !captured[1].body_contains_text("ROLLCTX_IMAGE_RESULT"),
        "rolling projection should not send partial image result payloads"
    );

    Ok(())
}
