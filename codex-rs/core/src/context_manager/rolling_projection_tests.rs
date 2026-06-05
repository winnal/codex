use super::*;
use crate::context::ContextualUserFragment;
use crate::context::ExtensionContextualUserFragment;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::LocalShellAction;
use codex_protocol::models::LocalShellExecAction;
use codex_protocol::models::LocalShellStatus;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::WebSearchAction;
use pretty_assertions::assert_eq;

const ITEM_LIMIT_TOKENS: i64 = 10_000;

fn item_token_estimate(item: &ResponseItem) -> i64 {
    estimate_response_item_token_count(item)
}

fn function_output(call_id: &str, output: &str) -> ResponseItem {
    ResponseItem::FunctionCallOutput {
        call_id: call_id.to_string(),
        output: FunctionCallOutputPayload::from_text(output.to_string()),
    }
}

fn custom_tool_output(call_id: &str, output: &str) -> ResponseItem {
    ResponseItem::CustomToolCallOutput {
        call_id: call_id.to_string(),
        name: Some("custom".to_string()),
        output: FunctionCallOutputPayload::from_text(output.to_string()),
    }
}

fn tool_search_output(call_id: &str, tools: Vec<serde_json::Value>) -> ResponseItem {
    ResponseItem::ToolSearchOutput {
        call_id: Some(call_id.to_string()),
        status: "completed".to_string(),
        execution: "client".to_string(),
        tools,
    }
}

fn function_call(call_id: &str, arguments: &str) -> ResponseItem {
    ResponseItem::FunctionCall {
        id: None,
        name: "echo".to_string(),
        namespace: Some("mcp__rmcp".to_string()),
        arguments: arguments.to_string(),
        call_id: call_id.to_string(),
    }
}

fn custom_tool_call(call_id: &str, input: &str) -> ResponseItem {
    ResponseItem::CustomToolCall {
        id: None,
        status: None,
        call_id: call_id.to_string(),
        name: "exec".to_string(),
        input: input.to_string(),
    }
}

fn tool_search_call(call_id: &str, arguments: serde_json::Value) -> ResponseItem {
    ResponseItem::ToolSearchCall {
        id: None,
        call_id: Some(call_id.to_string()),
        status: None,
        execution: "client".to_string(),
        arguments,
    }
}

fn web_search_call(query: &str) -> ResponseItem {
    ResponseItem::WebSearchCall {
        id: None,
        status: Some("completed".to_string()),
        action: Some(WebSearchAction::Search {
            query: Some(query.to_string()),
            queries: None,
        }),
    }
}

fn local_shell_call(command: Vec<String>) -> ResponseItem {
    ResponseItem::LocalShellCall {
        id: None,
        call_id: Some("local-shell-large-action".to_string()),
        status: LocalShellStatus::Completed,
        action: LocalShellAction::Exec(LocalShellExecAction {
            command,
            timeout_ms: None,
            working_directory: None,
            env: None,
            user: None,
        }),
    }
}

#[test]
fn extension_contextual_user_fragment_is_preserved_and_split_by_rolling_projection() {
    let source_text = format!(
        "{}TAIL_EXTENSION_CONTEXTUAL_USER_SENTINEL",
        "extension context ".repeat(30_000)
    );
    let rendered = ExtensionContextualUserFragment::new(source_text.clone()).render();
    assert!(
        rendered.contains("TAIL_EXTENSION_CONTEXTUAL_USER_SENTINEL"),
        "extension contextual-user rendering must not truncate before rolling projection"
    );
    assert!(
        !rendered.contains("tokens truncated"),
        "extension contextual-user rendering must preserve current context exactly"
    );
    let item = ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText { text: rendered }],
        phase: None,
    };

    let projected = project_rolling_message_item(item, ITEM_LIMIT_TOKENS);
    assert!(
        projected.len() > 1,
        "large extension contextual-user message should be split into bounded items"
    );
    let mut joined = String::new();
    for item in &projected {
        let ResponseItem::Message { content, .. } = item else {
            panic!("expected message projection: {item:?}");
        };
        assert!(
            item_token_estimate(item) <= ITEM_LIMIT_TOKENS,
            "rolling projection must enforce the 10K model-visible item cap"
        );
        for content_item in content {
            let ContentItem::InputText { text } = content_item else {
                panic!("expected input text content: {content_item:?}");
            };
            joined.push_str(text);
        }
    }
    assert!(
        joined.contains(&source_text),
        "rolling projection must preserve the extension contextual-user source text"
    );
}

#[test]
fn project_invariant_item_splits_large_message_without_losing_text() {
    let original_text = "developer prefix ".repeat(30_000);
    let item = ResponseItem::Message {
        id: None,
        role: "developer".to_string(),
        content: vec![ContentItem::InputText {
            text: original_text.clone(),
        }],
        phase: None,
    };

    let projected = project_rolling_message_item(item, ITEM_LIMIT_TOKENS);
    assert!(
        projected.len() > 1,
        "large invariant message should be split into bounded items"
    );
    let mut joined = String::new();
    for item in &projected {
        let ResponseItem::Message { content, .. } = item else {
            panic!("expected message invariant projection: {item:?}");
        };
        assert!(
            item_token_estimate(item) <= ITEM_LIMIT_TOKENS,
            "rolling invariant projection must enforce the 10K model-visible item cap"
        );
        for content_item in content {
            let ContentItem::InputText { text } = content_item else {
                panic!("expected text content: {content:?}");
            };
            joined.push_str(text);
        }
    }
    assert_eq!(joined, original_text);
    assert!(
        !serde_json::to_string(&projected)
            .expect("serialize projected invariant")
            .contains("tokens truncated"),
        "rolling invariant projection must preserve current authority instead of truncating it"
    );
}

#[test]
fn project_generated_context_message_splits_large_message_to_item_cap() {
    let item = ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: "<skill>\n".to_string() + &"skill prompt ".repeat(30_000) + "\n</skill>",
        }],
        phase: None,
    };

    let projected = project_rolling_message_item(item, ITEM_LIMIT_TOKENS);

    assert!(
        projected.len() > 1,
        "large generated turn context should be split into bounded items"
    );
    for item in &projected {
        assert!(
            item_token_estimate(item) <= ITEM_LIMIT_TOKENS,
            "rolling generated-context projection must enforce the 10K model-visible item cap"
        );
    }
}

#[test]
fn project_large_tool_output_to_model_visible_item_cap() {
    let projected = project_rolling_prompt_item(
        function_output("call-large-output", &"tool output ".repeat(30_000)),
        20_000,
        ITEM_LIMIT_TOKENS,
    );

    let ResponseItem::FunctionCallOutput { output, .. } = &projected else {
        panic!("expected function output: {projected:?}");
    };
    let output_text = output
        .text_content()
        .expect("test output should remain text after truncation");
    assert!(
        output_text.contains("tokens truncated"),
        "rolling projection should visibly truncate oversized tool output"
    );
    assert!(
        item_token_estimate(&projected) <= ITEM_LIMIT_TOKENS,
        "rolling projection must enforce the 10K model-visible item cap"
    );
}

#[test]
fn project_large_function_call_arguments_to_model_visible_item_cap() {
    let projected = project_rolling_prompt_item(
        function_call(
            "mcp-large-args",
            &serde_json::json!({
                "message": "mcp argument ".repeat(30_000),
            })
            .to_string(),
        ),
        20_000,
        ITEM_LIMIT_TOKENS,
    );

    let ResponseItem::FunctionCall { arguments, .. } = &projected else {
        panic!("expected function call: {projected:?}");
    };
    assert!(
        arguments.contains("tokens truncated"),
        "rolling projection should visibly truncate oversized function call arguments"
    );
    assert!(
        item_token_estimate(&projected) <= ITEM_LIMIT_TOKENS,
        "rolling projection must enforce the 10K model-visible item cap"
    );
}

#[test]
fn project_large_custom_tool_call_input_to_model_visible_item_cap() {
    let projected = project_rolling_prompt_item(
        custom_tool_call("custom-large-input", &"custom tool input ".repeat(30_000)),
        20_000,
        ITEM_LIMIT_TOKENS,
    );

    let ResponseItem::CustomToolCall { input, .. } = &projected else {
        panic!("expected custom tool call: {projected:?}");
    };
    assert!(
        input.contains("tokens truncated"),
        "rolling projection should visibly truncate oversized custom tool call input"
    );
    assert!(
        item_token_estimate(&projected) <= ITEM_LIMIT_TOKENS,
        "rolling projection must enforce the 10K model-visible item cap"
    );
}

#[test]
fn project_large_tool_search_call_arguments_to_model_visible_item_cap() {
    let projected = project_rolling_prompt_item(
        tool_search_call(
            "tool-search-large-args",
            serde_json::json!({
                "query": "tool search argument ".repeat(30_000),
            }),
        ),
        20_000,
        ITEM_LIMIT_TOKENS,
    );

    let ResponseItem::ToolSearchCall { arguments, .. } = &projected else {
        panic!("expected tool-search call: {projected:?}");
    };
    let arguments_json = serde_json::to_string(arguments).expect("serialize projected args");
    assert!(
        arguments_json.contains("truncated_tool_search_arguments"),
        "rolling projection should summarize oversized tool-search arguments"
    );
    assert!(
        arguments_json.contains("tokens truncated"),
        "rolling projection should visibly truncate oversized tool-search arguments"
    );
    assert!(
        item_token_estimate(&projected) <= ITEM_LIMIT_TOKENS,
        "rolling projection must enforce the 10K model-visible item cap"
    );
}

#[test]
fn project_large_web_search_call_to_model_visible_item_cap() {
    let projected = project_rolling_prompt_item(
        web_search_call(&"web search query ".repeat(30_000)),
        20_000,
        ITEM_LIMIT_TOKENS,
    );

    let ResponseItem::WebSearchCall {
        action:
            Some(WebSearchAction::Search {
                query: Some(query),
                queries: None,
            }),
        ..
    } = &projected
    else {
        panic!("expected projected web-search query: {projected:?}");
    };
    assert!(
        query.contains("tokens truncated"),
        "rolling projection should visibly truncate oversized web-search actions"
    );
    assert!(
        item_token_estimate(&projected) <= ITEM_LIMIT_TOKENS,
        "rolling projection must enforce the 10K model-visible item cap"
    );
}

#[test]
fn project_large_local_shell_call_to_model_visible_item_cap() {
    let projected = project_rolling_prompt_item(
        local_shell_call(vec!["local shell command ".repeat(30_000)]),
        20_000,
        ITEM_LIMIT_TOKENS,
    );

    let ResponseItem::LocalShellCall { action, .. } = &projected else {
        panic!("expected projected local shell call: {projected:?}");
    };
    let LocalShellAction::Exec(action) = action;
    assert!(
        action.command.join(" ").contains("tokens truncated"),
        "rolling projection should visibly truncate oversized local shell actions"
    );
    assert!(
        item_token_estimate(&projected) <= ITEM_LIMIT_TOKENS,
        "rolling projection must enforce the 10K model-visible item cap"
    );
}

#[test]
fn project_large_custom_tool_output_to_model_visible_item_cap() {
    let projected = project_rolling_prompt_item(
        custom_tool_output("custom-large-output", &"custom output ".repeat(30_000)),
        20_000,
        ITEM_LIMIT_TOKENS,
    );

    let ResponseItem::CustomToolCallOutput { output, .. } = &projected else {
        panic!("expected custom tool output: {projected:?}");
    };
    let output_text = output
        .text_content()
        .expect("test output should remain text after truncation");
    assert!(
        output_text.contains("tokens truncated"),
        "rolling projection should visibly truncate oversized custom tool output"
    );
    assert!(
        item_token_estimate(&projected) <= ITEM_LIMIT_TOKENS,
        "rolling projection must enforce the 10K model-visible item cap"
    );
}

#[test]
fn project_large_tool_search_output_to_model_visible_item_cap() {
    let tools = (0..600)
        .map(|index| {
            serde_json::json!({
                "name": format!("tool_{index}"),
                "description": "tool search output ".repeat(80),
            })
        })
        .collect::<Vec<_>>();

    let projected = project_rolling_prompt_item(
        tool_search_output("tool-search-large-output", tools),
        20_000,
        ITEM_LIMIT_TOKENS,
    );

    let ResponseItem::ToolSearchOutput { tools, .. } = &projected else {
        panic!("expected tool-search output: {projected:?}");
    };
    let tools_json = serde_json::to_string(tools).expect("serialize projected tools");
    assert!(
        tools_json.contains("truncated_tool_search_output"),
        "rolling projection should summarize oversized tool-search output"
    );
    assert!(
        tools_json.contains("tokens truncated"),
        "rolling projection should visibly truncate oversized tool-search output"
    );
    assert!(
        item_token_estimate(&projected) <= ITEM_LIMIT_TOKENS,
        "rolling projection must enforce the 10K model-visible item cap"
    );
}

#[test]
fn project_large_image_generation_result_to_empty_model_visible_result() {
    let projected = project_rolling_prompt_item(
        ResponseItem::ImageGenerationCall {
            id: "img-1".to_string(),
            status: "completed".to_string(),
            revised_prompt: Some("revised".to_string()),
            result: "base64-image-result ".repeat(30_000),
        },
        20_000,
        ITEM_LIMIT_TOKENS,
    );

    let ResponseItem::ImageGenerationCall { result, .. } = &projected else {
        panic!("expected image generation call: {projected:?}");
    };
    assert_eq!(result, "");
    assert!(
        item_token_estimate(&projected) <= ITEM_LIMIT_TOKENS,
        "rolling projection must enforce the 10K model-visible item cap"
    );
}

#[test]
fn project_large_image_generation_revised_prompt_to_model_visible_item_cap() {
    let projected = project_rolling_prompt_item(
        ResponseItem::ImageGenerationCall {
            id: "img-large-prompt".to_string(),
            status: "completed".to_string(),
            revised_prompt: Some("image revised prompt ".repeat(30_000)),
            result: "small-image-result".to_string(),
        },
        20_000,
        ITEM_LIMIT_TOKENS,
    );

    let ResponseItem::ImageGenerationCall {
        revised_prompt,
        result,
        ..
    } = &projected
    else {
        panic!("expected image generation call: {projected:?}");
    };
    let revised_prompt = revised_prompt.as_deref().expect("projected prompt");
    assert!(
        revised_prompt.contains("tokens truncated"),
        "rolling projection should visibly truncate oversized revised prompts"
    );
    assert_eq!(result, "small-image-result");
    assert!(
        item_token_estimate(&projected) <= ITEM_LIMIT_TOKENS,
        "rolling projection must enforce the 10K model-visible item cap"
    );
}
