use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::LocalShellAction;
use codex_protocol::models::LocalShellExecAction;
use codex_protocol::models::LocalShellStatus;
use codex_protocol::models::ResponseItem;
use codex_tools::ToolName;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::EXACT_TAIL_TOOL_SURFACE_REFERENCE_LIMIT;
use super::ExactTailToolSurfaceHint;
use super::derive_exact_tail_tool_surface_hint;

#[test]
fn exact_tail_hint_extracts_current_reference_names_from_hot_suffix() {
    let hot_suffix = vec![
        ResponseItem::FunctionCall {
            id: None,
            name: "read".to_string(),
            namespace: Some("repo".to_string()),
            arguments: "{}".to_string(),
            call_id: "call-read".to_string(),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::CustomToolCall {
            id: None,
            status: None,
            call_id: "call-custom".to_string(),
            name: "freeform_tool".to_string(),
            input: "{}".to_string(),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: "call-read".to_string(),
            output: FunctionCallOutputPayload::from_text("ok".to_string()),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::CustomToolCallOutput {
            id: None,
            call_id: "call-custom".to_string(),
            name: Some("freeform_tool".to_string()),
            output: FunctionCallOutputPayload::from_text("ok".to_string()),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::ToolSearchCall {
            id: None,
            call_id: Some("call-search".to_string()),
            status: None,
            execution: "search".to_string(),
            arguments: json!({"q": "repo"}),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::ToolSearchOutput {
            id: None,
            call_id: Some("call-search".to_string()),
            status: "completed".to_string(),
            execution: "search".to_string(),
            tools: vec![
                json!({
                    "type": "function",
                    "name": "standalone_lookup",
                    "description": "Historical description should not be authoritative.",
                    "parameters": {"type": "object"},
                }),
                json!({
                    "type": "namespace",
                    "name": "repo",
                    "description": "Historical namespace should not be authoritative.",
                    "tools": [{
                        "type": "function",
                        "name": "write",
                        "description": "Historical schema should be ignored.",
                        "parameters": {"type": "object"},
                    }],
                }),
            ],
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::LocalShellCall {
            id: None,
            call_id: Some("call-shell".to_string()),
            status: LocalShellStatus::Completed,
            action: LocalShellAction::Exec(LocalShellExecAction {
                command: vec!["echo".to_string(), "ok".to_string()],
                timeout_ms: None,
                working_directory: None,
                env: None,
                user: None,
            }),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::WebSearchCall {
            id: None,
            status: Some("completed".to_string()),
            action: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::ImageGenerationCall {
            id: None,
            status: "completed".to_string(),
            revised_prompt: None,
            result: String::new(),
            internal_chat_message_metadata_passthrough: None,
        },
    ];

    assert_eq!(
        derive_exact_tail_tool_surface_hint(&hot_suffix),
        ExactTailToolSurfaceHint {
            references: vec![
                ToolName::plain("standalone_lookup"),
                ToolName::namespaced("repo", "write"),
                ToolName::plain("freeform_tool"),
                ToolName::namespaced("repo", "read"),
            ],
            hot_tool_call_count: 6,
            hot_tool_namespace_count: 1,
            hot_tool_reference_count: 4,
            hot_tool_reference_overflow_count: 0,
            out_of_scope_dependency_protocol_count: 5,
        }
    );
}

#[test]
fn exact_tail_hint_caps_references_newest_first() {
    let hot_suffix = (0..EXACT_TAIL_TOOL_SURFACE_REFERENCE_LIMIT + 2)
        .map(|index| ResponseItem::FunctionCall {
            id: None,
            name: format!("tool_{index}"),
            namespace: None,
            arguments: "{}".to_string(),
            call_id: format!("call-{index}"),
            internal_chat_message_metadata_passthrough: None,
        })
        .collect::<Vec<_>>();

    let hint = derive_exact_tail_tool_surface_hint(&hot_suffix);

    assert_eq!(
        hint.references.len(),
        EXACT_TAIL_TOOL_SURFACE_REFERENCE_LIMIT
    );
    assert_eq!(
        hint.references.first(),
        Some(&ToolName::plain(format!(
            "tool_{}",
            EXACT_TAIL_TOOL_SURFACE_REFERENCE_LIMIT + 1
        )))
    );
    assert_eq!(hint.references.last(), Some(&ToolName::plain("tool_2")));
    assert_eq!(
        hint.hot_tool_reference_count,
        EXACT_TAIL_TOOL_SURFACE_REFERENCE_LIMIT + 2
    );
    assert_eq!(hint.hot_tool_reference_overflow_count, 2);
}
