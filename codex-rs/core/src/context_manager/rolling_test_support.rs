use super::*;
use crate::context::ContextualUserFragment;
use codex_protocol::models::AgentMessageInputContent;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::LocalShellAction;
use codex_protocol::models::LocalShellExecAction;
use codex_protocol::models::LocalShellStatus;
use codex_utils_output_truncation::TruncationPolicy;

pub(super) fn history(items: Vec<ResponseItem>) -> ContextManager {
    let mut history = ContextManager::new();
    history.record_items(items.iter(), TruncationPolicy::Tokens(50_000));
    history
}

pub(super) fn base_instructions() -> BaseInstructions {
    BaseInstructions {
        text: "base".to_string(),
    }
}

pub(super) fn developer_msg(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "developer".to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
    }
}

pub(super) fn rolling_invariant_developer_msg(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "developer".to_string(),
        content: vec![
            ContentItem::InputText {
                text: crate::context::RollingInvariantDeveloperContext.render(),
            },
            ContentItem::InputText {
                text: text.to_string(),
            },
        ],
        phase: None,
    }
}

pub(super) fn contextual_user_msg(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
    }
}

pub(super) fn user_msg(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
    }
}

pub(super) fn assistant_msg(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText {
            text: text.to_string(),
        }],
        phase: None,
    }
}

pub(super) fn function_call(call_id: &str) -> ResponseItem {
    ResponseItem::FunctionCall {
        id: None,
        name: "shell".to_string(),
        namespace: None,
        arguments: "{}".to_string(),
        call_id: call_id.to_string(),
    }
}

pub(super) fn function_output(call_id: &str, output: &str) -> ResponseItem {
    ResponseItem::FunctionCallOutput {
        call_id: call_id.to_string(),
        output: FunctionCallOutputPayload::from_text(output.to_string()),
    }
}

pub(super) fn local_shell_call(call_id: &str) -> ResponseItem {
    ResponseItem::LocalShellCall {
        id: None,
        call_id: Some(call_id.to_string()),
        status: LocalShellStatus::Completed,
        action: LocalShellAction::Exec(LocalShellExecAction {
            command: vec!["echo".to_string(), "ok".to_string()],
            timeout_ms: None,
            working_directory: None,
            env: None,
            user: None,
        }),
    }
}

pub(super) fn custom_tool_call(call_id: &str) -> ResponseItem {
    ResponseItem::CustomToolCall {
        id: None,
        status: None,
        call_id: call_id.to_string(),
        name: "custom".to_string(),
        input: "input".to_string(),
    }
}

pub(super) fn custom_tool_output(call_id: &str, output: &str) -> ResponseItem {
    ResponseItem::CustomToolCallOutput {
        call_id: call_id.to_string(),
        name: Some("custom".to_string()),
        output: FunctionCallOutputPayload::from_text(output.to_string()),
    }
}

pub(super) fn tool_search_call(call_id: &str) -> ResponseItem {
    ResponseItem::ToolSearchCall {
        id: None,
        call_id: Some(call_id.to_string()),
        status: None,
        execution: "client".to_string(),
        arguments: serde_json::json!({"q": "demo"}),
    }
}

pub(super) fn tool_search_output(call_id: &str) -> ResponseItem {
    tool_search_output_with_tools(call_id, Vec::new())
}

pub(super) fn tool_search_output_with_tools(
    call_id: &str,
    tools: Vec<serde_json::Value>,
) -> ResponseItem {
    ResponseItem::ToolSearchOutput {
        call_id: Some(call_id.to_string()),
        status: "completed".to_string(),
        execution: "client".to_string(),
        tools,
    }
}

pub(super) fn server_tool_search_output(call_id: &str) -> ResponseItem {
    ResponseItem::ToolSearchOutput {
        call_id: Some(call_id.to_string()),
        status: "completed".to_string(),
        execution: "server".to_string(),
        tools: Vec::new(),
    }
}

pub(super) fn oversized_pass_through_items() -> Vec<ResponseItem> {
    vec![
        ResponseItem::AgentMessage {
            author: "agent-a".to_string(),
            recipient: "agent-b".to_string(),
            content: vec![AgentMessageInputContent::EncryptedContent {
                encrypted_content: "agent encrypted ".repeat(30_000),
            }],
        },
        ResponseItem::Reasoning {
            id: "reasoning-1".to_string(),
            summary: Vec::new(),
            content: None,
            encrypted_content: Some("reasoning encrypted ".repeat(30_000)),
        },
        ResponseItem::Compaction {
            encrypted_content: "compaction encrypted ".repeat(30_000),
        },
        ResponseItem::ContextCompaction {
            encrypted_content: Some("context compaction encrypted ".repeat(30_000)),
        },
    ]
}
