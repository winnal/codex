use super::*;
use crate::context_manager::estimate_response_items_token_count;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::MessagePhase;
use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::models::ResponseItem;
use pretty_assertions::assert_eq;

fn message(role: &str, text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: Some(format!("{role}-id")),
        role: role.to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: Some(Default::default()),
    }
}

#[test]
fn semantic_transcript_strips_protocol_ids_and_keeps_tool_observation() {
    let cold_history = vec![
        message("user", "SEM_USER_DIRECTIVE"),
        ResponseItem::Message {
            id: Some("assistant-id".to_string()),
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: "SEM_ASSISTANT_VISIBLE".to_string(),
            }],
            phase: Some(MessagePhase::Commentary),
            internal_chat_message_metadata_passthrough: Some(Default::default()),
        },
        ResponseItem::FunctionCall {
            id: Some("fc-id".to_string()),
            name: "shell_command".to_string(),
            namespace: Some("functions".to_string()),
            arguments: "{\"command\":\"rg SEM\"}".to_string(),
            call_id: "call_sem".to_string(),
            internal_chat_message_metadata_passthrough: Some(Default::default()),
        },
        ResponseItem::FunctionCallOutput {
            id: Some("out-id".to_string()),
            call_id: "call_sem".to_string(),
            output: FunctionCallOutputPayload::from_text("SEM_TOOL_OUTPUT".to_string()),
            internal_chat_message_metadata_passthrough: Some(Default::default()),
        },
    ];

    let transcript = render_semantic_transcript(&cold_history, 10_000).expect("render transcript");

    assert_eq!(transcript.tool_observation_count, 1);
    assert_eq!(transcript.items.len(), 3);
    let serialized = serde_json::to_string(&transcript.items).expect("serialize transcript");
    assert!(serialized.contains("SEM_USER_DIRECTIVE"));
    assert!(serialized.contains("SEM_ASSISTANT_VISIBLE"));
    assert!(serialized.contains("SEM_TOOL_OUTPUT"));
    assert!(serialized.contains("functions.shell_command"));
    assert!(!serialized.contains("call_sem"));
    assert!(!serialized.contains("fc-id"));
    assert!(!serialized.contains("out-id"));
}

#[test]
fn semantic_transcript_keeps_prior_summaries_as_cold_summary_entries() {
    let cold_history = vec![
        ResponseItem::Compaction {
            id: Some("summary-id".to_string()),
            encrypted_content: "SEM_PRIOR_SUMMARY".to_string(),
            internal_chat_message_metadata_passthrough: Some(Default::default()),
        },
        ResponseItem::ContextCompaction {
            id: Some("context-summary-id".to_string()),
            encrypted_content: Some("SEM_CONTEXT_SUMMARY".to_string()),
            internal_chat_message_metadata_passthrough: Some(Default::default()),
        },
    ];

    let transcript = render_semantic_transcript(&cold_history, 10_000).expect("render transcript");

    let serialized = serde_json::to_string(&transcript.items).expect("serialize transcript");
    assert!(serialized.contains("SEM_PRIOR_SUMMARY"));
    assert!(serialized.contains("SEM_CONTEXT_SUMMARY"));
    assert!(!serialized.contains("summary-id"));
    assert!(!serialized.contains("context-summary-id"));
}

#[test]
fn semantic_transcript_bounds_large_tool_outputs() {
    let cold_history = vec![
        ResponseItem::FunctionCall {
            id: None,
            name: "shell_command".to_string(),
            namespace: None,
            arguments: "{}".to_string(),
            call_id: "call_large".to_string(),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: "call_large".to_string(),
            output: FunctionCallOutputPayload::from_text("word ".repeat(10_000)),
            internal_chat_message_metadata_passthrough: None,
        },
    ];

    let transcript = render_semantic_transcript(&cold_history, 512).expect("render transcript");
    let serialized = serde_json::to_string(&transcript.items).expect("serialize transcript");
    assert!(serialized.contains("tokens truncated"));

    for item in &transcript.items {
        assert!(
            estimate_response_items_token_count(std::slice::from_ref(item)) <= 512,
            "transcript item should fit cap: {item:?}"
        );
    }
}

#[test]
fn semantic_transcript_uses_reasoning_summary_not_hidden_content() {
    let cold_history = vec![ResponseItem::Reasoning {
        id: Some("reasoning-id".to_string()),
        summary: vec![ReasoningItemReasoningSummary::SummaryText {
            text: "SEM_REASONING_SUMMARY".to_string(),
        }],
        content: None,
        encrypted_content: Some("SEM_HIDDEN_REASONING".to_string()),
        internal_chat_message_metadata_passthrough: Some(Default::default()),
    }];

    let transcript = render_semantic_transcript(&cold_history, 10_000).expect("render transcript");

    let serialized = serde_json::to_string(&transcript.items).expect("serialize transcript");
    assert!(serialized.contains("SEM_REASONING_SUMMARY"));
    assert!(!serialized.contains("SEM_HIDDEN_REASONING"));
    assert!(!serialized.contains("reasoning-id"));
}

#[test]
fn semantic_transcript_flushes_orphan_tool_calls_in_source_order() {
    let cold_history = vec![
        ResponseItem::FunctionCall {
            id: None,
            name: "first_tool".to_string(),
            namespace: None,
            arguments: "{}".to_string(),
            call_id: "call_first".to_string(),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::FunctionCall {
            id: None,
            name: "second_tool".to_string(),
            namespace: None,
            arguments: "{}".to_string(),
            call_id: "call_second".to_string(),
            internal_chat_message_metadata_passthrough: None,
        },
    ];

    let transcript = render_semantic_transcript(&cold_history, 10_000).expect("render transcript");

    let serialized = serde_json::to_string(&transcript.items).expect("serialize transcript");
    let first = serialized.find("first_tool").expect("first tool present");
    let second = serialized.find("second_tool").expect("second tool present");
    assert!(
        first < second,
        "orphan calls should retain source order: {serialized}"
    );
    assert!(!serialized.contains("call_first"));
    assert!(!serialized.contains("call_second"));
}
