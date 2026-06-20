use super::*;
use codex_protocol::models::ContentItem;
use codex_protocol::models::MessagePhase;
use pretty_assertions::assert_eq;

fn message(role: &str, text: &str, phase: Option<MessagePhase>) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: role.to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase,
        metadata: None,
    }
}

#[test]
fn build_v2_compacted_history_filters_to_installed_retention_shape() {
    let input = vec![
        message("developer", "dev", /*phase*/ None),
        message("system", "sys", /*phase*/ None),
        message("user", "user", /*phase*/ None),
        message("assistant", "commentary", Some(MessagePhase::Commentary)),
        message("assistant", "final", Some(MessagePhase::FinalAnswer)),
        ResponseItem::FunctionCall {
            id: None,
            name: "shell_command".to_string(),
            namespace: None,
            arguments: "{}".to_string(),
            call_id: "call_1".to_string(),
            metadata: None,
        },
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: "call_1".to_string(),
            output: codex_protocol::models::FunctionCallOutputPayload::from_text(
                "output".to_string(),
            ),
            metadata: None,
        },
        ResponseItem::Compaction {
            id: None,
            encrypted_content: "old".to_string(),
            metadata: None,
        },
    ];
    let output = ResponseItem::Compaction {
        id: None,
        encrypted_content: "new".to_string(),
        metadata: None,
    };

    let (history, _) = build_v2_compacted_history(&input, output.clone());

    assert_eq!(
        history,
        vec![message("user", "user", /*phase*/ None), output]
    );
}

#[test]
fn build_v2_compacted_history_discards_messages_before_truncating() {
    let old = message("user", "old", /*phase*/ None);
    let new = message("user", "new", /*phase*/ None);
    let huge_developer_message =
        "d".repeat((REMOTE_COMPACTION_V2_RETAINED_MESSAGE_TOKEN_BUDGET + 1) * 4);
    let huge_contextual_message = format!(
        "<environment_context>\n{}\n</environment_context>",
        "c".repeat((REMOTE_COMPACTION_V2_RETAINED_MESSAGE_TOKEN_BUDGET + 1) * 4)
    );
    let input = vec![
        old.clone(),
        message("developer", &huge_developer_message, /*phase*/ None),
        message("user", &huge_contextual_message, /*phase*/ None),
        new.clone(),
    ];
    let output = ResponseItem::Compaction {
        id: None,
        encrypted_content: "new".to_string(),
        metadata: None,
    };

    let (history, _) = build_v2_compacted_history(&input, output.clone());

    assert_eq!(history, vec![old, new, output]);
}

#[test]
fn build_v2_compacted_history_counts_retained_input_images() {
    let input = vec![ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![
            ContentItem::InputText {
                text: "user".to_string(),
            },
            ContentItem::InputImage {
                image_url: "data:image/png;base64,abc".to_string(),
                detail: None,
            },
            ContentItem::InputImage {
                image_url: "data:image/png;base64,def".to_string(),
                detail: None,
            },
        ],
        phase: None,
        metadata: None,
    }];
    let output = ResponseItem::Compaction {
        id: None,
        encrypted_content: "new".to_string(),
        metadata: None,
    };

    let (_, retained_image_count) = build_v2_compacted_history(&input, output);

    assert_eq!(retained_image_count, 2);
}

#[test]
fn retained_history_truncation_keeps_newest_messages_first() {
    let middle = message("user", "middle1234", /*phase*/ None);
    let new = message("user", "new", /*phase*/ None);
    let retained = vec![
        message("user", "old-old", /*phase*/ None),
        middle,
        new.clone(),
    ];

    let truncated =
        truncate_retained_messages_for_remote_compaction(retained, /*max_tokens*/ 3);

    assert_eq!(
        truncated,
        vec![
            message("user", "midd…1 tokens truncated…1234", /*phase*/ None),
            new,
        ]
    );
}

#[test]
fn retained_history_truncation_preserves_images_and_truncates_later_text_parts() {
    let item = ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![
            ContentItem::InputText {
                text: "abcdef".to_string(),
            },
            ContentItem::InputImage {
                image_url: "data:image/png;base64,abc".to_string(),
                detail: None,
            },
            ContentItem::OutputText {
                text: "uvwxyz".to_string(),
            },
        ],
        phase: None,
        metadata: None,
    };

    let truncated =
        truncate_retained_messages_for_remote_compaction(vec![item], /*max_tokens*/ 3);

    assert_eq!(
        truncated,
        vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![
                ContentItem::InputText {
                    text: "abcdef".to_string(),
                },
                ContentItem::InputImage {
                    image_url: "data:image/png;base64,abc".to_string(),
                    detail: None,
                },
                ContentItem::OutputText {
                    text: "uv…1 tokens truncated…yz".to_string(),
                },
            ],
            phase: None,
            metadata: None,
        }]
    );
}

#[test]
fn retained_history_truncation_charges_image_only_messages() {
    let image_only_message = ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputImage {
            image_url: "data:image/png;base64,abc".to_string(),
            detail: None,
        }],
        phase: None,
        metadata: None,
    };
    let newest = message("user", "new", /*phase*/ None);
    let retained = vec![
        message("user", "old", /*phase*/ None),
        image_only_message.clone(),
        newest.clone(),
    ];

    let truncated =
        truncate_retained_messages_for_remote_compaction(retained, /*max_tokens*/ 2);

    assert_eq!(truncated, vec![image_only_message, newest]);
}

#[test]
fn retained_history_truncation_drops_image_only_messages_after_budget_is_spent() {
    let image_only_message = ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputImage {
            image_url: "data:image/png;base64,abc".to_string(),
            detail: None,
        }],
        phase: None,
        metadata: None,
    };
    let newest = message("user", "new", /*phase*/ None);
    let retained = vec![image_only_message, newest.clone()];

    let truncated =
        truncate_retained_messages_for_remote_compaction(retained, /*max_tokens*/ 1);

    assert_eq!(truncated, vec![newest]);
}
