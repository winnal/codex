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
        internal_chat_message_metadata_passthrough: None,
    }
}

fn message_text(item: &ResponseItem) -> String {
    let ResponseItem::Message { content, .. } = item else {
        panic!("expected message item: {item:?}");
    };
    content
        .iter()
        .filter_map(|item| match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                Some(text.as_str())
            }
            ContentItem::InputImage { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
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
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: "call_1".to_string(),
            output: codex_protocol::models::FunctionCallOutputPayload::from_text(
                "output".to_string(),
            ),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Compaction {
            id: None,
            encrypted_content: "old".to_string(),
            internal_chat_message_metadata_passthrough: None,
        },
    ];
    let output = ResponseItem::Compaction {
        id: None,
        encrypted_content: "new".to_string(),
        internal_chat_message_metadata_passthrough: None,
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
        internal_chat_message_metadata_passthrough: None,
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
        internal_chat_message_metadata_passthrough: None,
    }];
    let output = ResponseItem::Compaction {
        id: None,
        encrypted_content: "new".to_string(),
        internal_chat_message_metadata_passthrough: None,
    };

    let (_, retained_image_count) = build_v2_compacted_history(&input, output);

    assert_eq!(retained_image_count, 2);
}

#[test]
fn retained_history_standard_v2_keeps_text_budget_boundary() {
    let retained = truncate_retained_messages_for_remote_compaction(
        vec![
            message("user", "old should be dropped", /*phase*/ None),
            message("user", &"word ".repeat(30), /*phase*/ None),
            message("user", "new", /*phase*/ None),
        ],
        /*max_tokens*/ 3,
        /*max_item_tokens*/ usize::MAX,
    );

    assert_eq!(retained.len(), 2);
    assert!(message_text(&retained[0]).contains("tokens truncated"));
    assert_eq!(retained[1], message("user", "new", /*phase*/ None));
    assert!(
        !retained
            .iter()
            .any(|item| message_text(item).contains("old should be dropped")),
        "standard v2 boundary truncation should exhaust the retained budget"
    );
    assert!(
        estimate_response_items_token_count(&retained) > 3,
        "standard v2 should preserve text-budget semantics rather than wrapper-inclusive item cap"
    );
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

    let truncated = truncate_retained_messages_for_remote_compaction(
        retained,
        /*max_tokens*/ 3,
        /*max_item_tokens*/ usize::MAX,
    );

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
        internal_chat_message_metadata_passthrough: None,
    };

    let truncated = truncate_retained_messages_for_remote_compaction(
        vec![item],
        /*max_tokens*/ 3,
        /*max_item_tokens*/ usize::MAX,
    );

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
            internal_chat_message_metadata_passthrough: None,
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
        internal_chat_message_metadata_passthrough: None,
    };
    let newest = message("user", "new", /*phase*/ None);
    let retained = vec![
        message("user", "old", /*phase*/ None),
        image_only_message.clone(),
        newest.clone(),
    ];

    let truncated = truncate_retained_messages_for_remote_compaction(
        retained,
        /*max_tokens*/ 2,
        /*max_item_tokens*/ usize::MAX,
    );

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
        internal_chat_message_metadata_passthrough: None,
    };
    let newest = message("user", "new", /*phase*/ None);
    let retained = vec![image_only_message, newest.clone()];

    let truncated = truncate_retained_messages_for_remote_compaction(
        retained,
        /*max_tokens*/ 1,
        /*max_item_tokens*/ usize::MAX,
    );

    assert_eq!(truncated, vec![newest]);
}

#[test]
fn retained_history_semantic_item_cap_truncates_individual_messages() {
    let input = vec![message(
        "user",
        &"word ".repeat(10_000),
        /*phase*/ None,
    )];

    let (retained, _) = retained_messages_for_remote_compaction_v2_with_item_cap(
        &input, /*max_tokens*/ 20_000, /*max_item_tokens*/ 512,
    );

    assert_eq!(retained.len(), 1);
    let serialized = serde_json::to_string(&retained).expect("serialize retained messages");
    assert!(serialized.contains("tokens truncated"));
    assert!(
        estimate_response_items_token_count(&retained) <= 512,
        "retained semantic item should fit cap: {retained:?}"
    );
}
