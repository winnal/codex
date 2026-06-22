use super::*;
use crate::compact_route::compact_route_for_config;
use crate::config::Config;
use crate::config::ConfigBuilder;
use codex_app_server_protocol::ConfigLayerSource;
use codex_config::ConfigLayerEntry;
use codex_config::ConfigLayerStack;
use codex_config::ConfigLayerStackOrdering;
use codex_model_provider_info::ModelProviderInfo;
use codex_model_provider_info::WireApi;
use codex_protocol::config_types::CompactExactTailStrategy;
use codex_protocol::error::CodexErr;
use codex_protocol::models::DEFAULT_IMAGE_DETAIL;
use codex_protocol::models::InternalChatMessageMetadataPassthrough;
use pretty_assertions::assert_eq;

async fn process_compacted_history_with_test_session(
    compacted_history: Vec<ResponseItem>,
    previous_turn_settings: Option<&PreviousTurnSettings>,
) -> (Vec<ResponseItem>, Vec<ResponseItem>) {
    let (session, turn_context) = crate::session::tests::make_session_and_context().await;
    session
        .set_previous_turn_settings(previous_turn_settings.cloned())
        .await;
    let initial_context = session.build_initial_context(&turn_context).await;
    let refreshed = crate::compact_remote::process_compacted_history(
        &session,
        &turn_context,
        compacted_history,
        InitialContextInjection::BeforeLastUserMessage,
    )
    .await;
    (refreshed, initial_context)
}

fn user_message(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn compacted_user_message(text: &str) -> CompactedUserMessage {
    CompactedUserMessage {
        message: text.to_string(),
        internal_chat_message_metadata_passthrough: None,
    }
}

#[test]
fn content_items_to_text_joins_non_empty_segments() {
    let items = vec![
        ContentItem::InputText {
            text: "hello".to_string(),
        },
        ContentItem::OutputText {
            text: String::new(),
        },
        ContentItem::OutputText {
            text: "world".to_string(),
        },
    ];

    let joined = content_items_to_text(&items);

    assert_eq!(Some("hello\nworld".to_string()), joined);
}

#[test]
fn content_items_to_text_ignores_image_only_content() {
    let items = vec![ContentItem::InputImage {
        image_url: "file://image.png".to_string(),
        detail: Some(DEFAULT_IMAGE_DETAIL),
    }];

    let joined = content_items_to_text(&items);

    assert_eq!(None, joined);
}

#[test]
fn collect_user_messages_extracts_user_text_only() {
    let items = vec![
        ResponseItem::Message {
            id: Some("assistant".to_string()),
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: "ignored".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: Some("user".to_string()),
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "first".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Other,
    ];

    let collected = collect_user_messages(&items);

    assert_eq!(vec![compacted_user_message("first")], collected);
}

#[test]
fn collect_user_messages_filters_session_prefix_entries() {
    let items = vec![
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: r#"# AGENTS.md instructions for project

<INSTRUCTIONS>
do things
</INSTRUCTIONS>"#
                    .to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "<ENVIRONMENT_CONTEXT>cwd=/tmp</ENVIRONMENT_CONTEXT>".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "real user message".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
    ];

    let collected = collect_user_messages(&items);

    assert_eq!(vec![compacted_user_message("real user message")], collected);
}

#[test]
fn collect_user_messages_filters_legacy_warnings() {
    let items = vec![
        user_message(
            "Warning: The maximum number of unified exec processes you can keep open is 60 and you currently have 61 processes open. Reuse older processes or close them to prevent automatic pruning of old processes",
        ),
        user_message(
            "Warning: apply_patch was requested via exec_command. Use the apply_patch tool instead of exec_command.",
        ),
        user_message(
            "Warning: Your account was flagged for potentially high-risk cyber activity and this request was routed to gpt-5.2 as a fallback. To regain access to gpt-5.3-codex, apply for trusted access: https://chatgpt.com/cyber or learn more: https://developers.openai.com/codex/concepts/cyber-safety",
        ),
        user_message("real user message"),
    ];

    let collected = collect_user_messages(&items);

    assert_eq!(vec![compacted_user_message("real user message")], collected);
}

#[test]
fn build_token_limited_compacted_history_truncates_overlong_user_messages() {
    // Use a small truncation limit so the test remains fast while still validating
    // that oversized user content is truncated.
    let max_tokens = 16;
    let big = "word ".repeat(200);
    let user_message = compacted_user_message(&big);
    let history = super::build_compacted_history_with_limit(
        Vec::new(),
        std::slice::from_ref(&user_message),
        "SUMMARY",
        max_tokens,
    );
    assert_eq!(history.len(), 2);

    let truncated_message = &history[0];
    let summary_message = &history[1];

    let truncated_text = match truncated_message {
        ResponseItem::Message { role, content, .. } if role == "user" => {
            content_items_to_text(content).unwrap_or_default()
        }
        other => panic!("unexpected item in history: {other:?}"),
    };

    assert!(
        truncated_text.contains("tokens truncated"),
        "expected truncation marker in truncated user message"
    );
    assert!(
        !truncated_text.contains(&big),
        "truncated user message should not include the full oversized user text"
    );

    let summary_text = match summary_message {
        ResponseItem::Message { role, content, .. } if role == "user" => {
            content_items_to_text(content).unwrap_or_default()
        }
        other => panic!("unexpected item in history: {other:?}"),
    };
    assert_eq!(summary_text, "SUMMARY");
}

#[test]
fn build_token_limited_compacted_history_appends_summary_message() {
    let initial_context: Vec<ResponseItem> = Vec::new();
    let user_messages = vec![compacted_user_message("first user message")];
    let summary_text = "summary text";

    let history = build_compacted_history(initial_context, &user_messages, summary_text);
    assert!(
        !history.is_empty(),
        "expected compacted history to include summary"
    );

    let last = history.last().expect("history should have a summary entry");
    let summary = match last {
        ResponseItem::Message { role, content, .. } if role == "user" => {
            content_items_to_text(content).unwrap_or_default()
        }
        other => panic!("expected summary message, found {other:?}"),
    };
    assert_eq!(summary, summary_text);
}

#[test]
fn build_compacted_history_preserves_user_message_passthrough_metadata() {
    let history = build_compacted_history(
        Vec::new(),
        &[CompactedUserMessage {
            message: "first user message".to_string(),
            internal_chat_message_metadata_passthrough: Some(
                InternalChatMessageMetadataPassthrough {
                    turn_id: Some("turn-1".to_string()),
                },
            ),
        }],
        "summary text",
    );

    assert_eq!(history[0].turn_id(), Some("turn-1"));
    assert_eq!(history[1].turn_id(), None);
}

#[test]
fn should_use_remote_compact_task_for_azure_provider() {
    let provider = ModelProviderInfo {
        name: "Azure".into(),
        base_url: Some("https://example.com/openai".into()),
        env_key: Some("AZURE_OPENAI_API_KEY".into()),
        env_key_instructions: None,
        experimental_bearer_token: None,
        auth: None,
        aws: None,
        wire_api: WireApi::Responses,
        query_params: None,
        http_headers: None,
        env_http_headers: None,
        request_max_retries: None,
        stream_max_retries: None,
        stream_idle_timeout_ms: None,
        websocket_connect_timeout_ms: None,
        requires_openai_auth: false,
        supports_websockets: false,
    };

    assert!(should_use_remote_compact_task(&provider));
}

async fn load_config(config_toml: &str) -> Config {
    let codex_home = tempfile::tempdir().expect("create codex home");
    if !config_toml.is_empty() {
        std::fs::write(
            codex_home.path().join(codex_config::CONFIG_TOML_FILE),
            config_toml,
        )
        .expect("write config");
    }
    ConfigBuilder::without_managed_config_for_tests()
        .codex_home(codex_home.path().to_path_buf())
        .build()
        .await
        .expect("load config")
}

fn add_remote_compaction_v2_layer(config: &mut Config, name: ConfigLayerSource, enabled: bool) {
    let mut layers = config
        .config_layer_stack
        .get_layers(
            ConfigLayerStackOrdering::LowestPrecedenceFirst,
            /*include_disabled*/ true,
        )
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    let layer = ConfigLayerEntry::new(
        name,
        toml::toml! {
            [features]
            remote_compaction_v2 = enabled
        }
        .into(),
    );
    match layer.name {
        ConfigLayerSource::Project { .. } => {
            let index = layers
                .iter()
                .rposition(|existing| matches!(existing.name, ConfigLayerSource::Project { .. }))
                .map(|index| index + 1)
                .unwrap_or_else(|| {
                    layers
                        .iter()
                        .position(|existing| existing.name.precedence() > layer.name.precedence())
                        .unwrap_or(layers.len())
                });
            layers.insert(index, layer);
        }
        _ => {
            let index = layers
                .iter()
                .position(|existing| existing.name.precedence() > layer.name.precedence())
                .unwrap_or(layers.len());
            layers.insert(index, layer);
        }
    }
    config.config_layer_stack = ConfigLayerStack::new(
        layers,
        config.config_layer_stack.requirements().clone(),
        config.config_layer_stack.requirements_toml().clone(),
    )
    .expect("build config layer stack");
    if enabled {
        let _ = config
            .features
            .enable(codex_features::Feature::RemoteCompactionV2);
    } else {
        let _ = config
            .features
            .disable(codex_features::Feature::RemoteCompactionV2);
    }
}

fn add_session_remote_compaction_v2_flag(config: &mut Config, enabled: bool) {
    add_remote_compaction_v2_layer(config, ConfigLayerSource::SessionFlags, enabled);
}

fn add_project_remote_compaction_v2_flag(config: &mut Config, enabled: bool) {
    add_remote_compaction_v2_layer(
        config,
        ConfigLayerSource::Project {
            dot_codex_folder: config.cwd.join(".codex-test-exact-tail"),
        },
        enabled,
    );
}

#[tokio::test]
async fn remote_compaction_v2_route_uses_default_without_exact_tail() {
    let config = load_config("").await;

    assert!(
        config
            .features
            .enabled(codex_features::Feature::RemoteCompactionV2)
    );
    assert!(should_use_remote_compact_task_v2_for_config(&config));
}

#[tokio::test]
async fn exact_tail_remote_compaction_v2_requires_explicit_enablement() {
    let config = load_config("compact_preserve_recent_tokens = 60000\n").await;

    assert!(
        config
            .features
            .enabled(codex_features::Feature::RemoteCompactionV2)
    );
    assert!(!should_use_remote_compact_task_v2_for_config(&config));
}

#[tokio::test]
async fn exact_tail_remote_compaction_v2_honors_explicit_enablement() {
    let config = load_config(
        r#"compact_preserve_recent_tokens = 60000

[features]
remote_compaction_v2 = true
"#,
    )
    .await;

    assert!(should_use_remote_compact_task_v2_for_config(&config));
}

#[tokio::test]
async fn exact_tail_remote_compaction_v2_honors_explicit_disablement() {
    let config = load_config(
        r#"compact_preserve_recent_tokens = 60000

[features]
remote_compaction_v2 = false
"#,
    )
    .await;

    assert!(
        !config
            .features
            .enabled(codex_features::Feature::RemoteCompactionV2)
    );
    assert!(!should_use_remote_compact_task_v2_for_config(&config));
}

#[tokio::test]
async fn exact_tail_remote_compaction_v2_honors_project_enablement() {
    let mut config = load_config("compact_preserve_recent_tokens = 60000\n").await;
    add_project_remote_compaction_v2_flag(&mut config, true);

    assert!(should_use_remote_compact_task_v2_for_config(&config));
}

#[tokio::test]
async fn exact_tail_remote_compaction_v2_honors_project_disablement_over_user_enablement() {
    let mut config = load_config(
        r#"compact_preserve_recent_tokens = 60000

[features]
remote_compaction_v2 = true
"#,
    )
    .await;
    add_project_remote_compaction_v2_flag(&mut config, false);

    assert!(!should_use_remote_compact_task_v2_for_config(&config));
}

#[tokio::test]
async fn exact_tail_remote_compaction_v2_ignores_session_flag_enablement() {
    let mut config = load_config(
        r#"compact_preserve_recent_tokens = 60000

[features]
remote_compaction_v2 = false
"#,
    )
    .await;
    add_session_remote_compaction_v2_flag(&mut config, true);

    assert!(
        config
            .features
            .enabled(codex_features::Feature::RemoteCompactionV2)
    );
    assert!(!should_use_remote_compact_task_v2_for_config(&config));
}

#[tokio::test]
async fn semantic_transcript_strategy_uses_v2_summary_route_regardless_of_default_v2_setting() {
    for config_toml in [
        r#"compact_preserve_recent_tokens = 60000
compact_exact_tail_strategy = "semantic_transcript"
"#,
        r#"compact_preserve_recent_tokens = 60000
compact_exact_tail_strategy = "semantic_transcript"

[features]
remote_compaction_v2 = false
"#,
        r#"compact_preserve_recent_tokens = 60000
compact_exact_tail_strategy = "semantic_transcript"

[features]
remote_compaction_v2 = true
"#,
    ] {
        let config = load_config(config_toml).await;
        assert_eq!(
            config.compact_exact_tail_strategy,
            CompactExactTailStrategy::SemanticTranscript
        );
        assert_eq!(
            compact_route_for_config(&config, &config.model_provider).expect("route"),
            CompactRoute::SemanticTranscript
        );
    }
}

#[tokio::test]
async fn exact_tail_auto_route_preserves_remote_v2_explicit_source_semantics() {
    let config = load_config("compact_preserve_recent_tokens = 60000\n").await;
    assert_eq!(
        compact_route_for_config(&config, &config.model_provider).expect("route"),
        CompactRoute::RemoteLegacy
    );

    let config = load_config(
        r#"compact_preserve_recent_tokens = 60000

[features]
remote_compaction_v2 = true
"#,
    )
    .await;
    assert_eq!(
        compact_route_for_config(&config, &config.model_provider).expect("route"),
        CompactRoute::RemoteV2
    );

    let config = load_config(
        r#"compact_preserve_recent_tokens = 60000

[features]
remote_compaction_v2 = false
"#,
    )
    .await;
    assert_eq!(
        compact_route_for_config(&config, &config.model_provider).expect("route"),
        CompactRoute::RemoteLegacy
    );
}

#[tokio::test]
async fn exact_tail_remote_legacy_strategy_routes_to_legacy_even_when_v2_enabled() {
    let config = load_config(
        r#"compact_preserve_recent_tokens = 60000
compact_exact_tail_strategy = "remote_legacy"

[features]
remote_compaction_v2 = true
"#,
    )
    .await;

    assert_eq!(
        compact_route_for_config(&config, &config.model_provider).expect("route"),
        CompactRoute::RemoteLegacy
    );
}

#[tokio::test]
async fn exact_tail_explicit_remote_strategies_fail_when_remote_unavailable() {
    for (strategy, route_name) in [
        ("remote_legacy", "remote"),
        ("remote_v2", "remote_v2"),
        ("semantic_transcript", "semantic_transcript"),
    ] {
        let mut config = load_config(&format!(
            r#"compact_preserve_recent_tokens = 60000
compact_exact_tail_strategy = "{strategy}"
"#
        ))
        .await;
        config.model_provider.name = "Example".to_string();
        config.model_provider.base_url = Some("https://example.com/v1".to_string());

        let err = compact_route_for_config(&config, &config.model_provider)
            .expect_err("explicit exact-tail remote strategy should fail without remote provider");
        let CodexErr::Stream(message, _) = err else {
            panic!("expected stream error for {strategy}");
        };
        assert!(
            message.contains(route_name),
            "error should name unavailable route {route_name}: {message}"
        );
    }
}

#[tokio::test]
async fn process_compacted_history_replaces_developer_messages() {
    let compacted_history = vec![
        ResponseItem::Message {
            id: None,
            role: "developer".to_string(),
            content: vec![ContentItem::InputText {
                text: "stale permissions".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "summary".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "developer".to_string(),
            content: vec![ContentItem::InputText {
                text: "stale personality".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
    ];
    let (refreshed, mut expected) = process_compacted_history_with_test_session(
        compacted_history,
        /*previous_turn_settings*/ None,
    )
    .await;
    expected.push(ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: "summary".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    });
    assert_eq!(refreshed, expected);
}

#[tokio::test]
async fn process_compacted_history_reinjects_full_initial_context() {
    let compacted_history = vec![ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: "summary".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }];
    let (refreshed, mut expected) = process_compacted_history_with_test_session(
        compacted_history,
        /*previous_turn_settings*/ None,
    )
    .await;
    expected.push(ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: "summary".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    });
    assert_eq!(refreshed, expected);
}

#[tokio::test]
async fn process_compacted_history_drops_non_user_content_messages() {
    let compacted_history = vec![
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: r#"# AGENTS.md instructions for /repo

<INSTRUCTIONS>
keep me updated
</INSTRUCTIONS>"#
                    .to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: r#"<environment_context>
  <cwd>/repo</cwd>
  <shell>zsh</shell>
</environment_context>"#
                    .to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: r#"<turn_aborted>
  <turn_id>turn-1</turn_id>
  <reason>interrupted</reason>
</turn_aborted>"#
                    .to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "summary".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "developer".to_string(),
            content: vec![ContentItem::InputText {
                text: "stale developer instructions".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
    ];
    let (refreshed, mut expected) = process_compacted_history_with_test_session(
        compacted_history,
        /*previous_turn_settings*/ None,
    )
    .await;
    expected.push(ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: "summary".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    });
    assert_eq!(refreshed, expected);
}

#[tokio::test]
async fn process_compacted_history_drops_legacy_warnings() {
    let latest_user = user_message("latest user");
    let compacted_history = vec![
        user_message(
            "Warning: The maximum number of unified exec processes you can keep open is 60 and you currently have 61 processes open. Reuse older processes or close them to prevent automatic pruning of old processes",
        ),
        user_message(
            "Warning: apply_patch was requested via exec_command. Use the apply_patch tool instead of exec_command.",
        ),
        user_message(
            "Warning: Your account was flagged for potentially high-risk cyber activity and this request was routed to gpt-5.2 as a fallback. To regain access to gpt-5.3-codex, apply for trusted access: https://chatgpt.com/cyber or learn more: https://developers.openai.com/codex/concepts/cyber-safety",
        ),
        latest_user.clone(),
    ];
    let (refreshed, initial_context) = process_compacted_history_with_test_session(
        compacted_history,
        /*previous_turn_settings*/ None,
    )
    .await;
    let mut expected = initial_context;
    expected.push(latest_user);
    assert_eq!(refreshed, expected);
}

#[tokio::test]
async fn process_compacted_history_inserts_context_before_last_real_user_message_only() {
    let compacted_history = vec![
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "older user".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: format!("{SUMMARY_PREFIX}\nsummary text"),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "latest user".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
    ];

    let (refreshed, initial_context) = process_compacted_history_with_test_session(
        compacted_history,
        /*previous_turn_settings*/ None,
    )
    .await;
    let mut expected = vec![
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "older user".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: format!("{SUMMARY_PREFIX}\nsummary text"),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
    ];
    expected.extend(initial_context);
    expected.push(ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: "latest user".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    });
    assert_eq!(refreshed, expected);
}

#[tokio::test]
async fn process_compacted_history_reinjects_model_switch_message() {
    let compacted_history = vec![ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: "summary".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }];
    let previous_turn_settings = PreviousTurnSettings {
        model: "previous-regular-model".to_string(),
        comp_hash: None,
        realtime_active: None,
    };

    let (refreshed, initial_context) = process_compacted_history_with_test_session(
        compacted_history,
        Some(&previous_turn_settings),
    )
    .await;

    let ResponseItem::Message { role, content, .. } = &initial_context[0] else {
        panic!("expected developer message");
    };
    assert_eq!(role, "developer");
    let [ContentItem::InputText { text }, ..] = content.as_slice() else {
        panic!("expected developer text");
    };
    assert!(text.contains("<model_switch>"));

    let mut expected = initial_context;
    expected.push(ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: "summary".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    });
    assert_eq!(refreshed, expected);
}

#[test]
fn insert_initial_context_before_last_real_user_or_summary_keeps_summary_last() {
    let compacted_history = vec![
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "older user".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "latest user".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: format!("{SUMMARY_PREFIX}\nsummary text"),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
    ];
    let initial_context = vec![ResponseItem::Message {
        id: None,
        role: "developer".to_string(),
        content: vec![ContentItem::InputText {
            text: "fresh permissions".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }];

    let refreshed =
        insert_initial_context_before_last_real_user_or_summary(compacted_history, initial_context);
    let expected = vec![
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "older user".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "developer".to_string(),
            content: vec![ContentItem::InputText {
                text: "fresh permissions".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "latest user".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: format!("{SUMMARY_PREFIX}\nsummary text"),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
    ];
    assert_eq!(refreshed, expected);
}

#[test]
fn insert_initial_context_before_last_real_user_or_summary_keeps_compaction_last() {
    let compacted_history = vec![ResponseItem::Compaction {
        id: None,
        encrypted_content: "encrypted".to_string(),
        internal_chat_message_metadata_passthrough: None,
    }];
    let initial_context = vec![ResponseItem::Message {
        id: None,
        role: "developer".to_string(),
        content: vec![ContentItem::InputText {
            text: "fresh permissions".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }];

    let refreshed =
        insert_initial_context_before_last_real_user_or_summary(compacted_history, initial_context);
    let expected = vec![
        ResponseItem::Message {
            id: None,
            role: "developer".to_string(),
            content: vec![ContentItem::InputText {
                text: "fresh permissions".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Compaction {
            id: None,
            encrypted_content: "encrypted".to_string(),
            internal_chat_message_metadata_passthrough: None,
        },
    ];
    assert_eq!(refreshed, expected);
}
