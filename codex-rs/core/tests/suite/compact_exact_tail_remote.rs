use super::compact_exact_tail_support::*;
use codex_protocol::config_types::CompactExactTailStrategy;
use pretty_assertions::assert_eq;

const TURN_STATE_HEADER: &str = "x-codex-turn-state";
const BETA_FEATURES_HEADER: &str = "x-codex-beta-features";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_remote_legacy_manual_compact_excludes_newest_atomic_hot_suffix_from_compaction_request()
-> Result<()> {
    let server = start_mock_server().await;
    let cold_user = format!("REMOTE_COLD_USER {}", "remote cold context ".repeat(100));
    let cold_assistant = format!(
        "REMOTE_COLD_ASSISTANT {}",
        "remote cold answer ".repeat(100)
    );
    let hot_user = format!("REMOTE_HOT_USER {}", "remote hot context ".repeat(100));
    let hot_assistant = format!("REMOTE_HOT_ASSISTANT {}", "remote hot answer ".repeat(100));
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("remote-cold-assistant-message", &cold_assistant),
                ev_completed("remote-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message("remote-hot-assistant-message", &hot_assistant),
                ev_completed("remote-hot-response"),
            ]),
            sse(vec![
                ev_assistant_message(
                    "remote-follow-up-assistant-message",
                    "REMOTE_FOLLOW_UP_DONE",
                ),
                ev_completed("remote-follow-up-response"),
            ]),
        ],
    )
    .await;
    let compacted_history = vec![codex_protocol::models::ResponseItem::Compaction {
        id: None,
        encrypted_content: "REMOTE_EXACT_TAIL_SUMMARY".to_string(),
        internal_chat_message_metadata_passthrough: None,
    }];
    let compact_mock =
        mount_compact_json_once(&server, serde_json::json!({ "output": compacted_history })).await;
    let mut builder = test_codex()
        .with_auth(CodexAuth::from_api_key("dummy"))
        .with_config(|config| {
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.compact_preserve_recent_tokens = Some(1);
            let _ = config.features.disable(Feature::RemoteCompactionV2);
        });
    let test = builder.build(&server).await?;

    submit_turn(&test, &cold_user).await?;
    submit_turn(&test, &hot_user).await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_compact_turn_complete(&test).await;
    submit_turn(&test, "REMOTE_FOLLOW_UP_USER").await?;

    let compact_request = compact_mock.single_request();
    let compact_body = compact_request.body_json().to_string();
    assert!(
        body_contains_text(&compact_body, "REMOTE_COLD_USER"),
        "cold user should be sent to remote compact; compact body: {compact_body}"
    );
    assert!(
        body_contains_text(&compact_body, "REMOTE_COLD_ASSISTANT"),
        "cold assistant output should be sent to remote compact; compact body: {compact_body}"
    );
    assert!(
        body_contains_text(&compact_body, "REMOTE_HOT_USER"),
        "older same-turn user text should remain cold when the target only selects the newest atomic group; compact body: {compact_body}"
    );
    assert!(
        !body_contains_text(&compact_body, "REMOTE_HOT_ASSISTANT"),
        "newest atomic hot assistant text must be excluded from remote compact; compact body: {compact_body}"
    );

    let requests = response_mock.requests();
    assert_eq!(
        requests.len(),
        3,
        "expected cold turn, hot turn, and follow-up"
    );
    let follow_up_input = requests[2].input();
    assert_ordered_input_texts(
        &follow_up_input,
        &[
            "REMOTE_EXACT_TAIL_SUMMARY",
            "REMOTE_HOT_ASSISTANT",
            "REMOTE_FOLLOW_UP_USER",
        ],
    );
    assert!(
        !input_contains_text(&follow_up_input, "REMOTE_HOT_USER"),
        "follow-up should not replay older same-turn user text outside the remote summary"
    );
    assert!(
        !input_contains_text(&follow_up_input, "REMOTE_COLD_ASSISTANT"),
        "follow-up should not replay cold assistant text outside the remote summary"
    );

    shutdown_codex(&test).await?;
    Ok(())
}

async fn submit_turn(test: &TestCodex, prompt: &str) -> Result<()> {
    test.submit_turn_with_completion_timeout(prompt, Duration::from_secs(90))
        .await
}

async fn wait_for_compact_turn_complete(test: &TestCodex) {
    wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        Duration::from_secs(90),
    )
    .await;
}

async fn wait_for_error_message(test: &TestCodex) -> String {
    let event = wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::Error(_)),
        Duration::from_secs(90),
    )
    .await;
    let EventMsg::Error(err) = event else {
        unreachable!("predicate should only match error events");
    };
    err.message
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_remote_legacy_auto_compact_excludes_newest_atomic_hot_suffix_from_compaction_request()
-> Result<()> {
    let server = start_mock_server().await;
    let cold_user = format!("REMOTE_AUTO_COLD_USER {}", "remote auto cold ".repeat(100));
    let cold_assistant = format!(
        "REMOTE_AUTO_COLD_ASSISTANT {}",
        "remote auto cold answer ".repeat(100)
    );
    let hot_user = format!("REMOTE_AUTO_HOT_USER {}", "remote auto hot ".repeat(100));
    let hot_assistant = format!(
        "REMOTE_AUTO_HOT_ASSISTANT {}",
        "remote auto hot answer ".repeat(100)
    );
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("remote-auto-cold-assistant-message", &cold_assistant),
                ev_completed_with_tokens("remote-auto-cold-response", /*total_tokens*/ 50),
            ]),
            sse(vec![
                ev_assistant_message("remote-auto-hot-assistant-message", &hot_assistant),
                ev_completed_with_tokens("remote-auto-hot-response", /*total_tokens*/ 500),
            ]),
            sse(vec![
                ev_assistant_message(
                    "remote-auto-follow-up-assistant-message",
                    "REMOTE_AUTO_FOLLOW_UP_DONE",
                ),
                ev_completed("remote-auto-follow-up-response"),
            ]),
        ],
    )
    .await;
    let compacted_history = vec![codex_protocol::models::ResponseItem::Compaction {
        id: None,
        encrypted_content: "REMOTE_AUTO_EXACT_TAIL_SUMMARY".to_string(),
        internal_chat_message_metadata_passthrough: None,
    }];
    let compact_mock =
        mount_compact_json_once(&server, serde_json::json!({ "output": compacted_history })).await;
    let mut builder = test_codex()
        .with_auth(CodexAuth::from_api_key("dummy"))
        .with_config(|config| {
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.model_auto_compact_token_limit = Some(100);
            config.model_auto_compact_token_limit_scope =
                AutoCompactTokenLimitScope::BodyAfterPrefix;
            config.compact_preserve_recent_tokens = Some(1);
            let _ = config.features.disable(Feature::RemoteCompactionV2);
        });
    let test = builder.build(&server).await?;

    submit_turn(&test, &cold_user).await?;
    submit_turn(&test, &hot_user).await?;
    submit_turn(&test, "REMOTE_AUTO_FOLLOW_UP_USER").await?;

    let compact_request = compact_mock.single_request();
    let compact_body = compact_request.body_json().to_string();
    assert!(
        body_contains_text(&compact_body, "REMOTE_AUTO_COLD_USER"),
        "cold user should be sent to remote auto compact; compact body: {compact_body}"
    );
    assert!(
        body_contains_text(&compact_body, "REMOTE_AUTO_COLD_ASSISTANT"),
        "cold assistant output should be sent to remote auto compact; compact body: {compact_body}"
    );
    assert!(
        body_contains_text(&compact_body, "REMOTE_AUTO_HOT_USER"),
        "older same-turn user text should remain cold when the target only selects the newest atomic group; compact body: {compact_body}"
    );
    assert!(
        !body_contains_text(&compact_body, "REMOTE_AUTO_HOT_ASSISTANT"),
        "newest atomic hot assistant text must be excluded from remote auto compact; compact body: {compact_body}"
    );

    let requests = response_mock.requests();
    assert_eq!(
        requests.len(),
        3,
        "expected cold turn, hot turn, and post-compaction follow-up"
    );
    let follow_up_input = requests[2].input();
    assert_ordered_input_texts(
        &follow_up_input,
        &[
            "REMOTE_AUTO_EXACT_TAIL_SUMMARY",
            "REMOTE_AUTO_HOT_ASSISTANT",
            "REMOTE_AUTO_FOLLOW_UP_USER",
        ],
    );
    assert!(
        !input_contains_text(&follow_up_input, "REMOTE_AUTO_HOT_USER"),
        "follow-up should not replay older same-turn user text outside the remote auto summary"
    );
    assert!(
        !input_contains_text(&follow_up_input, "REMOTE_AUTO_COLD_ASSISTANT"),
        "follow-up should not replay cold assistant text outside the remote auto summary"
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_remote_legacy_user_only_output_fails_without_installing_history() -> Result<()>
{
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("remote-user-only-cold-assistant", "REMOTE_USER_ONLY_COLD"),
                ev_completed("remote-user-only-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message("remote-user-only-hot-assistant", "REMOTE_USER_ONLY_HOT"),
                ev_completed("remote-user-only-hot-response"),
            ]),
        ],
    )
    .await;
    let compacted_history = vec![codex_protocol::models::ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![codex_protocol::models::ContentItem::InputText {
            text: "REMOTE_USER_ONLY_RETAINED_TEXT".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }];
    let compact_mock =
        mount_compact_json_once(&server, serde_json::json!({ "output": compacted_history })).await;
    let mut builder = test_codex()
        .with_auth(CodexAuth::from_api_key("dummy"))
        .with_config(|config| {
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.compact_preserve_recent_tokens = Some(1);
            let _ = config.features.disable(Feature::RemoteCompactionV2);
        });
    let test = builder.build(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    submit_turn(&test, "REMOTE_USER_ONLY_COLD_USER").await?;
    submit_turn(&test, "REMOTE_USER_ONLY_HOT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    let error_message = wait_for_error_message(&test).await;

    assert!(
        error_message.contains("ExactTailNoUsableColdSummary"),
        "expected exact-tail user-only remote output error, got {error_message}"
    );
    assert_eq!(
        response_mock.requests().len(),
        2,
        "remote user-only output should fail before any follow-up request"
    );
    assert_eq!(
        compact_mock.requests().len(),
        1,
        "remote compact endpoint should be called before user-only output failure"
    );
    assert!(
        replacement_history_from_rollout(&rollout_path).is_err(),
        "failed remote exact-tail user-only output must not install compacted history"
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_remote_legacy_backend_context_window_error_emits_diagnostic() -> Result<()> {
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("remote-legacy-backend-cold", "REMOTE_LEGACY_BACKEND_COLD"),
                ev_completed("remote-legacy-backend-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message("remote-legacy-backend-hot", "REMOTE_LEGACY_BACKEND_HOT"),
                ev_completed("remote-legacy-backend-hot-response"),
            ]),
        ],
    )
    .await;
    let compact_mock = mount_compact_response_once(
        &server,
        ResponseTemplate::new(400)
            .insert_header("content-type", "application/json")
            .set_body_json(json!({
                "error": {
                    "code": "context_length_exceeded",
                    "message": "Your input exceeds the context window of this model. Please adjust your input and try again."
                }
            })),
    )
    .await;
    let mut builder = test_codex()
        .with_auth(CodexAuth::from_api_key("dummy"))
        .with_config(|config| {
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.compact_preserve_recent_tokens = Some(1);
            let _ = config.features.disable(Feature::RemoteCompactionV2);
        });
    let test = builder.build(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    submit_turn(&test, "REMOTE_LEGACY_BACKEND_COLD_USER").await?;
    submit_turn(&test, "REMOTE_LEGACY_BACKEND_HOT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    let error_message = wait_for_error_message(&test).await;

    assert!(
        error_message.contains("ExactTailBackendContextExceeded"),
        "expected exact-tail remote legacy backend context-window failure, got {error_message}"
    );
    assert_eq!(
        response_mock.requests().len(),
        2,
        "remote legacy backend failure should happen after the two setup turns"
    );
    assert_eq!(
        compact_mock.requests().len(),
        1,
        "remote legacy backend failure should issue exactly one compact request"
    );
    assert!(
        replacement_history_from_rollout(&rollout_path).is_err(),
        "failed remote legacy backend context-window error must not install compacted history"
    );
    let diagnostics = exact_tail_diagnostics_from_rollout(&rollout_path)?;
    assert_eq!(diagnostics.len(), 1);
    let diagnostic = &diagnostics[0];
    assert_eq!(
        diagnostic.get("route").and_then(Value::as_str),
        Some("remote_legacy")
    );
    assert_eq!(
        diagnostic.get("fit_result").and_then(Value::as_str),
        Some("failure")
    );
    assert_eq!(
        diagnostic.get("failure_reason").and_then(Value::as_str),
        Some("ExactTailBackendContextExceeded")
    );
    assert_eq!(
        diagnostic
            .get("hot_suffix_exact_match")
            .and_then(Value::as_bool),
        None
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_remote_v2_enabled_manual_uses_cold_only_v2_request() -> Result<()> {
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message(
                    "remote-v2-route-cold-assistant",
                    "REMOTE_V2_ROUTE_COLD_ASSISTANT",
                ),
                ev_completed("remote-v2-route-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message(
                    "remote-v2-route-hot-assistant",
                    "REMOTE_V2_ROUTE_ASSISTANT_EXACT",
                ),
                ev_completed("remote-v2-route-hot-response"),
            ]),
            sse(vec![
                ev_compaction_item("REMOTE_V2_ROUTE_EXACT_TAIL_SUMMARY"),
                ev_completed("remote-v2-route-compact-response"),
            ]),
            sse(vec![
                ev_assistant_message(
                    "remote-v2-route-follow-up-assistant",
                    "REMOTE_V2_ROUTE_FOLLOW_UP_DONE",
                ),
                ev_completed("remote-v2-route-follow-up-response"),
            ]),
        ],
    )
    .await;
    let mut builder = test_codex()
        .with_auth(CodexAuth::from_api_key("dummy"))
        .with_config(|config| {
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.compact_preserve_recent_tokens = Some(1);
            explicitly_enable_remote_compaction_v2(config);
        });
    let test = builder.build(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    submit_turn(&test, "REMOTE_V2_ROUTE_COLD_USER").await?;
    submit_turn(&test, "REMOTE_V2_ROUTE_HOT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_compact_turn_complete(&test).await;
    submit_turn(&test, "REMOTE_V2_ROUTE_FOLLOW_UP_USER").await?;

    let requests = response_mock.requests();
    assert_eq!(
        requests.len(),
        4,
        "expected cold turn, hot turn, v2 compaction turn, and follow-up"
    );
    let compact_request = &requests[2];
    assert_eq!(compact_request.path(), "/v1/responses");
    let compact_body = compact_request.body_json();
    let compact_body_text = compact_body.to_string();
    assert_eq!(
        compact_body.get("previous_response_id"),
        None,
        "exact-tail v2 compaction must not continue a previous Responses chain"
    );
    assert!(
        compact_body
            .get("prompt_cache_key")
            .and_then(Value::as_str)
            .is_some_and(|key| key.starts_with("exact-tail-v2-compaction:")),
        "exact-tail v2 compaction must use an isolated prompt cache key; body: {compact_body}"
    );
    assert!(
        body_contains_text(&compact_body_text, "REMOTE_V2_ROUTE_COLD_USER"),
        "cold user should be sent to v2 compact; compact body: {compact_body}"
    );
    assert!(
        body_contains_text(&compact_body_text, "REMOTE_V2_ROUTE_COLD_ASSISTANT"),
        "cold assistant should be sent to v2 compact; compact body: {compact_body}"
    );
    assert!(
        body_contains_text(&compact_body_text, "REMOTE_V2_ROUTE_HOT_USER"),
        "older same-turn user should remain cold when the target only selects the newest atomic group; compact body: {compact_body}"
    );
    assert!(
        !body_contains_text(&compact_body_text, "REMOTE_V2_ROUTE_ASSISTANT_EXACT"),
        "hot assistant must stay out of the v2 compact request; compact body: {compact_body}"
    );
    assert_eq!(
        compact_request
            .input()
            .iter()
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("compaction_trigger"))
            .count(),
        1,
        "v2 compact input should append exactly one compaction_trigger"
    );
    assert_eq!(
        compact_request
            .input()
            .last()
            .and_then(|item| item.get("type"))
            .and_then(Value::as_str),
        Some("compaction_trigger"),
        "v2 compact input should end with compaction_trigger"
    );
    assert_ordered_input_texts(
        &requests[3].input(),
        &[
            "REMOTE_V2_ROUTE_COLD_USER",
            "REMOTE_V2_ROUTE_HOT_USER",
            "REMOTE_V2_ROUTE_EXACT_TAIL_SUMMARY",
            "REMOTE_V2_ROUTE_ASSISTANT_EXACT",
            "REMOTE_V2_ROUTE_FOLLOW_UP_USER",
        ],
    );
    assert!(
        !input_contains_text(&requests[3].input(), "REMOTE_V2_ROUTE_COLD_ASSISTANT"),
        "follow-up should not replay cold assistant text outside the v2 summary"
    );
    let diagnostics = exact_tail_diagnostics_from_rollout(&rollout_path)?;
    assert_eq!(diagnostics.len(), 1);
    let diagnostic = &diagnostics[0];
    assert_eq!(
        diagnostic.get("route").and_then(Value::as_str),
        Some("remote_v2")
    );
    assert_eq!(
        diagnostic.get("trigger").and_then(Value::as_str),
        Some("manual")
    );
    assert_eq!(
        diagnostic.get("fit_result").and_then(Value::as_str),
        Some("success")
    );
    assert_eq!(
        diagnostic
            .get("hot_suffix_exact_match")
            .and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        diagnostic
            .get("planned_hot_suffix_item_count")
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        diagnostic
            .get("installed_hot_suffix_item_count")
            .and_then(Value::as_u64),
        Some(1)
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_remote_v2_strategy_advertises_beta_when_default_v2_disabled() -> Result<()> {
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message(
                    "remote-v2-disabled-cold-assistant",
                    "REMOTE_V2_DISABLED_COLD_ASSISTANT",
                ),
                ev_completed("remote-v2-disabled-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message(
                    "remote-v2-disabled-hot-assistant",
                    "REMOTE_V2_DISABLED_HOT_ASSISTANT",
                ),
                ev_completed("remote-v2-disabled-hot-response"),
            ]),
            sse(vec![
                ev_compaction_item("REMOTE_V2_DISABLED_SUMMARY"),
                ev_completed("remote-v2-disabled-compact-response"),
            ]),
        ],
    )
    .await;
    let mut builder = test_codex()
        .with_auth(CodexAuth::from_api_key("dummy"))
        .with_config(|config| {
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.compact_preserve_recent_tokens = Some(1);
            config.compact_exact_tail_strategy = CompactExactTailStrategy::RemoteV2;
            let _ = config.features.disable(Feature::RemoteCompactionV2);
        });
    let test = builder.build(&server).await?;

    submit_turn(&test, "REMOTE_V2_DISABLED_COLD_USER").await?;
    submit_turn(&test, "REMOTE_V2_DISABLED_HOT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_compact_turn_complete(&test).await;

    let requests = response_mock.requests();
    assert_eq!(
        requests.len(),
        3,
        "expected cold turn, hot turn, and explicit remote-v2 compaction request"
    );
    let compact_request = &requests[2];
    assert_eq!(compact_request.path(), "/v1/responses");
    assert!(
        compact_request
            .header(BETA_FEATURES_HEADER)
            .as_deref()
            .is_some_and(header_contains_remote_v2),
        "explicit remote-v2 exact-tail compaction must advertise remote_compaction_v2 even when the default feature is disabled"
    );
    let compact_body = compact_request.body_json();
    assert_eq!(
        compact_body.get("previous_response_id"),
        None,
        "exact-tail v2 compaction must not continue a previous Responses chain"
    );
    assert!(
        compact_body
            .get("prompt_cache_key")
            .and_then(Value::as_str)
            .is_some_and(|key| key.starts_with("exact-tail-v2-compaction:")),
        "exact-tail v2 compaction must use an isolated prompt cache key; body: {compact_body}"
    );
    assert_eq!(
        compact_request
            .input()
            .iter()
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("compaction_trigger"))
            .count(),
        1,
        "explicit remote-v2 input should append exactly one compaction_trigger"
    );
    assert_eq!(
        compact_request
            .input()
            .last()
            .and_then(|item| item.get("type"))
            .and_then(Value::as_str),
        Some("compaction_trigger"),
        "explicit remote-v2 input should end with compaction_trigger"
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_remote_v2_manual_clears_cached_websocket_continuation() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_websocket_server_concurrent(vec![
        vec![
            ws_warm_response("remote-v2-manual-warm"),
            ws_assistant_response(
                "remote-v2-manual-cold",
                "remote-v2-manual-cold-response",
                "REMOTE_V2_MANUAL_COLD",
                None,
            ),
            ws_assistant_response(
                "remote-v2-manual-hot",
                "remote-v2-manual-hot-response",
                "REMOTE_V2_MANUAL_HOT",
                None,
            ),
            ws_assistant_response(
                "remote-v2-manual-stale",
                "remote-v2-manual-stale-response",
                "REMOTE_V2_MANUAL_STALE_REUSE",
                None,
            ),
        ],
        vec![ws_compaction_response(
            "REMOTE_V2_MANUAL_SUMMARY",
            "remote-v2-manual-compact-response",
        )],
        vec![ws_assistant_response(
            "remote-v2-manual-follow",
            "remote-v2-manual-follow-response",
            "REMOTE_V2_MANUAL_FOLLOW",
            None,
        )],
    ])
    .await;
    let mut builder = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_config(|config| {
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.compact_preserve_recent_tokens = Some(1);
            explicitly_enable_remote_compaction_v2(config);
        });
    let test = builder.build_with_websocket_server(&server).await?;

    submit_turn(&test, "REMOTE_V2_MANUAL_COLD_USER").await?;
    submit_turn(&test, "REMOTE_V2_MANUAL_HOT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_compact_turn_complete(&test).await;
    submit_turn(&test, "REMOTE_V2_MANUAL_FOLLOW_USER").await?;

    let connections = server.connections();
    assert_eq!(connections.len(), 3);
    assert_eq!(
        connections[0].len(),
        3,
        "manual follow-up must not reuse the pre-compaction websocket"
    );
    let post_compaction = &connections[2];
    assert_eq!(post_compaction.len(), 1);
    let follow_body = post_compaction[0].body_json();
    assert_ne!(
        follow_body
            .get("previous_response_id")
            .and_then(Value::as_str),
        Some("remote-v2-manual-hot-response"),
        "manual exact-tail v2 compaction must clear stale pre-compaction continuation"
    );
    assert_eq!(follow_body.get("previous_response_id"), None);

    shutdown_codex(&test).await?;
    server.shutdown().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_remote_v2_auto_clears_cached_websocket_continuation() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_websocket_server_concurrent(vec![
        vec![
            ws_warm_response("remote-v2-auto-warm"),
            ws_assistant_response(
                "remote-v2-auto-cold",
                "remote-v2-auto-cold-response",
                "REMOTE_V2_AUTO_COLD",
                Some(50),
            ),
            vec![
                json!({
                    "type": "response.metadata",
                    "headers": {(TURN_STATE_HEADER): "sampling-state"},
                }),
                ev_function_call("remote-v2-auto-call", DUMMY_FUNCTION_NAME, "{}"),
                ev_completed_with_tokens("remote-v2-auto-hot-response", /*total_tokens*/ 500),
            ],
            ws_assistant_response(
                "remote-v2-auto-stale",
                "remote-v2-auto-stale-response",
                "REMOTE_V2_AUTO_STALE_REUSE",
                None,
            ),
        ],
        vec![ws_compaction_response(
            "REMOTE_V2_AUTO_SUMMARY",
            "remote-v2-auto-compact-response",
        )],
        vec![ws_assistant_response(
            "remote-v2-auto-final",
            "remote-v2-auto-final-response",
            "REMOTE_V2_AUTO_FINAL",
            None,
        )],
    ])
    .await;
    let mut builder = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_config(|config| {
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.model_auto_compact_token_limit = Some(100);
            config.model_auto_compact_token_limit_scope =
                AutoCompactTokenLimitScope::BodyAfterPrefix;
            config.compact_preserve_recent_tokens = Some(1);
            explicitly_enable_remote_compaction_v2(config);
        });
    let test = builder.build_with_websocket_server(&server).await?;

    submit_turn(&test, "REMOTE_V2_AUTO_COLD_USER").await?;
    submit_turn(&test, "REMOTE_V2_AUTO_HOT_USER").await?;

    let connections = server.connections();
    assert_eq!(connections.len(), 3);
    assert_eq!(
        connections[0].len(),
        3,
        "auto follow-up must not reuse the pre-compaction websocket"
    );
    let compact_body = connections[1][0].body_json();
    assert_eq!(compact_body.get("previous_response_id"), None);
    assert_eq!(
        compact_body["client_metadata"][TURN_STATE_HEADER],
        json!("sampling-state"),
        "exact-tail v2 auto compaction must carry the active turn-state"
    );
    assert!(
        compact_body
            .get("prompt_cache_key")
            .and_then(Value::as_str)
            .is_some_and(|key| key.starts_with("exact-tail-v2-compaction:")),
        "exact-tail v2 auto compaction must use an isolated prompt cache key; body: {compact_body}"
    );
    assert!(
        compact_body
            .to_string()
            .contains("\"type\":\"compaction_trigger\""),
        "exact-tail v2 auto compaction should append a compaction trigger; body: {compact_body}"
    );

    let post_compaction = &connections[2];
    assert_eq!(post_compaction.len(), 1);
    let follow_body = post_compaction[0].body_json();
    assert_ne!(
        follow_body
            .get("previous_response_id")
            .and_then(Value::as_str),
        Some("remote-v2-auto-hot-response"),
        "exact-tail v2 auto compaction must clear stale pre-compaction continuation"
    );
    assert_eq!(follow_body.get("previous_response_id"), None);
    assert!(
        !follow_body
            .get("prompt_cache_key")
            .and_then(Value::as_str)
            .is_some_and(|key| key.starts_with("exact-tail-v2-compaction:")),
        "post-compaction sampling must not inherit the isolated compact cache key"
    );

    shutdown_codex(&test).await?;
    server.shutdown().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_remote_v2_backend_context_window_error_emits_diagnostic() -> Result<()> {
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("remote-v2-backend-cold", "REMOTE_V2_BACKEND_COLD"),
                ev_completed("remote-v2-backend-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message("remote-v2-backend-hot", "REMOTE_V2_BACKEND_HOT"),
                ev_completed("remote-v2-backend-hot-response"),
            ]),
            sse_failed(
                "remote-v2-backend-compact-failed",
                "context_length_exceeded",
                "Your input exceeds the context window of this model. Please adjust your input and try again.",
            ),
        ],
    )
    .await;
    let mut builder = test_codex()
        .with_auth(CodexAuth::from_api_key("dummy"))
        .with_config(|config| {
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.compact_preserve_recent_tokens = Some(1);
            explicitly_enable_remote_compaction_v2(config);
        });
    let test = builder.build(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    submit_turn(&test, "REMOTE_V2_BACKEND_COLD_USER").await?;
    submit_turn(&test, "REMOTE_V2_BACKEND_HOT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    let error_message = wait_for_error_message(&test).await;

    assert!(
        error_message.contains("ExactTailBackendContextExceeded"),
        "expected exact-tail remote v2 backend context-window failure, got {error_message}"
    );
    assert_eq!(
        response_mock.requests().len(),
        3,
        "remote v2 backend failure should issue one v2 compaction request"
    );
    assert!(
        replacement_history_from_rollout(&rollout_path).is_err(),
        "failed remote v2 backend context-window error must not install compacted history"
    );
    let diagnostics = exact_tail_diagnostics_from_rollout(&rollout_path)?;
    assert_eq!(diagnostics.len(), 1);
    let diagnostic = &diagnostics[0];
    assert_eq!(
        diagnostic.get("route").and_then(Value::as_str),
        Some("remote_v2")
    );
    assert_eq!(
        diagnostic.get("fit_result").and_then(Value::as_str),
        Some("failure")
    );
    assert_eq!(
        diagnostic.get("failure_reason").and_then(Value::as_str),
        Some("ExactTailBackendContextExceeded")
    );
    assert_eq!(
        diagnostic
            .get("hot_suffix_exact_match")
            .and_then(Value::as_bool),
        None
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_remote_v2_empty_compaction_fails_without_installing_history() -> Result<()> {
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message(
                    "remote-v2-empty-cold-assistant",
                    "REMOTE_V2_EMPTY_COLD_ASSISTANT",
                ),
                ev_completed("remote-v2-empty-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message(
                    "remote-v2-empty-hot-assistant",
                    "REMOTE_V2_EMPTY_HOT_ASSISTANT",
                ),
                ev_completed("remote-v2-empty-hot-response"),
            ]),
            sse(vec![
                ev_compaction_item(""),
                ev_completed("remote-v2-empty-compact-response"),
            ]),
        ],
    )
    .await;
    let mut builder = test_codex()
        .with_auth(CodexAuth::from_api_key("dummy"))
        .with_config(|config| {
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.compact_preserve_recent_tokens = Some(1);
            explicitly_enable_remote_compaction_v2(config);
        });
    let test = builder.build(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    submit_turn(&test, "REMOTE_V2_EMPTY_COLD_USER").await?;
    submit_turn(&test, "REMOTE_V2_EMPTY_HOT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    let error_message = wait_for_error_message(&test).await;

    assert!(
        error_message.contains("ExactTailNoUsableColdSummary"),
        "expected exact-tail empty v2 compaction error, got {error_message}"
    );
    let requests = response_mock.requests();
    assert_eq!(
        requests.len(),
        3,
        "empty v2 compaction should fail before any follow-up request"
    );
    let compact_body = requests[2].body_json().to_string();
    assert!(
        body_contains_text(&compact_body, "REMOTE_V2_EMPTY_COLD_USER"),
        "cold user should be sent to v2 compact before failure; compact body: {compact_body}"
    );
    assert!(
        body_contains_text(&compact_body, "REMOTE_V2_EMPTY_HOT_USER"),
        "older same-turn user should remain cold when the target only selects the newest atomic group; compact body: {compact_body}"
    );
    assert!(
        !body_contains_text(&compact_body, "REMOTE_V2_EMPTY_HOT_ASSISTANT"),
        "hot assistant must stay out of failed v2 compact request; compact body: {compact_body}"
    );
    assert!(
        replacement_history_from_rollout(&rollout_path).is_err(),
        "failed remote v2 exact-tail output must not install compacted history"
    );
    let diagnostics = exact_tail_diagnostics_from_rollout(&rollout_path)?;
    assert_eq!(diagnostics.len(), 1);
    let diagnostic = &diagnostics[0];
    assert_eq!(
        diagnostic.get("route").and_then(Value::as_str),
        Some("remote_v2")
    );
    assert_eq!(
        diagnostic.get("fit_result").and_then(Value::as_str),
        Some("failure")
    );
    assert_eq!(
        diagnostic.get("failure_reason").and_then(Value::as_str),
        Some("ExactTailNoUsableColdSummary")
    );
    assert_eq!(
        diagnostic
            .get("hot_suffix_exact_match")
            .and_then(Value::as_bool),
        None
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_remote_v2_empty_compaction_fails_even_with_retained_old_summary() -> Result<()>
{
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message(
                    "remote-v2-old-summary-assistant",
                    "REMOTE_V2_OLD_SUMMARY_ASSISTANT",
                ),
                ev_completed("remote-v2-old-summary-response"),
            ]),
            sse(vec![
                ev_assistant_message(
                    "remote-v2-new-cold-assistant",
                    "REMOTE_V2_NEW_COLD_ASSISTANT",
                ),
                ev_completed("remote-v2-new-cold-response"),
            ]),
            sse(vec![
                ev_assistant_message("remote-v2-retained-hot", "REMOTE_V2_RETAINED_HOT"),
                ev_completed("remote-v2-retained-hot-response"),
            ]),
            sse(vec![
                ev_compaction_item(""),
                ev_completed("remote-v2-old-summary-empty-compact-response"),
            ]),
        ],
    )
    .await;
    let mut builder = test_codex()
        .with_auth(CodexAuth::from_api_key("dummy"))
        .with_config(|config| {
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.compact_preserve_recent_tokens = Some(1);
            explicitly_enable_remote_compaction_v2(config);
        });
    let test = builder.build(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");
    let old_summary_user = summary_with_prefix("REMOTE_V2_OLD_RETAINED_SUMMARY");

    submit_turn(&test, &old_summary_user).await?;
    submit_turn(&test, "REMOTE_V2_NEWLY_COLD_POST_SUMMARY_USER").await?;
    submit_turn(&test, "REMOTE_V2_EMPTY_WITH_OLD_SUMMARY_HOT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    let error_message = wait_for_error_message(&test).await;

    assert!(
        error_message.contains("ExactTailNoUsableColdSummary"),
        "expected empty new v2 compaction output to fail even with retained old summary, got {error_message}"
    );
    let requests = response_mock.requests();
    assert_eq!(
        requests.len(),
        4,
        "empty v2 compaction should fail before any follow-up request"
    );
    let compact_body = requests[3].body_json().to_string();
    assert!(
        body_contains_text(&compact_body, "REMOTE_V2_OLD_RETAINED_SUMMARY"),
        "retained old summary-shaped user should be present in compact request; compact body: {compact_body}"
    );
    assert!(
        body_contains_text(&compact_body, "REMOTE_V2_NEWLY_COLD_POST_SUMMARY_USER"),
        "new post-summary cold user should be present in compact request; compact body: {compact_body}"
    );
    assert!(
        !body_contains_text(&compact_body, "REMOTE_V2_RETAINED_HOT"),
        "hot assistant must stay out of failed v2 compact request; compact body: {compact_body}"
    );
    assert!(
        replacement_history_from_rollout(&rollout_path).is_err(),
        "failed remote v2 exact-tail output must not install compacted history"
    );
    let diagnostics = exact_tail_diagnostics_from_rollout(&rollout_path)?;
    assert_eq!(diagnostics.len(), 1);
    let diagnostic = &diagnostics[0];
    assert_eq!(
        diagnostic.get("route").and_then(Value::as_str),
        Some("remote_v2")
    );
    assert_eq!(
        diagnostic.get("fit_result").and_then(Value::as_str),
        Some("failure")
    );
    assert_eq!(
        diagnostic.get("failure_reason").and_then(Value::as_str),
        Some("ExactTailNoUsableColdSummary")
    );
    assert_eq!(
        diagnostic
            .get("hot_suffix_exact_match")
            .and_then(Value::as_bool),
        None
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_remote_v2_enabled_auto_uses_cold_only_v2_request() -> Result<()> {
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message(
                    "remote-v2-route-auto-cold-assistant",
                    "REMOTE_V2_ROUTE_AUTO_COLD_ASSISTANT",
                ),
                ev_completed_with_tokens(
                    "remote-v2-route-auto-cold-response",
                    /*total_tokens*/ 50,
                ),
            ]),
            sse(vec![
                ev_assistant_message(
                    "remote-v2-route-auto-hot-assistant",
                    "REMOTE_V2_ROUTE_AUTO_ASSISTANT_EXACT",
                ),
                ev_completed_with_tokens(
                    "remote-v2-route-auto-hot-response",
                    /*total_tokens*/ 500,
                ),
            ]),
            sse(vec![
                ev_compaction_item("REMOTE_V2_ROUTE_AUTO_EXACT_TAIL_SUMMARY"),
                ev_completed("remote-v2-route-auto-compact-response"),
            ]),
            sse(vec![
                ev_assistant_message(
                    "remote-v2-route-auto-follow-up-assistant",
                    "REMOTE_V2_ROUTE_AUTO_FOLLOW_UP_DONE",
                ),
                ev_completed("remote-v2-route-auto-follow-up-response"),
            ]),
        ],
    )
    .await;
    let mut builder = test_codex()
        .with_auth(CodexAuth::from_api_key("dummy"))
        .with_config(|config| {
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.model_auto_compact_token_limit = Some(100);
            config.model_auto_compact_token_limit_scope =
                AutoCompactTokenLimitScope::BodyAfterPrefix;
            config.compact_preserve_recent_tokens = Some(1);
            explicitly_enable_remote_compaction_v2(config);
        });
    let test = builder.build(&server).await?;

    submit_turn(&test, "REMOTE_V2_ROUTE_AUTO_COLD_USER").await?;
    submit_turn(&test, "REMOTE_V2_ROUTE_AUTO_HOT_USER").await?;
    submit_turn(&test, "REMOTE_V2_ROUTE_AUTO_FOLLOW_UP_USER").await?;

    let requests = response_mock.requests();
    assert_eq!(
        requests.len(),
        4,
        "expected cold turn, hot turn, v2 compaction turn, and post-compaction follow-up"
    );
    let compact_request = &requests[2];
    assert_eq!(compact_request.path(), "/v1/responses");
    let compact_body = compact_request.body_json();
    let compact_body_text = compact_body.to_string();
    assert_eq!(
        compact_body.get("previous_response_id"),
        None,
        "exact-tail v2 auto compaction must not continue a previous Responses chain"
    );
    assert!(
        compact_body
            .get("prompt_cache_key")
            .and_then(Value::as_str)
            .is_some_and(|key| key.starts_with("exact-tail-v2-compaction:")),
        "exact-tail v2 auto compaction must use an isolated prompt cache key; body: {compact_body}"
    );
    assert!(
        body_contains_text(&compact_body_text, "REMOTE_V2_ROUTE_AUTO_COLD_USER"),
        "cold user should be sent to v2 auto compact; compact body: {compact_body}"
    );
    assert!(
        body_contains_text(&compact_body_text, "REMOTE_V2_ROUTE_AUTO_COLD_ASSISTANT"),
        "cold assistant should be sent to v2 auto compact; compact body: {compact_body}"
    );
    assert!(
        body_contains_text(&compact_body_text, "REMOTE_V2_ROUTE_AUTO_HOT_USER"),
        "older same-turn user should remain cold when the target only selects the newest atomic group; compact body: {compact_body}"
    );
    assert!(
        !body_contains_text(&compact_body_text, "REMOTE_V2_ROUTE_AUTO_ASSISTANT_EXACT"),
        "hot assistant must stay out of the v2 auto compact request; compact body: {compact_body}"
    );
    assert_eq!(
        compact_request
            .input()
            .iter()
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("compaction_trigger"))
            .count(),
        1,
        "v2 auto compact input should append exactly one compaction_trigger"
    );
    assert_eq!(
        compact_request
            .input()
            .last()
            .and_then(|item| item.get("type"))
            .and_then(Value::as_str),
        Some("compaction_trigger"),
        "v2 auto compact input should end with compaction_trigger"
    );
    assert_ordered_input_texts(
        &requests[3].input(),
        &[
            "REMOTE_V2_ROUTE_AUTO_COLD_USER",
            "REMOTE_V2_ROUTE_AUTO_HOT_USER",
            "REMOTE_V2_ROUTE_AUTO_EXACT_TAIL_SUMMARY",
            "REMOTE_V2_ROUTE_AUTO_ASSISTANT_EXACT",
            "REMOTE_V2_ROUTE_AUTO_FOLLOW_UP_USER",
        ],
    );
    assert!(
        !input_contains_text(&requests[3].input(), "REMOTE_V2_ROUTE_AUTO_COLD_ASSISTANT"),
        "follow-up should not replay cold assistant text outside the v2 auto summary"
    );

    shutdown_codex(&test).await?;
    Ok(())
}

fn header_contains_remote_v2(header: &str) -> bool {
    header
        .split(',')
        .any(|feature| feature.trim() == "remote_compaction_v2")
}
