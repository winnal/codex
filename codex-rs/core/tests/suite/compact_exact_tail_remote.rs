use super::compact_exact_tail_support::*;
use pretty_assertions::assert_eq;

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
        metadata: None,
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

    test.submit_turn(&cold_user).await?;
    test.submit_turn(&hot_user).await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    test.submit_turn("REMOTE_FOLLOW_UP_USER").await?;

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
        metadata: None,
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

    test.submit_turn(&cold_user).await?;
    test.submit_turn(&hot_user).await?;
    test.submit_turn("REMOTE_AUTO_FOLLOW_UP_USER").await?;

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
        metadata: None,
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

    test.submit_turn("REMOTE_USER_ONLY_COLD_USER").await?;
    test.submit_turn("REMOTE_USER_ONLY_HOT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    let error_message = wait_for_event_match(&test.codex, |event| match event {
        EventMsg::Error(err) => Some(err.message.clone()),
        _ => None,
    })
    .await;

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
async fn exact_tail_remote_v2_enabled_manual_routes_to_legacy_remote() -> Result<()> {
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("remote-v2-route-cold-assistant", "REMOTE_V2_ROUTE_COLD"),
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
                ev_assistant_message(
                    "remote-v2-route-follow-up-assistant",
                    "REMOTE_V2_ROUTE_FOLLOW_UP_DONE",
                ),
                ev_completed("remote-v2-route-follow-up-response"),
            ]),
        ],
    )
    .await;
    let compacted_history = vec![codex_protocol::models::ResponseItem::Compaction {
        id: None,
        encrypted_content: "REMOTE_V2_ROUTE_EXACT_TAIL_SUMMARY".to_string(),
        metadata: None,
    }];
    let compact_mock =
        mount_compact_json_once(&server, serde_json::json!({ "output": compacted_history })).await;
    let mut builder = test_codex()
        .with_auth(CodexAuth::from_api_key("dummy"))
        .with_config(|config| {
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.compact_preserve_recent_tokens = Some(1);
            let _ = config.features.enable(Feature::RemoteCompactionV2);
        });
    let test = builder.build(&server).await?;

    test.submit_turn("REMOTE_V2_ROUTE_COLD_USER").await?;
    test.submit_turn("REMOTE_V2_ROUTE_HOT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    test.submit_turn("REMOTE_V2_ROUTE_FOLLOW_UP_USER").await?;

    assert_eq!(
        compact_mock.requests().len(),
        1,
        "exact-tail should route remote-v2-enabled manual compaction through legacy remote"
    );
    let compact_body = compact_mock.single_request().body_json().to_string();
    assert!(
        body_contains_text(&compact_body, "REMOTE_V2_ROUTE_COLD_USER"),
        "cold user should be sent to legacy remote compact; compact body: {compact_body}"
    );
    assert_eq!(
        response_mock.requests().len(),
        3,
        "expected cold turn, hot turn, and follow-up"
    );
    assert_ordered_input_texts(
        &response_mock.requests()[2].input(),
        &[
            "REMOTE_V2_ROUTE_EXACT_TAIL_SUMMARY",
            "REMOTE_V2_ROUTE_ASSISTANT_EXACT",
            "REMOTE_V2_ROUTE_FOLLOW_UP_USER",
        ],
    );
    assert!(
        !input_contains_text(
            &response_mock.requests()[2].input(),
            "REMOTE_V2_ROUTE_HOT_USER"
        ),
        "follow-up should not replay older same-turn user text outside the remote-v2-routed summary"
    );

    shutdown_codex(&test).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_remote_v2_enabled_auto_routes_to_legacy_remote() -> Result<()> {
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message(
                    "remote-v2-route-auto-cold-assistant",
                    "REMOTE_V2_ROUTE_AUTO_COLD",
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
                ev_assistant_message(
                    "remote-v2-route-auto-follow-up-assistant",
                    "REMOTE_V2_ROUTE_AUTO_FOLLOW_UP_DONE",
                ),
                ev_completed("remote-v2-route-auto-follow-up-response"),
            ]),
        ],
    )
    .await;
    let compacted_history = vec![codex_protocol::models::ResponseItem::Compaction {
        id: None,
        encrypted_content: "REMOTE_V2_ROUTE_AUTO_EXACT_TAIL_SUMMARY".to_string(),
        metadata: None,
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
            let _ = config.features.enable(Feature::RemoteCompactionV2);
        });
    let test = builder.build(&server).await?;

    test.submit_turn("REMOTE_V2_ROUTE_AUTO_COLD_USER").await?;
    test.submit_turn("REMOTE_V2_ROUTE_AUTO_HOT_USER").await?;
    test.submit_turn("REMOTE_V2_ROUTE_AUTO_FOLLOW_UP_USER")
        .await?;

    assert_eq!(
        compact_mock.requests().len(),
        1,
        "exact-tail should route remote-v2-enabled auto compaction through legacy remote"
    );
    let compact_body = compact_mock.single_request().body_json().to_string();
    assert!(
        body_contains_text(&compact_body, "REMOTE_V2_ROUTE_AUTO_COLD_USER"),
        "cold user should be sent to legacy remote auto compact; compact body: {compact_body}"
    );
    assert_eq!(
        response_mock.requests().len(),
        3,
        "expected cold turn, hot turn, and post-compaction follow-up"
    );
    assert_ordered_input_texts(
        &response_mock.requests()[2].input(),
        &[
            "REMOTE_V2_ROUTE_AUTO_EXACT_TAIL_SUMMARY",
            "REMOTE_V2_ROUTE_AUTO_ASSISTANT_EXACT",
            "REMOTE_V2_ROUTE_AUTO_FOLLOW_UP_USER",
        ],
    );
    assert!(
        !input_contains_text(
            &response_mock.requests()[2].input(),
            "REMOTE_V2_ROUTE_AUTO_HOT_USER"
        ),
        "follow-up should not replay older same-turn user text outside the remote-v2-routed auto summary"
    );

    shutdown_codex(&test).await?;
    Ok(())
}
