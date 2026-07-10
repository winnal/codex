use super::compact_exact_tail_support::*;
use codex_protocol::config_types::CompactExactTailStrategy;
use pretty_assertions::assert_eq;

const BETA_FEATURES_HEADER: &str = "x-codex-beta-features";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_semantic_transcript_manual_clears_cached_websocket_continuation() -> Result<()>
{
    skip_if_no_network!(Ok(()));

    let server = start_websocket_server_concurrent(vec![
        vec![
            ws_warm_response("semantic-manual-warm"),
            ws_assistant_response(
                "semantic-manual-cold",
                "semantic-manual-cold-response",
                "SEM_MANUAL_COLD",
                None,
            ),
            ws_assistant_response(
                "semantic-manual-hot",
                "semantic-manual-hot-response",
                "SEM_MANUAL_HOT",
                None,
            ),
            ws_assistant_response(
                "semantic-manual-stale",
                "semantic-manual-stale-response",
                "SEM_MANUAL_STALE_REUSE",
                None,
            ),
        ],
        vec![ws_compaction_response(
            "SEM_MANUAL_SUMMARY",
            "semantic-manual-compact-response",
        )],
        vec![ws_assistant_response(
            "semantic-manual-follow",
            "semantic-manual-follow-response",
            "SEM_MANUAL_FOLLOW",
            None,
        )],
    ])
    .await;
    let mut builder = semantic_builder().with_config(|config| {
        let _ = config.features.disable(Feature::RemoteCompactionV2);
    });
    let test = builder.build_with_websocket_server(&server).await?;
    wait_for_startup_websocket_prewarm(&server).await;

    submit_turn(&test, "SEM_MANUAL_COLD_USER").await?;
    submit_turn(&test, "SEM_MANUAL_HOT_USER").await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        Duration::from_secs(90),
    )
    .await;
    submit_turn(&test, "SEM_MANUAL_FOLLOW_USER").await?;

    let connections = server.connections();
    assert_eq!(connections.len(), 3);
    assert_eq!(
        connections[0].len(),
        3,
        "manual follow-up must not reuse the pre-compaction websocket"
    );
    assert_v2_compaction_handshake(&server, /*connection_index*/ 1);
    assert_semantic_v2_compaction_request(&connections[1][0]);
    assert_follow_up_does_not_reuse_stale_continuation(
        &connections[2],
        "semantic-manual-hot-response",
        "semantic manual compaction must clear stale pre-compaction continuation",
    );

    shutdown_codex(&test).await?;
    server.shutdown().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_tail_semantic_transcript_auto_clears_cached_websocket_continuation() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_websocket_server_concurrent(vec![
        vec![
            ws_warm_response("semantic-auto-warm"),
            ws_assistant_response(
                "semantic-auto-cold",
                "semantic-auto-cold-response",
                "SEM_AUTO_WS_COLD",
                Some(50),
            ),
            ws_assistant_response(
                "semantic-auto-hot",
                "semantic-auto-hot-response",
                "SEM_AUTO_WS_HOT",
                Some(500),
            ),
            ws_assistant_response(
                "semantic-auto-stale",
                "semantic-auto-stale-response",
                "SEM_AUTO_WS_STALE_REUSE",
                None,
            ),
        ],
        vec![ws_compaction_response(
            "SEM_AUTO_WS_SUMMARY",
            "semantic-auto-compact-response",
        )],
        vec![ws_assistant_response(
            "semantic-auto-follow",
            "semantic-auto-follow-response",
            "SEM_AUTO_WS_FOLLOW",
            None,
        )],
    ])
    .await;
    let mut builder = semantic_builder().with_config(|config| {
        config.model_auto_compact_token_limit = Some(100);
        config.model_auto_compact_token_limit_scope = AutoCompactTokenLimitScope::BodyAfterPrefix;
    });
    let test = builder.build_with_websocket_server(&server).await?;
    wait_for_startup_websocket_prewarm(&server).await;

    submit_turn(&test, "SEM_AUTO_WS_COLD_USER").await?;
    submit_turn(&test, "SEM_AUTO_WS_HOT_USER").await?;
    submit_turn(&test, "SEM_AUTO_WS_FOLLOW_USER").await?;

    let connections = server.connections();
    assert_eq!(connections.len(), 3);
    assert_eq!(
        connections[0].len(),
        3,
        "auto follow-up must not reuse the pre-compaction websocket"
    );
    assert_v2_compaction_handshake(&server, /*connection_index*/ 1);
    assert_semantic_v2_compaction_request(&connections[1][0]);
    assert_follow_up_does_not_reuse_stale_continuation(
        &connections[2],
        "semantic-auto-hot-response",
        "semantic auto compaction must clear stale pre-compaction continuation",
    );

    shutdown_codex(&test).await?;
    server.shutdown().await;
    Ok(())
}

fn semantic_builder() -> core_test_support::test_codex::TestCodexBuilder {
    test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_config(|config| {
            set_test_compact_prompt(config);
            config.model_context_window = Some(200_000);
            config.compact_preserve_recent_tokens = Some(1);
            config.compact_exact_tail_strategy = CompactExactTailStrategy::SemanticTranscript;
            config.compact_exact_tail_semantic_transcript_retained_message_token_budget = 32_000;
        })
}

async fn submit_turn(test: &TestCodex, prompt: &str) -> Result<()> {
    test.submit_turn_with_completion_timeout(prompt, Duration::from_secs(90))
        .await
}

async fn wait_for_startup_websocket_prewarm(
    server: &core_test_support::responses::WebSocketTestServer,
) {
    let request = tokio::time::timeout(
        Duration::from_secs(90),
        server.wait_for_request(/*connection_index*/ 0, /*request_index*/ 0),
    )
    .await
    .expect("startup websocket prewarm request");
    assert_eq!(request.body_json()["generate"].as_bool(), Some(false));
}

fn assert_v2_compaction_handshake(
    server: &core_test_support::responses::WebSocketTestServer,
    connection_index: usize,
) {
    assert!(
        server
            .handshakes()
            .get(connection_index)
            .and_then(|handshake| handshake.header(BETA_FEATURES_HEADER))
            .as_deref()
            .is_some_and(header_contains_remote_v2),
        "semantic v2 compaction must advertise remote_compaction_v2 on its websocket handshake"
    );
}

fn assert_follow_up_does_not_reuse_stale_continuation(
    post_compaction: &[core_test_support::responses::WebSocketRequest],
    stale_response_id: &str,
    message: &str,
) {
    assert_eq!(post_compaction.len(), 1);
    let follow_body = post_compaction[0].body_json();
    assert_ne!(
        follow_body
            .get("previous_response_id")
            .and_then(Value::as_str),
        Some(stale_response_id),
        "{message}"
    );
    assert_eq!(follow_body.get("previous_response_id"), None);
    assert!(
        !follow_body
            .get("prompt_cache_key")
            .and_then(Value::as_str)
            .is_some_and(|key| key.starts_with("semantic-transcript-v2-compaction:")),
        "post-compaction sampling must not inherit the isolated semantic compact cache key"
    );
}

fn assert_semantic_v2_compaction_request(request: &core_test_support::responses::WebSocketRequest) {
    let body = request.body_json();
    assert_eq!(body.get("previous_response_id"), None);
    assert!(
        body.get("prompt_cache_key")
            .and_then(Value::as_str)
            .is_some_and(|key| key.starts_with("semantic-transcript-v2-compaction:")),
        "semantic v2 compaction must use an isolated prompt cache key; body: {body}"
    );
    assert_eq!(
        body.get("input").and_then(Value::as_array).map(|input| {
            input
                .iter()
                .filter(|item| {
                    item.get("type").and_then(Value::as_str) == Some("compaction_trigger")
                })
                .count()
        }),
        Some(1),
        "semantic v2 compaction input should contain exactly one compaction_trigger; body: {body}"
    );
    assert_eq!(
        body.get("input")
            .and_then(Value::as_array)
            .and_then(|input| input.last())
            .and_then(|item| item.get("type"))
            .and_then(Value::as_str),
        Some("compaction_trigger"),
        "semantic v2 compaction input should end with compaction_trigger; body: {body}"
    );
}

fn header_contains_remote_v2(header: &str) -> bool {
    header
        .split(',')
        .any(|feature| feature == "remote_compaction_v2")
}
