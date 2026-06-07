use super::rolling_prompt_support::turn_metadata;
use super::rolling_prompt_support::wait_for_turn_complete;
use anyhow::Result;
use codex_core::config::Config;
use codex_protocol::config_types::PromptRetentionMode;
use codex_protocol::config_types::RollingCompactionMode;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use core_test_support::responses;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::json;

fn configure_pairwise_bootstrap(config: &mut Config) {
    configure_pairwise(config);
    config.max_summary_levels = Some(1);
}

fn configure_pairwise(config: &mut Config) {
    config.prompt_retention = PromptRetentionMode::Rolling;
    config.rolling_compaction = RollingCompactionMode::Pairwise;
    config.rolling_context_reserve_percent = Some(0);
    config.rolling_context_target_tokens = Some(50_000);
    config.protected_hot_exact_tokens = Some(1);
    config.summary_group_token_cap = Some(256);
    config.max_summary_levels = Some(3);
    config.compact_when_level_group_count_gt = Some(2);
    config.include_permissions_instructions = false;
    config.include_apps_instructions = false;
    config.include_collaboration_mode_instructions = false;
    config.include_skill_instructions = false;
    config.include_environment_context = false;
}

async fn submit_text_turn(test: &TestCodex, text: &str) -> Result<String> {
    Ok(test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: text.into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?)
}

fn many_assistant_messages(prefix: &str, count: usize) -> Vec<serde_json::Value> {
    let mut events = Vec::new();
    for index in 0..count {
        events.push(responses::ev_assistant_message(
            &format!("{prefix}-id-{index}"),
            &format!("{prefix}_SENTINEL_{index:02}"),
        ));
    }
    events
}

fn summary_response(index: usize) -> String {
    summary_response_with_prefix("BOOT", index)
}

fn summary_response_with_prefix(prefix: &str, index: usize) -> String {
    sse(vec![
        responses::ev_assistant_message(
            &format!("{prefix}-summary-msg-{index}"),
            &format!("{prefix}_PAIRWISE_SUMMARY_SENTINEL_{index:02}"),
        ),
        responses::ev_completed(&format!("{prefix}-summary-resp-{index}")),
    ])
}

fn incomplete_summary_response() -> String {
    sse(vec![json!({
        "type": "response.incomplete",
        "response": {
            "id": "bootstrap-summary-incomplete",
            "incomplete_details": {"reason": "max_output_tokens"}
        }
    })])
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pairwise_natural_backfill_converges_when_each_turn_fits() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut seed_events = many_assistant_messages("NAT_ASSISTANT", 3);
    seed_events.push(responses::ev_completed("seed-resp"));
    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(seed_events),
            summary_response_with_prefix("NAT", 0),
            sse(vec![
                responses::ev_assistant_message("natural-turn-one", "natural turn one done"),
                responses::ev_completed("natural-turn-resp-one"),
            ]),
            summary_response_with_prefix("NAT", 1),
            sse(vec![
                responses::ev_assistant_message("natural-turn-two", "natural turn two done"),
                responses::ev_completed("natural-turn-resp-two"),
            ]),
        ],
    )
    .await;
    let test = test_codex()
        .with_config(configure_pairwise)
        .build(&server)
        .await?;

    let seed_turn_id = submit_text_turn(&test, "NAT_USER_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &seed_turn_id).await?;
    let first_turn_id = submit_text_turn(&test, "NAT_HOT_SENTINEL_ONE").await?;
    wait_for_turn_complete(&test.codex, &first_turn_id).await?;
    let second_turn_id = submit_text_turn(&test, "NAT_HOT_SENTINEL_TWO").await?;
    wait_for_turn_complete(&test.codex, &second_turn_id).await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 5);
    assert_eq!(captured[1].body_json()["tools"], json!([]));
    assert_eq!(captured[3].body_json()["tools"], json!([]));
    assert!(captured[1].body_contains_text("NAT_USER_SENTINEL"));
    assert!(captured[2].body_contains_text("NAT_PAIRWISE_SUMMARY_SENTINEL_00"));
    assert!(!captured[2].body_contains_text("NAT_USER_SENTINEL"));
    assert!(!captured[3].body_contains_text("NAT_USER_SENTINEL"));
    assert!(captured[4].body_contains_text("NAT_PAIRWISE_SUMMARY_SENTINEL_01"));
    assert!(!captured[4].body_contains_text("NAT_USER_SENTINEL"));
    assert!(captured[4].body_contains_text("NAT_HOT_SENTINEL_TWO"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pairwise_bootstrap_resumes_until_prompt_fits() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut seed_events = many_assistant_messages("BOOT_ASSISTANT", 80);
    seed_events.push(responses::ev_completed("seed-resp"));
    let mut response_sequence = vec![sse(seed_events)];
    for index in 0..83 {
        response_sequence.push(summary_response(index));
    }
    let requests = mount_sse_sequence(&server, response_sequence).await;
    let test = test_codex()
        .with_config(configure_pairwise_bootstrap)
        .build(&server)
        .await?;

    let seed_turn_id = submit_text_turn(&test, "BOOT_USER_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &seed_turn_id).await?;

    let mut failed_bootstrap_passes = 0usize;
    let mut successful_attempt = None::<usize>;
    for attempt in 0..8 {
        let turn_id = submit_text_turn(&test, &format!("BOOT_HOT_SENTINEL_{attempt}")).await?;
        match wait_for_turn_complete(&test.codex, &turn_id).await {
            Ok(()) => {
                successful_attempt = Some(attempt);
                break;
            }
            Err(error) => {
                assert!(
                    error
                        .to_string()
                        .contains("Codex ran out of room in the model's context window"),
                    "unexpected bootstrap error: {error:?}"
                );
                failed_bootstrap_passes += 1;
                for request in requests.requests().iter().skip(1) {
                    assert_eq!(request.body_json()["tools"], json!([]));
                    assert_eq!(request.header("x-codex-turn-metadata"), None);
                }
            }
        }
    }
    let successful_attempt =
        successful_attempt.expect("cached bootstrap summaries should eventually fit");

    let captured = requests.requests();
    assert!(failed_bootstrap_passes > 0);
    assert!(captured.len() > 20, "captured requests: {captured:#?}");
    for request in captured
        .iter()
        .skip(1)
        .take(captured.len().saturating_sub(2))
    {
        assert_eq!(request.body_json()["tools"], json!([]));
    }
    let turn_request = captured.last().expect("normal turn request");
    let metadata = turn_metadata(turn_request)?;
    assert_eq!(metadata["request_kind"].as_str(), Some("turn"));
    assert!(turn_request.body_contains_text("BOOT_PAIRWISE_SUMMARY_SENTINEL"));
    assert!(!turn_request.body_contains_text("BOOT_USER_SENTINEL"));
    assert!(!turn_request.body_contains_text("BOOT_ASSISTANT_SENTINEL_00"));
    assert!(turn_request.body_contains_text(&format!("BOOT_HOT_SENTINEL_{successful_attempt}")));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pairwise_bootstrap_preserves_cache_after_normal_sampling_failure() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut seed_events = many_assistant_messages("BOOTFAIL_ASSISTANT", 80);
    seed_events.push(responses::ev_completed("bootfail-seed-resp"));
    let mut response_sequence = vec![sse(seed_events)];
    for index in 0..82 {
        response_sequence.push(summary_response_with_prefix("BOOTFAIL", index));
    }
    response_sequence.push(responses::sse_failed(
        "bootfail-normal-resp",
        "invalid_prompt",
        "mocked normal sampling failure after bootstrap cache is ready",
    ));
    response_sequence.push(sse(vec![
        responses::ev_assistant_message("bootfail-replay", "bootstrap replay done"),
        responses::ev_completed("bootfail-replay-resp"),
    ]));
    let requests = mount_sse_sequence(&server, response_sequence).await;
    let test = test_codex()
        .with_config(configure_pairwise_bootstrap)
        .build(&server)
        .await?;

    let seed_turn_id = submit_text_turn(&test, "BOOTFAIL_USER_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &seed_turn_id).await?;
    let mut failed_bootstrap_passes = 0usize;
    let mut failed_normal_index = None::<usize>;
    for attempt in 0..8 {
        let turn_id = submit_text_turn(&test, &format!("BOOTFAIL_HOT_SENTINEL_{attempt}")).await?;
        let error = wait_for_turn_complete(&test.codex, &turn_id)
            .await
            .expect_err("turn should fail until mocked normal sampling failure is reached");
        let message = error.to_string();
        if message.contains("Codex ran out of room in the model's context window") {
            failed_bootstrap_passes += 1;
            continue;
        }
        assert!(
            message.contains("mocked normal sampling failure"),
            "unexpected bootstrap/normal failure after {} requests: {error:?}",
            requests.requests().len()
        );
        failed_normal_index = requests.requests().len().checked_sub(1);
        break;
    }
    assert!(failed_bootstrap_passes > 0);
    let failed_normal_index = failed_normal_index.expect("mocked normal failure request");

    let captured_after_failure = requests.requests();
    let failed_metadata = turn_metadata(&captured_after_failure[failed_normal_index])?;
    assert_eq!(failed_metadata["request_kind"].as_str(), Some("turn"));

    let replay_turn_id = submit_text_turn(&test, "BOOTFAIL_REPLAY_HOT_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &replay_turn_id).await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), failed_normal_index + 2);
    let replay_request = captured.last().expect("replay normal request");
    let replay_metadata = turn_metadata(replay_request)?;
    assert_eq!(replay_metadata["request_kind"].as_str(), Some("turn"));
    assert!(replay_request.body_contains_text("BOOTFAIL_PAIRWISE_SUMMARY_SENTINEL"));
    assert!(!replay_request.body_contains_text("BOOTFAIL_USER_SENTINEL"));
    assert!(!replay_request.body_contains_text("BOOTFAIL_ASSISTANT_SENTINEL_00"));
    assert!(replay_request.body_contains_text("BOOTFAIL_REPLAY_HOT_SENTINEL"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pairwise_bootstrap_failed_summary_preserves_prior_cache_progress() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut seed_events = many_assistant_messages("PARTIAL_ASSISTANT", 6);
    seed_events.push(responses::ev_completed("partial-seed-resp"));
    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(seed_events),
            summary_response_with_prefix("PARTIAL", 0),
            summary_response_with_prefix("PARTIAL", 1),
            incomplete_summary_response(),
            summary_response_with_prefix("PARTIAL", 2),
            summary_response_with_prefix("PARTIAL", 3),
            sse(vec![
                responses::ev_assistant_message("partial-replay", "partial replay done"),
                responses::ev_completed("partial-replay-resp"),
            ]),
        ],
    )
    .await;
    let test = test_codex()
        .with_config(configure_pairwise)
        .build(&server)
        .await?;

    let seed_turn_id = submit_text_turn(&test, "PARTIAL_USER_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &seed_turn_id).await?;
    let failed_turn_id = submit_text_turn(&test, "PARTIAL_HOT_SENTINEL").await?;
    let error = wait_for_turn_complete(&test.codex, &failed_turn_id)
        .await
        .expect_err("incomplete bootstrap summary should fail before normal sampling");
    assert!(error.to_string().contains("max_output_tokens"));

    let replay_turn_id = submit_text_turn(&test, "PARTIAL_REPLAY_HOT_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &replay_turn_id).await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 7);
    assert_eq!(captured[1].body_json()["tools"], json!([]));
    assert_eq!(captured[2].body_json()["tools"], json!([]));
    assert_eq!(captured[3].body_json()["tools"], json!([]));
    assert_eq!(captured[4].body_json()["tools"], json!([]));
    assert!(captured[1].body_contains_text("PARTIAL_USER_SENTINEL"));
    assert!(captured[2].body_contains_text("PARTIAL_ASSISTANT_SENTINEL_01"));
    assert!(!captured[4].body_contains_text("PARTIAL_USER_SENTINEL"));
    assert!(!captured[4].body_contains_text("PARTIAL_ASSISTANT_SENTINEL_00"));
    let replay_request = captured.last().expect("replay normal request");
    let replay_metadata = turn_metadata(replay_request)?;
    assert_eq!(replay_metadata["request_kind"].as_str(), Some("turn"));
    assert!(replay_request.body_contains_text("PARTIAL_PAIRWISE_SUMMARY_SENTINEL_02"));
    assert!(!replay_request.body_contains_text("PARTIAL_USER_SENTINEL"));
    assert!(replay_request.body_contains_text("PARTIAL_REPLAY_HOT_SENTINEL"));

    Ok(())
}
