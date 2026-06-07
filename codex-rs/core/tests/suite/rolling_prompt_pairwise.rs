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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn uses_no_history_summary_request_before_turn_prompt() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                responses::ev_assistant_message("seed-a", "COVERED_ASSISTANT_SENTINEL_A"),
                responses::ev_assistant_message("seed-b", "COVERED_ASSISTANT_SENTINEL_B"),
                responses::ev_assistant_message("seed-c", "COLD_WORKBENCH_SENTINEL"),
                responses::ev_completed("seed-resp"),
            ]),
            sse(vec![
                responses::ev_output_text_delta("LIVE_PAIRWISE_SUMMARY_SENTINEL"),
                responses::ev_completed("summary-resp"),
            ]),
            sse(vec![
                responses::ev_assistant_message("turn-msg", "pairwise turn done"),
                responses::ev_completed("turn-resp"),
            ]),
        ],
    )
    .await;
    let test = test_codex()
        .with_config(configure_pairwise)
        .build(&server)
        .await?;

    let seed_turn_id = submit_text_turn(&test, "COVERED_USER_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &seed_turn_id).await?;
    let pairwise_turn_id = submit_text_turn(&test, "HOT_USER_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &pairwise_turn_id).await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 3);
    let seed_request = &captured[0];
    let summary_request = &captured[1];
    let turn_request = &captured[2];

    assert!(seed_request.body_contains_text("COVERED_USER_SENTINEL"));
    assert_eq!(summary_request.body_json()["tools"], json!([]));
    assert_eq!(summary_request.body_json()["parallel_tool_calls"], false);
    assert_eq!(summary_request.body_json()["max_output_tokens"], json!(256));
    assert_eq!(summary_request.header("x-codex-turn-metadata"), None);
    assert!(
        summary_request
            .instructions_text()
            .contains("ROLLCTX pair summary")
    );
    assert!(summary_request.body_contains_text("COVERED_USER_SENTINEL"));
    assert!(summary_request.body_contains_text("COVERED_ASSISTANT_SENTINEL_A"));
    assert!(!summary_request.body_contains_text("HOT_USER_SENTINEL"));

    let turn_metadata = turn_metadata(turn_request)?;
    assert_eq!(turn_metadata["request_kind"].as_str(), Some("turn"));
    assert!(turn_request.body_contains_text("LIVE_PAIRWISE_SUMMARY_SENTINEL"));
    assert!(!turn_request.body_contains_text("COVERED_USER_SENTINEL"));
    assert!(!turn_request.body_contains_text("COVERED_ASSISTANT_SENTINEL_A"));
    assert!(turn_request.body_contains_text("COLD_WORKBENCH_SENTINEL"));
    assert!(turn_request.body_contains_text("HOT_USER_SENTINEL"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn budget_pressure_compacts_threshold_pair_before_failing() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let cold_user = "BUDGET_COLD_USER_SENTINEL";
    let cold_assistant = format!(
        "BUDGET_COLD_ASSISTANT_SENTINEL {}",
        "assistant-cold ".repeat(1_000)
    );
    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                responses::ev_assistant_message("seed-assistant", &cold_assistant),
                responses::ev_completed("seed-resp"),
            ]),
            sse(vec![
                responses::ev_output_text_delta("BUDGET_PAIRWISE_SUMMARY_SENTINEL"),
                responses::ev_completed("summary-resp"),
            ]),
            sse(vec![
                responses::ev_assistant_message("turn-msg", "budget turn done"),
                responses::ev_completed("turn-resp"),
            ]),
        ],
    )
    .await;
    let test = test_codex()
        .with_config(|config| {
            configure_pairwise(config);
            config.base_instructions = Some("base".to_string());
            config.rolling_context_target_tokens = Some(500);
        })
        .build(&server)
        .await?;

    let seed_turn_id = submit_text_turn(&test, cold_user).await?;
    wait_for_turn_complete(&test.codex, &seed_turn_id).await?;
    let pairwise_turn_id = submit_text_turn(&test, "BUDGET_HOT_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &pairwise_turn_id).await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 3);
    let summary_request = &captured[1];
    let turn_request = &captured[2];
    assert!(summary_request.body_contains_text("BUDGET_COLD_USER_SENTINEL"));
    assert!(summary_request.body_contains_text("BUDGET_COLD_ASSISTANT_SENTINEL"));
    assert_eq!(summary_request.body_json()["tools"], json!([]));
    assert!(turn_request.body_contains_text("BUDGET_PAIRWISE_SUMMARY_SENTINEL"));
    assert!(!turn_request.body_contains_text("BUDGET_COLD_USER_SENTINEL"));
    assert!(!turn_request.body_contains_text("BUDGET_COLD_ASSISTANT_SENTINEL"));
    assert!(turn_request.body_contains_text("BUDGET_HOT_SENTINEL"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retry_reuses_generated_pair_summary_after_context_window_error() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                responses::ev_assistant_message("seed-a", "RETRY_ASSISTANT_SENTINEL_A"),
                responses::ev_assistant_message("seed-b", "RETRY_ASSISTANT_SENTINEL_B"),
                responses::ev_assistant_message("seed-c", "RETRY_COLD_WORKBENCH_SENTINEL"),
                responses::ev_completed("seed-resp"),
            ]),
            sse(vec![
                responses::ev_output_text_delta("RETRY_PAIRWISE_SUMMARY_SENTINEL"),
                responses::ev_completed("summary-resp"),
            ]),
            responses::sse_failed(
                "overflow-resp",
                "context_length_exceeded",
                "Your input exceeds the context window of this model. Please adjust your input and try again.",
            ),
            sse(vec![
                responses::ev_assistant_message("retry-msg", "retry pairwise turn done"),
                responses::ev_completed("retry-resp"),
            ]),
        ],
    )
    .await;
    let test = test_codex()
        .with_config(configure_pairwise)
        .build(&server)
        .await?;

    let seed_turn_id = submit_text_turn(&test, "RETRY_USER_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &seed_turn_id).await?;
    let pairwise_turn_id = submit_text_turn(&test, "RETRY_HOT_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &pairwise_turn_id).await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 4);
    assert_eq!(captured[1].body_json()["tools"], json!([]));
    assert_eq!(captured[1].header("x-codex-turn-metadata"), None);
    assert!(captured[1].body_contains_text("RETRY_USER_SENTINEL"));

    let retry_metadata = turn_metadata(&captured[3])?;
    assert_eq!(retry_metadata["request_kind"].as_str(), Some("turn"));
    assert!(captured[3].body_contains_text("RETRY_PAIRWISE_SUMMARY_SENTINEL"));
    assert!(!captured[3].body_contains_text("RETRY_USER_SENTINEL"));
    assert!(!captured[3].body_contains_text("RETRY_ASSISTANT_SENTINEL_A"));
    assert!(captured[3].body_contains_text("RETRY_COLD_WORKBENCH_SENTINEL"));
    assert!(captured[3].body_contains_text("RETRY_HOT_SENTINEL"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_pairwise_mocked_side_channel_stress_respects_cap_and_commit_semantics()
-> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                responses::ev_assistant_message("seed-a", "COMMIT_ASSISTANT_SENTINEL_A"),
                responses::ev_assistant_message("seed-b", "COMMIT_ASSISTANT_SENTINEL_B"),
                responses::ev_assistant_message("seed-c", "COMMIT_COLD_WORKBENCH_SENTINEL"),
                responses::ev_completed("seed-resp"),
            ]),
            sse(vec![
                responses::ev_output_text_delta("COMMIT_PAIRWISE_SUMMARY_SENTINEL"),
                responses::ev_completed("summary-resp"),
            ]),
            sse(vec![
                responses::ev_assistant_message("turn-msg", "commit pairwise turn done"),
                responses::ev_completed("turn-resp"),
            ]),
            sse(vec![
                responses::ev_output_text_delta("COMMIT_LATER_SUMMARY_SENTINEL"),
                responses::ev_completed("later-summary-resp"),
            ]),
            sse(vec![
                responses::ev_assistant_message("reuse-msg", "reuse pairwise turn done"),
                responses::ev_completed("reuse-resp"),
            ]),
        ],
    )
    .await;
    let test = test_codex()
        .with_config(configure_pairwise)
        .build(&server)
        .await?;

    let seed_turn_id = submit_text_turn(&test, "COMMIT_USER_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &seed_turn_id).await?;
    let first_turn_id = submit_text_turn(&test, "COMMIT_HOT_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &first_turn_id).await?;
    let reuse_turn_id = submit_text_turn(&test, "COMMIT_REUSE_HOT_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &reuse_turn_id).await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 5);
    assert_eq!(captured[1].body_json()["tools"], json!([]));
    assert_eq!(captured[3].body_json()["tools"], json!([]));
    assert_eq!(captured[1].header("x-codex-turn-metadata"), None);
    assert!(captured[1].body_contains_text("COMMIT_USER_SENTINEL"));
    assert!(captured[2].body_contains_text("COMMIT_PAIRWISE_SUMMARY_SENTINEL"));
    assert!(!captured[2].body_contains_text("COMMIT_USER_SENTINEL"));
    assert!(!captured[3].body_contains_text("COMMIT_USER_SENTINEL"));
    assert!(captured[4].body_contains_text("COMMIT_LATER_SUMMARY_SENTINEL"));
    assert!(!captured[4].body_contains_text("COMMIT_USER_SENTINEL"));

    let server = start_mock_server().await;
    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                responses::ev_assistant_message("fail-seed-a", "FAIL_ASSISTANT_SENTINEL_A"),
                responses::ev_assistant_message("fail-seed-b", "FAIL_ASSISTANT_SENTINEL_B"),
                responses::ev_assistant_message("fail-seed-c", "FAIL_COLD_WORKBENCH_SENTINEL"),
                responses::ev_completed("fail-seed-resp"),
            ]),
            sse(vec![
                responses::ev_output_text_delta("FAIL_PAIRWISE_SUMMARY_SENTINEL"),
                responses::ev_completed("fail-summary-resp"),
            ]),
            responses::sse_failed(
                "fail-turn-resp",
                "invalid_prompt",
                "mocked normal sampling failure before committing rolling state",
            ),
            sse(vec![
                responses::ev_assistant_message(
                    "replay-summary-a",
                    "FAIL_REPLAY_PAIRWISE_SUMMARY_SENTINEL_A",
                ),
                responses::ev_completed("replay-summary-resp-a"),
            ]),
            sse(vec![
                responses::ev_assistant_message("replay", "normal replay done"),
                responses::ev_completed("replay-resp"),
            ]),
        ],
    )
    .await;
    let test = test_codex()
        .with_config(configure_pairwise)
        .build(&server)
        .await?;

    let seed_turn_id = submit_text_turn(&test, "FAIL_USER_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &seed_turn_id).await?;
    let failed_turn_id = submit_text_turn(&test, "FAIL_HOT_SENTINEL").await?;
    let error = wait_for_turn_complete(&test.codex, &failed_turn_id)
        .await
        .expect_err("normal sampling failure should abort before committing summary state");
    assert!(error.to_string().contains("mocked normal sampling failure"));
    let replay_turn_id = submit_text_turn(&test, "FAIL_REPLAY_HOT_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &replay_turn_id).await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 5);
    assert_eq!(captured[1].body_json()["tools"], json!([]));
    assert!(captured[1].body_contains_text("FAIL_USER_SENTINEL"));
    assert_eq!(captured[3].body_json()["tools"], json!([]));
    assert!(!captured[3].body_contains_text("FAIL_USER_SENTINEL"));
    let replay_turn_request = captured.last().expect("replay turn request");
    let replay_metadata = turn_metadata(replay_turn_request)?;
    assert_eq!(replay_metadata["request_kind"].as_str(), Some("turn"));
    assert!(replay_turn_request.body_contains_text("FAIL_PAIRWISE_SUMMARY_SENTINEL"));
    assert!(!replay_turn_request.body_contains_text("FAIL_USER_SENTINEL"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn does_not_publish_incomplete_summary() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                responses::ev_assistant_message("seed-a", "INCOMPLETE_ASSISTANT_SENTINEL_A"),
                responses::ev_assistant_message("seed-b", "INCOMPLETE_ASSISTANT_SENTINEL_B"),
                responses::ev_completed("seed-resp"),
            ]),
            sse(vec![json!({
                "type": "response.incomplete",
                "response": {
                    "id": "summary-incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"}
                }
            })]),
        ],
    )
    .await;
    let test = test_codex()
        .with_config(configure_pairwise)
        .build(&server)
        .await?;

    let seed_turn_id = submit_text_turn(&test, "INCOMPLETE_USER_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &seed_turn_id).await?;
    let pairwise_turn_id = submit_text_turn(&test, "INCOMPLETE_HOT_SENTINEL").await?;
    let error = wait_for_turn_complete(&test.codex, &pairwise_turn_id)
        .await
        .expect_err("incomplete pair-summary output must fail before prompt publication");

    assert!(error.to_string().contains("max_output_tokens"));
    let captured = requests.requests();
    assert_eq!(captured.len(), 2);
    assert_eq!(captured[1].body_json()["tools"], json!([]));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rejects_oversized_pair_summary_input() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let oversized = format!("OVERSIZED_PAIR_SENTINEL {}", "wide ".repeat(20_000));
    let requests = mount_sse_sequence(
        &server,
        vec![sse(vec![
            responses::ev_assistant_message("seed-oversized", &oversized),
            responses::ev_assistant_message("seed-small", "OVERSIZED_SMALL_SENTINEL"),
            responses::ev_completed("seed-resp"),
        ])],
    )
    .await;
    let test = test_codex()
        .with_config(configure_pairwise)
        .build(&server)
        .await?;

    let seed_turn_id = submit_text_turn(&test, "OVERSIZED_USER_SENTINEL").await?;
    wait_for_turn_complete(&test.codex, &seed_turn_id).await?;
    let pairwise_turn_id = submit_text_turn(&test, "OVERSIZED_HOT_SENTINEL").await?;
    let error = wait_for_turn_complete(&test.codex, &pairwise_turn_id)
        .await
        .expect_err("oversized pair-summary input must fail before model request");

    assert!(
        error
            .to_string()
            .contains("Codex ran out of room in the model's context window")
    );
    assert_eq!(requests.requests().len(), 1);

    Ok(())
}
