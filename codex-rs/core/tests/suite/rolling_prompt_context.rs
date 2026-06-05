use super::rolling_prompt_support::wait_for_turn_complete;
use anyhow::Result;
use codex_protocol::config_types::PromptRetentionMode;
use codex_protocol::protocol::AdditionalContextEntry;
use codex_protocol::protocol::AdditionalContextKind;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use core_test_support::responses;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::mount_sse_once_match;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use std::collections::BTreeMap;
use wiremock::matchers::body_string_contains;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_mode_runtime_additional_context_stays_grouped_with_current_user_input()
-> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                responses::ev_assistant_message(
                    "msg-1",
                    &format!("OLD_RUNTIME_ASSISTANT_BODY {}", "old ".repeat(30_000)),
                ),
                responses::ev_completed("resp-1"),
            ]),
            sse(vec![
                responses::ev_assistant_message("msg-2", "second turn done"),
                responses::ev_completed("resp-2"),
            ]),
        ],
    )
    .await;

    let test = test_codex()
        .with_config(|config| {
            config.prompt_retention = PromptRetentionMode::Rolling;
            config.rolling_context_reserve_percent = Some(0);
            config.rolling_context_target_tokens = Some(20_000);
            config.include_permissions_instructions = false;
            config.include_apps_instructions = false;
            config.include_collaboration_mode_instructions = false;
            config.include_skill_instructions = false;
            config.include_environment_context = false;
        })
        .build(&server)
        .await?;

    let first_turn_id = test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "OLD_RUNTIME_USER_CONTEXT".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: BTreeMap::from([(
                "old_runtime_context".to_string(),
                AdditionalContextEntry {
                    value: "OLD_RUNTIME_ADDITIONAL_CONTEXT".to_string(),
                    kind: AdditionalContextKind::Untrusted,
                },
            )]),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&test.codex, &first_turn_id).await?;

    let second_turn_id = test
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "NEW_RUNTIME_USER_CONTEXT".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: BTreeMap::from([(
                "new_runtime_context".to_string(),
                AdditionalContextEntry {
                    value: "NEW_RUNTIME_ADDITIONAL_CONTEXT".to_string(),
                    kind: AdditionalContextKind::Untrusted,
                },
            )]),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&test.codex, &second_turn_id).await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 2);
    assert!(
        captured[1].body_contains_text("NEW_RUNTIME_ADDITIONAL_CONTEXT"),
        "rolling prompt should keep current additional_context with the current user input"
    );
    assert!(
        captured[1].body_contains_text("NEW_RUNTIME_USER_CONTEXT"),
        "rolling prompt should keep the current user input"
    );
    assert!(
        !captured[1].body_contains_text("OLD_RUNTIME_ADDITIONAL_CONTEXT"),
        "rolling prompt should eject stale additional_context under pressure"
    );
    assert!(
        !captured[1].body_contains_text("OLD_RUNTIME_USER_CONTEXT"),
        "rolling prompt should eject the stale user/context group under pressure"
    );
    assert!(
        !captured[1].body_contains_text("OLD_RUNTIME_ASSISTANT_BODY"),
        "rolling prompt should eject oversized stale assistant history under pressure"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_mode_resume_uses_current_prefix_and_newest_suffix() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut initial_builder = test_codex();
    let initial = initial_builder.build(&server).await?;
    let home = initial.home.clone();
    let rollout_path = initial
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");
    let old_body = format!("OLD_RESUME_BODY_SENTINEL {}", "old ".repeat(20_000));

    mount_sse_once(
        &server,
        sse(vec![
            responses::ev_assistant_message("seed-msg", &old_body),
            responses::ev_completed("seed-resp"),
        ]),
    )
    .await;
    let seed_turn_id = initial
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "seed resume history".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&initial.codex, &seed_turn_id).await?;

    let resumed_request = mount_sse_once_match(
        &server,
        body_string_contains("RESUME_NEWEST_BODY_SENTINEL"),
        sse(vec![
            responses::ev_assistant_message("resume-msg", "resumed rolling turn"),
            responses::ev_completed("resume-resp"),
        ]),
    )
    .await;

    let mut resume_builder = test_codex().with_config(|config| {
        config.prompt_retention = PromptRetentionMode::Rolling;
        config.rolling_context_reserve_percent = Some(0);
        config.rolling_context_target_tokens = Some(50_000);
        config.developer_instructions = Some("CURRENT_RESUME_DEV_SENTINEL".to_string());
        config.include_permissions_instructions = false;
        config.include_apps_instructions = false;
        config.include_collaboration_mode_instructions = false;
        config.include_skill_instructions = false;
        config.include_environment_context = false;
    });
    let resumed = resume_builder.resume(&server, home, rollout_path).await?;
    let resumed_turn_id = resumed
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "RESUME_NEWEST_BODY_SENTINEL".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_turn_complete(&resumed.codex, &resumed_turn_id).await?;

    let resumed_request = resumed_request.single_request();
    assert!(
        resumed_request.body_contains_text("CURRENT_RESUME_DEV_SENTINEL"),
        "resumed rolling request should include freshly rendered current developer prefix"
    );
    assert!(
        resumed_request.body_contains_text("RESUME_NEWEST_BODY_SENTINEL"),
        "resumed rolling request should keep newest body suffix"
    );
    assert!(
        !resumed_request.body_contains_text("OLD_RESUME_BODY_SENTINEL"),
        "resumed rolling request should eject old oversized body prefix"
    );

    Ok(())
}
