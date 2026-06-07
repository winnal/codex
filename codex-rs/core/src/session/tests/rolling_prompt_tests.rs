use super::*;
use codex_protocol::config_types::RollingCompactionMode;
use codex_protocol::error::CodexErr;
use pretty_assertions::assert_eq;

#[tokio::test]
async fn rolling_invariant_context_has_no_side_effects() {
    let (session, turn_context, rx) = make_session_and_context_with_rx().await;
    let mut turn_context = Arc::into_inner(turn_context).expect("sole thread settings owner");
    let mut outcome = SkillLoadOutcome::default();
    outcome.skills = vec![
        SkillMetadata {
            name: "admin-skill".to_string(),
            description: "desc".to_string(),
            short_description: None,
            interface: None,
            dependencies: None,
            policy: None,
            path_to_skills_md: test_path_buf("/tmp/admin-skill/SKILL.md").abs(),
            scope: SkillScope::Admin,
            plugin_id: None,
        },
        SkillMetadata {
            name: "repo-skill".to_string(),
            description: "desc".to_string(),
            short_description: None,
            interface: None,
            dependencies: None,
            policy: None,
            path_to_skills_md: test_path_buf("/tmp/repo-skill/SKILL.md").abs(),
            scope: SkillScope::Repo,
            plugin_id: None,
        },
    ];
    turn_context.model_info.context_window = Some(100);
    turn_context.turn_skills = TurnSkillsContext::new(Arc::new(outcome));

    let before = session.clone_history().await.into_raw_items();
    let _ = session.build_rolling_invariant_context(&turn_context).await;
    let _ = session.build_rolling_invariant_context(&turn_context).await;

    assert_eq!(session.clone_history().await.into_raw_items(), before);
    assert!(
        rx.try_recv().is_err(),
        "rolling invariant prefix rendering must not emit thread-start skill warnings"
    );
}

#[tokio::test]
async fn rolling_sampling_prompt_uses_current_prefix_and_newest_suffix() -> anyhow::Result<()> {
    let session = make_session_with_config(|config| {
        config.prompt_retention = PromptRetentionMode::Rolling;
        config.rolling_context_reserve_percent = Some(0);
        config.rolling_context_target_tokens = Some(20_000);
        config.developer_instructions = Some("CURRENT_DEV_SENTINEL".to_string());
        config.include_permissions_instructions = false;
        config.include_apps_instructions = false;
        config.include_collaboration_mode_instructions = false;
        config.include_skill_instructions = false;
        config.include_environment_context = false;
    })
    .await?;
    let turn_context = session.new_default_turn().await;
    let history_items = vec![
        developer_message("<permissions instructions>\nSTALE_DEV_SENTINEL"),
        user_message("<environment_context>\nSTALE_ENV_SENTINEL\n</environment_context>"),
        user_message(&"OLD_BODY_SENTINEL ".repeat(8_000)),
    ];
    session
        .record_conversation_items(&turn_context, &history_items)
        .await;
    let rolling_invariant_items = session
        .record_context_updates_and_set_reference_context_item(&turn_context)
        .await;
    session
        .record_conversation_items(&turn_context, &[user_message("NEWEST_BODY_SENTINEL")])
        .await;

    let base_instructions = session.get_base_instructions().await;
    let prompt = session
        .build_sampling_prompt_input(
            &turn_context,
            &base_instructions,
            &rolling_invariant_items,
            /*rolling_target_scale_percent*/ None,
        )
        .await?
        .into_input();

    let developer_texts = developer_input_texts(&prompt);
    assert!(
        developer_texts
            .iter()
            .any(|text| text.contains("CURRENT_DEV_SENTINEL")),
        "rolling prompt should include the freshly rendered current developer prefix: {developer_texts:?}"
    );
    assert!(
        developer_texts
            .iter()
            .all(|text| !text.contains("STALE_DEV_SENTINEL")),
        "rolling prompt must not retain stale historical developer prefix: {developer_texts:?}"
    );

    let user_texts = user_input_texts(&prompt);
    assert!(
        user_texts
            .iter()
            .any(|text| text.contains("NEWEST_BODY_SENTINEL")),
        "rolling prompt should keep newest body suffix: {user_texts:?}"
    );
    assert!(
        user_texts
            .iter()
            .all(|text| !text.contains("OLD_BODY_SENTINEL")),
        "rolling prompt should eject old oversized body prefix: {user_texts:?}"
    );
    assert!(
        user_texts
            .iter()
            .all(|text| !text.contains("STALE_ENV_SENTINEL")),
        "rolling prompt must filter stale contextual user prefix: {user_texts:?}"
    );

    Ok(())
}

#[tokio::test]
async fn prompt_debug_uses_rolling_retention_for_successful_prompt() -> anyhow::Result<()> {
    let session = make_session_with_config(|config| {
        config.prompt_retention = PromptRetentionMode::Rolling;
        config.rolling_context_reserve_percent = Some(0);
        config.rolling_context_target_tokens = Some(20_000);
        config.developer_instructions = Some("CURRENT_DEBUG_DEV_SENTINEL".to_string());
        config.include_permissions_instructions = false;
        config.include_apps_instructions = false;
        config.include_collaboration_mode_instructions = false;
        config.include_skill_instructions = false;
        config.include_environment_context = false;
    })
    .await?;
    let turn_context = session.new_default_turn().await;
    let old_body = user_message(&"OLD_DEBUG_BODY_SENTINEL ".repeat(20_000));
    session
        .record_conversation_items(&turn_context, std::slice::from_ref(&old_body))
        .await;

    let prompt = crate::prompt_debug::build_prompt_input_from_session(
        session.as_ref(),
        vec![UserInput::Text {
            text: "CURRENT_DEBUG_USER_SENTINEL".to_string(),
            text_elements: Vec::new(),
        }],
    )
    .await?;

    let user_texts = user_input_texts(&prompt);
    assert!(
        user_texts
            .iter()
            .any(|text| text.contains("CURRENT_DEBUG_USER_SENTINEL")),
        "rolling prompt-debug path should keep the requested debug turn: {user_texts:?}"
    );
    assert!(
        user_texts
            .iter()
            .all(|text| !text.contains("OLD_DEBUG_BODY_SENTINEL")),
        "rolling prompt-debug path should drop old oversized body history: {user_texts:?}"
    );
    let developer_texts = developer_input_texts(&prompt);
    assert!(
        developer_texts
            .iter()
            .any(|text| text.contains("CURRENT_DEBUG_DEV_SENTINEL")),
        "rolling prompt-debug path should include the current developer prefix: {developer_texts:?}"
    );

    Ok(())
}

#[tokio::test]
async fn prompt_debug_rejects_pairwise_rolling_compaction() -> anyhow::Result<()> {
    let session = make_session_with_config(|config| {
        config.prompt_retention = PromptRetentionMode::Rolling;
        config.rolling_compaction = RollingCompactionMode::Pairwise;
        config.rolling_context_reserve_percent = Some(0);
        config.rolling_context_target_tokens = Some(20_000);
        config.include_permissions_instructions = false;
        config.include_apps_instructions = false;
        config.include_collaboration_mode_instructions = false;
        config.include_skill_instructions = false;
        config.include_environment_context = false;
    })
    .await?;

    let err = crate::prompt_debug::build_prompt_input_from_session(
        session.as_ref(),
        vec![UserInput::Text {
            text: "CURRENT_DEBUG_USER_SENTINEL".to_string(),
            text_elements: Vec::new(),
        }],
    )
    .await
    .expect_err("pairwise prompt-debug must not emit synthetic summaries");

    match err {
        CodexErr::UnsupportedOperation(message) => {
            assert!(message.contains("pairwise rolling compaction"));
        }
        other => panic!("expected UnsupportedOperation, got {other:?}"),
    }

    Ok(())
}

#[tokio::test]
async fn rolling_sampling_prompt_preserves_current_dynamic_context_updates() -> anyhow::Result<()> {
    let session = make_session_with_config(|config| {
        config.prompt_retention = PromptRetentionMode::Rolling;
        config.rolling_context_reserve_percent = Some(0);
        config.rolling_context_target_tokens = Some(20_000);
        config.include_permissions_instructions = false;
        config.include_apps_instructions = false;
        config.include_collaboration_mode_instructions = false;
        config.include_skill_instructions = false;
        config.include_environment_context = false;
    })
    .await?;
    let previous_context = session.new_default_turn().await;
    let next_model = if previous_context.model_info.slug == "gpt-5.4" {
        "gpt-5.2"
    } else {
        "gpt-5.4"
    };
    let mut turn_context = previous_context
        .with_model(next_model.to_string(), &session.services.models_manager)
        .await;
    turn_context.realtime_active = true;
    {
        let mut state = session.state.lock().await;
        state.set_reference_context_item(Some(previous_context.to_turn_context_item()));
    }
    session
        .set_previous_turn_settings(Some(PreviousTurnSettings {
            model: previous_context.model_info.slug.clone(),
            realtime_active: Some(previous_context.realtime_active),
        }))
        .await;

    let rolling_invariant_items = session
        .record_context_updates_and_set_reference_context_item(&turn_context)
        .await;
    session
        .set_previous_turn_settings(Some(PreviousTurnSettings {
            model: turn_context.model_info.slug.clone(),
            realtime_active: Some(turn_context.realtime_active),
        }))
        .await;
    let body_item = user_message("BODY_SENTINEL");
    session
        .record_conversation_items(&turn_context, std::slice::from_ref(&body_item))
        .await;

    let base_instructions = session.get_base_instructions().await;
    let prompt = session
        .build_sampling_prompt_input(
            &turn_context,
            &base_instructions,
            &rolling_invariant_items,
            /*rolling_target_scale_percent*/ None,
        )
        .await?
        .into_input();
    let developer_texts = developer_input_texts(&prompt);

    assert_eq!(
        developer_texts
            .iter()
            .filter(|text| text.trim_start().starts_with("<model_switch>"))
            .count(),
        1,
        "expected current model switch update exactly once, got {developer_texts:?}"
    );
    assert_eq!(
        developer_texts
            .iter()
            .filter(|text| text.trim_start().starts_with("<realtime_conversation>"))
            .count(),
        1,
        "expected current realtime update exactly once, got {developer_texts:?}"
    );
    assert!(
        user_input_texts(&prompt)
            .iter()
            .any(|text| text.contains("BODY_SENTINEL")),
        "expected newest rolling body item to remain in prompt"
    );

    Ok(())
}
