use super::*;
use crate::ModelsManagerConfig;
use pretty_assertions::assert_eq;

#[test]
fn reasoning_summaries_override_true_enables_support() {
    let model = model_info_from_slug("unknown-model");
    let config = ModelsManagerConfig {
        model_supports_reasoning_summaries: Some(true),
        ..Default::default()
    };

    let updated = with_config_overrides(model.clone(), &config);
    let mut expected = model;
    expected.supports_reasoning_summaries = true;

    assert_eq!(updated, expected);
}

#[test]
fn reasoning_summaries_override_false_does_not_disable_support() {
    let mut model = model_info_from_slug("unknown-model");
    model.supports_reasoning_summaries = true;
    let config = ModelsManagerConfig {
        model_supports_reasoning_summaries: Some(false),
        ..Default::default()
    };

    let updated = with_config_overrides(model.clone(), &config);

    assert_eq!(updated, model);
}

#[test]
fn reasoning_summaries_override_false_is_noop_when_model_is_false() {
    let model = model_info_from_slug("unknown-model");
    let config = ModelsManagerConfig {
        model_supports_reasoning_summaries: Some(false),
        ..Default::default()
    };

    let updated = with_config_overrides(model.clone(), &config);

    assert_eq!(updated, model);
}

#[test]
fn model_context_window_override_clamps_to_max_context_window() {
    let mut model = model_info_from_slug("unknown-model");
    model.context_window = Some(273_000);
    model.max_context_window = Some(400_000);
    let config = ModelsManagerConfig {
        model_context_window: Some(500_000),
        ..Default::default()
    };

    let updated = with_config_overrides(model.clone(), &config);
    let mut expected = model;
    expected.context_window = Some(400_000);

    assert_eq!(updated, expected);
}

#[test]
fn model_context_window_uses_model_value_without_override() {
    let mut model = model_info_from_slug("unknown-model");
    model.context_window = Some(273_000);
    model.max_context_window = Some(400_000);
    let config = ModelsManagerConfig::default();

    let updated = with_config_overrides(model.clone(), &config);

    assert_eq!(updated, model);
}

#[test]
fn tool_output_cap_limits_larger_token_override() {
    let model = model_info_from_slug("unknown-model");
    let config = ModelsManagerConfig {
        tool_output_token_limit: Some(20_000),
        max_tool_output_token_limit: Some(8_000),
        ..Default::default()
    };

    let updated = with_config_overrides(model, &config);

    assert_eq!(
        updated.truncation_policy,
        TruncationPolicyConfig::bytes(/*limit*/ 32_000)
    );
}

#[test]
fn tool_output_cap_preserves_smaller_token_override() {
    let model = model_info_from_slug("unknown-model");
    let config = ModelsManagerConfig {
        tool_output_token_limit: Some(4_000),
        max_tool_output_token_limit: Some(8_000),
        ..Default::default()
    };

    let updated = with_config_overrides(model, &config);

    assert_eq!(
        updated.truncation_policy,
        TruncationPolicyConfig::bytes(/*limit*/ 16_000)
    );
}

#[test]
fn tool_output_cap_preserves_token_mode_when_model_uses_token_mode() {
    let mut model = model_info_from_slug("unknown-model");
    model.truncation_policy = TruncationPolicyConfig::tokens(/*limit*/ 10_000);
    let config = ModelsManagerConfig {
        tool_output_token_limit: Some(20_000),
        max_tool_output_token_limit: Some(8_000),
        ..Default::default()
    };

    let updated = with_config_overrides(model, &config);

    assert_eq!(
        updated.truncation_policy,
        TruncationPolicyConfig::tokens(/*limit*/ 8_000)
    );
}
