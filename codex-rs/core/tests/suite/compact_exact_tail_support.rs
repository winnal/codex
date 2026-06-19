pub(super) use anyhow::Result;
pub(super) use anyhow::anyhow;
pub(super) use codex_core::compact::SUMMARIZATION_PROMPT;
pub(super) use codex_core::compact::SUMMARY_PREFIX;
pub(super) use codex_core::config::Config;
pub(super) use codex_features::Feature;
pub(super) use codex_login::CodexAuth;
pub(super) use codex_model_provider_info::ModelProviderInfo;
pub(super) use codex_model_provider_info::built_in_model_providers;
pub(super) use codex_protocol::config_types::AutoCompactTokenLimitScope;
pub(super) use codex_protocol::protocol::EventMsg;
pub(super) use codex_protocol::protocol::Op;
pub(super) use codex_protocol::protocol::RolloutItem;
pub(super) use codex_protocol::protocol::RolloutLine;
pub(super) use core_test_support::responses::ev_assistant_message;
pub(super) use core_test_support::responses::ev_completed;
pub(super) use core_test_support::responses::ev_completed_with_tokens;
pub(super) use core_test_support::responses::ev_function_call;
pub(super) use core_test_support::responses::mount_compact_json_once;
pub(super) use core_test_support::responses::mount_sse_sequence;
pub(super) use core_test_support::responses::sse;
pub(super) use core_test_support::responses::sse_failed;
pub(super) use core_test_support::responses::start_mock_server;
pub(super) use core_test_support::skip_if_no_network;
pub(super) use core_test_support::test_codex::TestCodex;
pub(super) use core_test_support::test_codex::test_codex;
pub(super) use core_test_support::wait_for_event;
pub(super) use core_test_support::wait_for_event_match;
pub(super) use serde_json::Value;
pub(super) use serde_json::json;
pub(super) use std::fs;
pub(super) use std::path::Path;

pub(super) const FIRST_REPLY: &str = "FIRST_REPLY";
pub(super) const AUTO_SUMMARY_TEXT: &str = "AUTO_SUMMARY";
pub(super) const SECOND_LARGE_REPLY: &str = "SECOND_LARGE_REPLY";
pub(super) const FINAL_REPLY: &str = "FINAL_REPLY";
pub(super) const DUMMY_FUNCTION_NAME: &str = "test_tool";
pub(super) const DUMMY_CALL_ID: &str = "call-multi-auto";

pub(super) fn summary_with_prefix(summary: &str) -> String {
    format!("{SUMMARY_PREFIX}\n{summary}")
}

pub(super) fn set_test_compact_prompt(config: &mut Config) {
    config.compact_prompt = Some(SUMMARIZATION_PROMPT.to_string());
}

pub(super) fn ev_completed_with_usage(id: &str, input_tokens: i64, output_tokens: i64) -> Value {
    json!({
        "type": "response.completed",
        "response": {
            "id": id,
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": null,
                "output_tokens": output_tokens,
                "output_tokens_details": null,
                "total_tokens": input_tokens + output_tokens
            }
        }
    })
}

pub(super) fn body_contains_text(body: &str, text: &str) -> bool {
    body.contains(&json_fragment(text))
}

pub(super) fn item_contains_text(item: &Value, expected: &str) -> bool {
    if item
        .get("encrypted_content")
        .and_then(Value::as_str)
        .is_some_and(|text| text.contains(expected))
    {
        return true;
    }
    item.get("content")
        .and_then(Value::as_array)
        .is_some_and(|content| {
            content.iter().any(|span| {
                span.get("text")
                    .and_then(Value::as_str)
                    .is_some_and(|text| text.contains(expected))
            })
        })
}

pub(super) fn input_contains_text(input: &[Value], expected: &str) -> bool {
    input.iter().any(|item| item_contains_text(item, expected))
}

pub(super) fn input_text_match_count(input: &[Value], expected: &str) -> usize {
    input
        .iter()
        .filter(|item| item_contains_text(item, expected))
        .count()
}

pub(super) fn assert_ordered_input_texts(input: &[Value], expected_texts: &[&str]) {
    let mut search_from = 0usize;
    for expected in expected_texts {
        let Some(offset) = input[search_from..]
            .iter()
            .position(|item| item_contains_text(item, expected))
        else {
            panic!("expected text {expected:?} after index {search_from} in input {input:#?}");
        };
        search_from += offset + 1;
    }
}

pub(super) fn input_item_index_by_type_and_call_id(
    input: &[Value],
    item_type: &str,
    call_id: &str,
) -> Option<usize> {
    input.iter().position(|item| {
        item.get("type").and_then(Value::as_str) == Some(item_type)
            && item.get("call_id").and_then(Value::as_str) == Some(call_id)
    })
}

pub(super) fn json_fragment(text: &str) -> String {
    serde_json::to_string(text)
        .expect("serialize text to JSON")
        .trim_matches('"')
        .to_string()
}

pub(super) fn replacement_history_from_rollout(path: &Path) -> Result<Vec<Value>> {
    let rollout_text = fs::read_to_string(path)?;
    let mut replacement_history = None;
    for line in rollout_text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let entry: RolloutLine = serde_json::from_str(line)?;
        if let RolloutItem::Compacted(compacted) = entry.item
            && let Some(items) = compacted.replacement_history
        {
            replacement_history = Some(
                items
                    .into_iter()
                    .map(serde_json::to_value)
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            );
        }
    }
    replacement_history.ok_or_else(|| anyhow!("expected rollout replacement history"))
}

pub(super) fn local_compaction_provider(server: &wiremock::MockServer) -> ModelProviderInfo {
    let mut provider = built_in_model_providers(/*openai_base_url*/ None)["openai"].clone();
    provider.name = "OpenAI-compatible test provider".to_string();
    provider.base_url = Some(format!("{}/v1", server.uri()));
    provider.supports_websockets = false;
    provider
}

pub(super) async fn shutdown_codex(test: &TestCodex) -> Result<()> {
    test.codex.submit(Op::Shutdown).await?;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::ShutdownComplete)
    })
    .await;
    Ok(())
}
