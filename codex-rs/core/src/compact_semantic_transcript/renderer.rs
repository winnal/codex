use std::collections::HashMap;

use crate::compact_exact_tail::ExactTailError;
use crate::compact_exact_tail::ExactTailFailReason;
use crate::context_manager::estimate_response_items_token_count;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::LocalShellAction;
use codex_protocol::models::MessagePhase;
use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::WebSearchAction;
use codex_protocol::models::plaintext_agent_message_content;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SemanticTranscript {
    pub(crate) items: Vec<ResponseItem>,
    pub(crate) estimated_tokens: i64,
    pub(crate) tool_observation_count: usize,
}

#[derive(Clone, Debug)]
struct PendingToolCall {
    label: String,
    details: Vec<String>,
}

pub(crate) fn render_semantic_transcript(
    cold_history: &[ResponseItem],
    max_model_visible_item_tokens: i64,
) -> Result<SemanticTranscript, ExactTailError> {
    let max_model_visible_item_tokens = max_model_visible_item_tokens.max(1);
    let mut items = Vec::new();
    let mut pending_calls = HashMap::<String, PendingToolCall>::new();
    let mut pending_call_order = Vec::<String>::new();
    let mut tool_observation_count = 0usize;

    for item in cold_history {
        match item {
            ResponseItem::Message {
                role,
                content,
                phase,
                ..
            } => {
                if let Some(text) = content_items_to_text(content) {
                    items.push(bounded_message(
                        role,
                        text,
                        phase.clone(),
                        max_model_visible_item_tokens,
                    )?);
                }
            }
            ResponseItem::AgentMessage {
                author,
                recipient,
                content,
                ..
            } => {
                if let Some(text) = plaintext_agent_message_content(content) {
                    items.push(bounded_message(
                        "assistant",
                        format!("Agent message from {author} to {recipient}:\n{text}"),
                        None,
                        max_model_visible_item_tokens,
                    )?);
                }
            }
            ResponseItem::Reasoning { summary, .. } => {
                let text = reasoning_summary_text(summary);
                if !text.is_empty() {
                    items.push(bounded_message(
                        "assistant",
                        format!("Reasoning summary:\n{text}"),
                        None,
                        max_model_visible_item_tokens,
                    )?);
                }
            }
            ResponseItem::FunctionCall {
                name,
                namespace,
                arguments,
                call_id,
                ..
            } => {
                if !pending_calls.contains_key(call_id) {
                    pending_call_order.push(call_id.clone());
                }
                pending_calls.insert(
                    call_id.clone(),
                    PendingToolCall {
                        label: namespace
                            .as_ref()
                            .map(|namespace| format!("{namespace}.{name}"))
                            .unwrap_or_else(|| name.clone()),
                        details: vec![format!("arguments: {arguments}")],
                    },
                );
            }
            ResponseItem::CustomToolCall {
                call_id,
                name,
                input,
                status,
                ..
            } => {
                let mut details = vec![format!("input: {input}")];
                if let Some(status) = status {
                    details.push(format!("status: {status}"));
                }
                if !pending_calls.contains_key(call_id) {
                    pending_call_order.push(call_id.clone());
                }
                pending_calls.insert(
                    call_id.clone(),
                    PendingToolCall {
                        label: name.clone(),
                        details,
                    },
                );
            }
            ResponseItem::ToolSearchCall {
                call_id,
                status,
                execution,
                arguments,
                ..
            } => {
                if let Some(call_id) = call_id {
                    if !pending_calls.contains_key(call_id) {
                        pending_call_order.push(call_id.clone());
                    }
                    let mut details = vec![
                        format!("execution: {execution}"),
                        format!("arguments: {arguments}"),
                    ];
                    if let Some(status) = status {
                        details.push(format!("status: {status}"));
                    }
                    pending_calls.insert(
                        call_id.clone(),
                        PendingToolCall {
                            label: "tool_search".to_string(),
                            details,
                        },
                    );
                }
            }
            ResponseItem::LocalShellCall { status, action, .. } => {
                tool_observation_count += 1;
                items.push(bounded_message(
                    "assistant",
                    local_shell_observation_text(status, action),
                    None,
                    max_model_visible_item_tokens,
                )?);
            }
            ResponseItem::FunctionCallOutput {
                call_id, output, ..
            } => {
                tool_observation_count += 1;
                let call = pending_calls.remove(call_id);
                items.push(tool_observation_message(
                    call.as_ref(),
                    output.success,
                    output_text(output),
                    max_model_visible_item_tokens,
                )?);
            }
            ResponseItem::CustomToolCallOutput {
                call_id,
                name,
                output,
                ..
            } => {
                tool_observation_count += 1;
                let fallback;
                let call = match pending_calls.remove(call_id) {
                    Some(call) => Some(call),
                    None => {
                        fallback = PendingToolCall {
                            label: name.clone().unwrap_or_else(|| "custom_tool".to_string()),
                            details: Vec::new(),
                        };
                        Some(fallback)
                    }
                };
                items.push(tool_observation_message(
                    call.as_ref(),
                    output.success,
                    output_text(output),
                    max_model_visible_item_tokens,
                )?);
            }
            ResponseItem::ToolSearchOutput { call_id, .. } => {
                tool_observation_count += 1;
                let call = call_id
                    .as_ref()
                    .and_then(|call_id| pending_calls.remove(call_id));
                items.push(tool_search_observation_message(
                    call.as_ref(),
                    item,
                    max_model_visible_item_tokens,
                )?);
            }
            ResponseItem::WebSearchCall { status, action, .. } => {
                tool_observation_count += 1;
                items.push(bounded_message(
                    "assistant",
                    web_search_observation_text(status.as_deref(), action.as_ref()),
                    None,
                    max_model_visible_item_tokens,
                )?);
            }
            ResponseItem::ImageGenerationCall {
                status,
                revised_prompt,
                result,
                ..
            } => {
                tool_observation_count += 1;
                let mut text = format!("Image generation observation\nstatus: {status}");
                if let Some(revised_prompt) = revised_prompt {
                    text.push_str("\nrevised_prompt: ");
                    text.push_str(revised_prompt);
                }
                if !result.is_empty() {
                    text.push_str("\nresult: ");
                    text.push_str(result);
                }
                items.push(bounded_message(
                    "assistant",
                    text,
                    None,
                    max_model_visible_item_tokens,
                )?);
            }
            ResponseItem::Compaction {
                encrypted_content, ..
            } => {
                if !encrypted_content.trim().is_empty() {
                    items.push(bounded_message(
                        "assistant",
                        format!("Prior compaction summary:\n{encrypted_content}"),
                        None,
                        max_model_visible_item_tokens,
                    )?);
                }
            }
            ResponseItem::ContextCompaction {
                encrypted_content, ..
            } => {
                if let Some(content) = encrypted_content
                    && !content.trim().is_empty()
                {
                    items.push(bounded_message(
                        "assistant",
                        format!("Prior context compaction summary:\n{content}"),
                        None,
                        max_model_visible_item_tokens,
                    )?);
                }
            }
            ResponseItem::CompactionTrigger { .. } | ResponseItem::Other => {}
        }
    }

    for call_id in pending_call_order {
        if let Some(call) = pending_calls.remove(&call_id) {
            tool_observation_count += 1;
            items.push(bounded_message(
                "assistant",
                tool_call_text(&call, None, None),
                None,
                max_model_visible_item_tokens,
            )?);
        }
    }

    Ok(SemanticTranscript {
        estimated_tokens: estimate_response_items_token_count(&items),
        items,
        tool_observation_count,
    })
}

fn content_items_to_text(content: &[ContentItem]) -> Option<String> {
    let mut parts = Vec::new();
    for item in content {
        match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                if !text.trim().is_empty() {
                    parts.push(text.as_str());
                }
            }
            ContentItem::InputImage { .. } => {}
        }
    }
    (!parts.is_empty()).then(|| parts.join("\n"))
}

fn reasoning_summary_text(summary: &[ReasoningItemReasoningSummary]) -> String {
    summary
        .iter()
        .map(|item| match item {
            ReasoningItemReasoningSummary::SummaryText { text } => text.as_str(),
        })
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn bounded_message(
    role: &str,
    text: String,
    phase: Option<MessagePhase>,
    max_model_visible_item_tokens: i64,
) -> Result<ResponseItem, ExactTailError> {
    let max_tokens = usize::try_from(max_model_visible_item_tokens.saturating_sub(256).max(1))
        .unwrap_or(usize::MAX);
    let mut budget = max_tokens;
    loop {
        let truncated = truncate_semantic_text(&text, budget);
        let item = ResponseItem::Message {
            id: None,
            role: role.to_string(),
            content: vec![if role == "assistant" {
                ContentItem::OutputText { text: truncated }
            } else {
                ContentItem::InputText { text: truncated }
            }],
            phase: phase.clone(),
            metadata: None,
        };
        let item_tokens = estimate_response_items_token_count(std::slice::from_ref(&item));
        if item_tokens <= max_model_visible_item_tokens {
            return Ok(item);
        }
        if budget <= 1 {
            return Err(ExactTailError::new(
                ExactTailFailReason::ModelVisibleItemTooLarge,
                format!(
                    "{}: semantic transcript item estimates to {item_tokens} tokens, exceeding per-item cap {max_model_visible_item_tokens}",
                    ExactTailFailReason::ModelVisibleItemTooLarge.as_str()
                ),
            ));
        }
        budget = budget.saturating_mul(3).saturating_div(4).max(1);
    }
}

fn truncate_semantic_text(text: &str, max_tokens: usize) -> String {
    codex_utils_output_truncation::truncate_text(
        text,
        codex_utils_output_truncation::TruncationPolicy::Tokens(max_tokens),
    )
}

fn tool_observation_message(
    call: Option<&PendingToolCall>,
    success: Option<bool>,
    output: Option<String>,
    max_model_visible_item_tokens: i64,
) -> Result<ResponseItem, ExactTailError> {
    let fallback;
    let call = match call {
        Some(call) => call,
        None => {
            fallback = PendingToolCall {
                label: "tool".to_string(),
                details: Vec::new(),
            };
            &fallback
        }
    };
    bounded_message(
        "assistant",
        tool_call_text(call, success, output.as_deref()),
        None,
        max_model_visible_item_tokens,
    )
}

fn tool_search_observation_message(
    call: Option<&PendingToolCall>,
    item: &ResponseItem,
    max_model_visible_item_tokens: i64,
) -> Result<ResponseItem, ExactTailError> {
    let ResponseItem::ToolSearchOutput {
        status,
        execution,
        tools,
        ..
    } = item
    else {
        unreachable!("tool search observation should only render tool search outputs");
    };
    let mut output = format!("status: {status}\nexecution: {execution}");
    if !tools.is_empty() {
        output.push_str("\ntools: ");
        output
            .push_str(&serde_json::to_string(tools).unwrap_or_else(|_| "<unserializable>".into()));
    }
    let fallback;
    let call = match call {
        Some(call) => call,
        None => {
            fallback = PendingToolCall {
                label: "tool_search".to_string(),
                details: Vec::new(),
            };
            &fallback
        }
    };
    bounded_message(
        "assistant",
        tool_call_text(call, None, Some(&output)),
        None,
        max_model_visible_item_tokens,
    )
}

fn tool_call_text(call: &PendingToolCall, success: Option<bool>, output: Option<&str>) -> String {
    let mut text = format!("Tool observation\nname: {}", call.label);
    for detail in &call.details {
        text.push('\n');
        text.push_str(detail);
    }
    if let Some(success) = success {
        text.push_str("\nsuccess: ");
        text.push_str(if success { "true" } else { "false" });
    }
    if let Some(output) = output
        && !output.trim().is_empty()
    {
        text.push_str("\noutput:\n");
        text.push_str(output);
    }
    text
}

fn output_text(output: &FunctionCallOutputPayload) -> Option<String> {
    output.body.to_text().filter(|text| !text.trim().is_empty())
}

fn local_shell_observation_text(
    status: &codex_protocol::models::LocalShellStatus,
    action: &LocalShellAction,
) -> String {
    let mut text = format!("Local shell observation\nstatus: {status:?}");
    match action {
        LocalShellAction::Exec(exec) => {
            text.push_str("\ncommand: ");
            text.push_str(&exec.command.join(" "));
            if let Some(working_directory) = &exec.working_directory {
                text.push_str("\nworkdir: ");
                text.push_str(working_directory);
            }
            if let Some(timeout_ms) = exec.timeout_ms {
                text.push_str("\ntimeout_ms: ");
                text.push_str(&timeout_ms.to_string());
            }
        }
    }
    text
}

fn web_search_observation_text(status: Option<&str>, action: Option<&WebSearchAction>) -> String {
    let mut text = "Web search observation".to_string();
    if let Some(status) = status {
        text.push_str("\nstatus: ");
        text.push_str(status);
    }
    match action {
        Some(WebSearchAction::Search { query, queries }) => {
            if let Some(query) = query {
                text.push_str("\nquery: ");
                text.push_str(query);
            }
            if let Some(queries) = queries
                && !queries.is_empty()
            {
                text.push_str("\nqueries: ");
                text.push_str(&queries.join(" | "));
            }
        }
        Some(WebSearchAction::OpenPage { url }) => {
            if let Some(url) = url {
                text.push_str("\nurl: ");
                text.push_str(url);
            }
        }
        Some(WebSearchAction::FindInPage { url, pattern }) => {
            if let Some(url) = url {
                text.push_str("\nurl: ");
                text.push_str(url);
            }
            if let Some(pattern) = pattern {
                text.push_str("\npattern: ");
                text.push_str(pattern);
            }
        }
        Some(WebSearchAction::Other) | None => {}
    }
    text
}
