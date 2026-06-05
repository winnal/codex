use super::estimate_response_item_token_count;
use super::truncate_function_output_payload;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::LocalShellAction;
use codex_protocol::models::LocalShellExecAction;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::WebSearchAction;
use codex_utils_output_truncation::TruncationPolicy;
use codex_utils_output_truncation::approx_bytes_for_tokens;
use codex_utils_output_truncation::approx_token_count;
use codex_utils_output_truncation::truncate_text;

pub(super) fn project_rolling_message_item(
    item: ResponseItem,
    item_limit_tokens: i64,
) -> Vec<ResponseItem> {
    match item {
        ResponseItem::Message {
            id,
            role,
            content,
            phase,
        } => split_message_item_to_item_limit(id, role, content, phase, item_limit_tokens),
        _ => vec![item],
    }
}

pub(super) fn project_rolling_prompt_item(
    item: ResponseItem,
    tool_output_limit_tokens: usize,
    item_limit_tokens: i64,
) -> ResponseItem {
    match item {
        ResponseItem::FunctionCall {
            id,
            name,
            namespace,
            arguments,
            call_id,
        } => {
            let arguments = truncate_text_to_item_limit(
                &arguments,
                usize::try_from(item_limit_tokens).unwrap_or(usize::MAX),
                item_limit_tokens,
                |arguments| ResponseItem::FunctionCall {
                    id: id.clone(),
                    name: name.clone(),
                    namespace: namespace.clone(),
                    arguments,
                    call_id: call_id.clone(),
                },
            );
            ResponseItem::FunctionCall {
                id,
                name,
                namespace,
                arguments,
                call_id,
            }
        }
        ResponseItem::FunctionCallOutput { call_id, output } => {
            let output = truncate_tool_output_payload_to_item_limit(
                &output,
                tool_output_limit_tokens,
                item_limit_tokens,
                |output| ResponseItem::FunctionCallOutput {
                    call_id: call_id.clone(),
                    output,
                },
            );
            ResponseItem::FunctionCallOutput { call_id, output }
        }
        ResponseItem::CustomToolCallOutput {
            call_id,
            name,
            output,
        } => {
            let output = truncate_tool_output_payload_to_item_limit(
                &output,
                tool_output_limit_tokens,
                item_limit_tokens,
                |output| ResponseItem::CustomToolCallOutput {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    output,
                },
            );
            ResponseItem::CustomToolCallOutput {
                call_id,
                name,
                output,
            }
        }
        ResponseItem::ToolSearchOutput {
            call_id,
            status,
            execution,
            tools,
        } => {
            let tools = project_tool_search_tools(
                &call_id,
                &status,
                &execution,
                &tools,
                tool_output_limit_tokens,
                item_limit_tokens,
            );
            ResponseItem::ToolSearchOutput {
                call_id,
                status,
                execution,
                tools,
            }
        }
        ResponseItem::CustomToolCall {
            id,
            status,
            call_id,
            name,
            input,
        } => {
            let input = truncate_text_to_item_limit(
                &input,
                usize::try_from(item_limit_tokens).unwrap_or(usize::MAX),
                item_limit_tokens,
                |input| ResponseItem::CustomToolCall {
                    id: id.clone(),
                    status: status.clone(),
                    call_id: call_id.clone(),
                    name: name.clone(),
                    input,
                },
            );
            ResponseItem::CustomToolCall {
                id,
                status,
                call_id,
                name,
                input,
            }
        }
        ResponseItem::ToolSearchCall {
            id,
            call_id,
            status,
            execution,
            arguments,
        } => {
            let arguments = project_serialized_fallback_to_item_limit(
                &arguments,
                item_limit_tokens,
                |arguments| ResponseItem::ToolSearchCall {
                    id: id.clone(),
                    call_id: call_id.clone(),
                    status: status.clone(),
                    execution: execution.clone(),
                    arguments,
                },
                |serialized, budget| {
                    serde_json::json!({
                        "type": "truncated_tool_search_arguments",
                        "arguments_json": truncate_text(
                            serialized,
                            TruncationPolicy::Tokens(budget),
                        ),
                    })
                },
            );
            ResponseItem::ToolSearchCall {
                id,
                call_id,
                status,
                execution,
                arguments,
            }
        }
        ResponseItem::WebSearchCall { id, status, action } => {
            let action = project_serialized_fallback_to_item_limit(
                &action,
                item_limit_tokens,
                |action| ResponseItem::WebSearchCall {
                    id: id.clone(),
                    status: status.clone(),
                    action,
                },
                |serialized, budget| {
                    action.as_ref().map(|action| match action {
                        WebSearchAction::Search { .. } => WebSearchAction::Search {
                            query: Some(truncate_text(
                                serialized,
                                TruncationPolicy::Tokens(budget),
                            )),
                            queries: None,
                        },
                        WebSearchAction::OpenPage { .. } => WebSearchAction::OpenPage {
                            url: Some(truncate_text(serialized, TruncationPolicy::Tokens(budget))),
                        },
                        WebSearchAction::FindInPage { .. } => WebSearchAction::FindInPage {
                            url: None,
                            pattern: Some(truncate_text(
                                serialized,
                                TruncationPolicy::Tokens(budget),
                            )),
                        },
                        WebSearchAction::Other => WebSearchAction::Other,
                    })
                },
            );
            ResponseItem::WebSearchCall { id, status, action }
        }
        ResponseItem::LocalShellCall {
            id,
            call_id,
            status,
            action,
        } => {
            let action = project_serialized_fallback_to_item_limit(
                &action,
                item_limit_tokens,
                |action| ResponseItem::LocalShellCall {
                    id: id.clone(),
                    call_id: call_id.clone(),
                    status: status.clone(),
                    action,
                },
                |serialized, budget| {
                    LocalShellAction::Exec(LocalShellExecAction {
                        command: vec![truncate_text(serialized, TruncationPolicy::Tokens(budget))],
                        timeout_ms: None,
                        working_directory: None,
                        env: None,
                        user: None,
                    })
                },
            );
            ResponseItem::LocalShellCall {
                id,
                call_id,
                status,
                action,
            }
        }
        ResponseItem::ImageGenerationCall {
            id,
            status,
            revised_prompt,
            result,
        } => {
            let make_item = |revised_prompt: Option<String>, result: String| {
                ResponseItem::ImageGenerationCall {
                    id: id.clone(),
                    status: status.clone(),
                    revised_prompt,
                    result,
                }
            };
            let project_revised_prompt = |revised_prompt: Option<String>, result: &str| {
                revised_prompt.map(|prompt| {
                    truncate_text_to_item_limit(
                        &prompt,
                        usize::try_from(item_limit_tokens).unwrap_or(usize::MAX),
                        item_limit_tokens,
                        |revised_prompt| make_item(Some(revised_prompt), result.to_string()),
                    )
                })
            };

            let mut projected_result = result;
            let mut projected_revised_prompt =
                project_revised_prompt(revised_prompt.clone(), &projected_result);
            let projected_item =
                make_item(projected_revised_prompt.clone(), projected_result.clone());
            if estimate_response_item_token_count(&projected_item) > item_limit_tokens {
                projected_result = String::new();
                projected_revised_prompt =
                    project_revised_prompt(revised_prompt, &projected_result);
            }
            let result = truncate_text_to_item_limit(
                &projected_result,
                tool_output_limit_tokens,
                item_limit_tokens,
                |result| make_item(projected_revised_prompt.clone(), result),
            );
            ResponseItem::ImageGenerationCall {
                id: id.clone(),
                status: status.clone(),
                revised_prompt: projected_revised_prompt,
                result,
            }
        }
        ResponseItem::Message { .. }
        | ResponseItem::AgentMessage { .. }
        | ResponseItem::Reasoning { .. }
        | ResponseItem::Compaction { .. }
        | ResponseItem::CompactionTrigger
        | ResponseItem::ContextCompaction { .. }
        | ResponseItem::Other => item,
    }
}

fn split_message_item_to_item_limit(
    id: Option<String>,
    role: String,
    content: Vec<ContentItem>,
    phase: Option<codex_protocol::models::MessagePhase>,
    item_limit_tokens: i64,
) -> Vec<ResponseItem> {
    let make_item = |content| ResponseItem::Message {
        id: id.clone(),
        role: role.clone(),
        content,
        phase: phase.clone(),
    };
    if estimate_response_item_token_count(&make_item(content.clone())) <= item_limit_tokens {
        return vec![make_item(content)];
    }

    let mut items = Vec::new();
    let mut current_content = Vec::new();
    for content_item in content {
        for fragment in
            split_content_item_to_item_limit(content_item, item_limit_tokens, &make_item)
        {
            push_message_fragment_to_item_limit(
                &mut items,
                &mut current_content,
                fragment,
                item_limit_tokens,
                &make_item,
            );
        }
    }
    if !current_content.is_empty() {
        items.push(make_item(current_content));
    }
    if items.is_empty() {
        items.push(make_item(Vec::new()));
    }
    items
}

fn split_content_item_to_item_limit(
    item: ContentItem,
    item_limit_tokens: i64,
    make_item: &impl Fn(Vec<ContentItem>) -> ResponseItem,
) -> Vec<ContentItem> {
    match item {
        ContentItem::InputText { text } => {
            split_text_to_item_limit(&text, item_limit_tokens, |text| {
                estimate_response_item_token_count(&make_item(vec![ContentItem::InputText {
                    text: text.to_string(),
                }]))
            })
            .into_iter()
            .map(|text| ContentItem::InputText { text })
            .collect()
        }
        ContentItem::OutputText { text } => {
            split_text_to_item_limit(&text, item_limit_tokens, |text| {
                estimate_response_item_token_count(&make_item(vec![ContentItem::OutputText {
                    text: text.to_string(),
                }]))
            })
            .into_iter()
            .map(|text| ContentItem::OutputText { text })
            .collect()
        }
        ContentItem::InputImage { .. } => vec![item],
    }
}

fn push_message_fragment_to_item_limit(
    items: &mut Vec<ResponseItem>,
    current_content: &mut Vec<ContentItem>,
    fragment: ContentItem,
    item_limit_tokens: i64,
    make_item: &impl Fn(Vec<ContentItem>) -> ResponseItem,
) {
    let mut candidate = current_content.clone();
    candidate.push(fragment.clone());
    if estimate_response_item_token_count(&make_item(candidate.clone())) <= item_limit_tokens
        || current_content.is_empty()
    {
        *current_content = candidate;
        return;
    }

    items.push(make_item(std::mem::take(current_content)));
    current_content.push(fragment);
}

fn split_text_to_item_limit(
    text: &str,
    item_limit_tokens: i64,
    mut estimate_text_tokens: impl FnMut(&str) -> i64,
) -> Vec<String> {
    if text.is_empty() || estimate_text_tokens(text) <= item_limit_tokens {
        return vec![text.to_string()];
    }

    let chunk_bytes = usize::try_from(item_limit_tokens)
        .map(approx_bytes_for_tokens)
        .unwrap_or(usize::MAX)
        .max(1);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < text.len() {
        let mut end = floor_char_boundary(text, start.saturating_add(chunk_bytes).min(text.len()));
        if end <= start {
            end = next_char_boundary(text, start);
        }
        while estimate_text_tokens(&text[start..end]) > item_limit_tokens && end > start {
            let current_len = end.saturating_sub(start);
            let next_len = current_len.saturating_mul(9).saturating_div(10).max(1);
            let next_end = floor_char_boundary(text, start.saturating_add(next_len));
            if next_end <= start || next_end == end {
                end = next_char_boundary(text, start);
                break;
            }
            end = next_end;
        }
        chunks.push(text[start..end].to_string());
        start = end;
    }
    chunks
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn next_char_boundary(text: &str, index: usize) -> usize {
    text[index..]
        .char_indices()
        .nth(1)
        .map_or(text.len(), |(offset, _)| index + offset)
}

fn project_serialized_fallback_to_item_limit<T>(
    value: &T,
    item_limit_tokens: i64,
    mut make_item: impl FnMut(T) -> ResponseItem,
    mut make_fallback: impl FnMut(&str, usize) -> T,
) -> T
where
    T: Clone + serde::Serialize,
{
    if estimate_response_item_token_count(&make_item(value.clone())) <= item_limit_tokens {
        return value.clone();
    }

    let serialized = serde_json::to_string(value)
        .unwrap_or_else(|err| format!("failed to serialize item for rolling projection: {err}"));
    let mut budget = usize::try_from(item_limit_tokens).unwrap_or(usize::MAX);
    loop {
        let projected = make_fallback(&serialized, budget);
        if estimate_response_item_token_count(&make_item(projected.clone())) <= item_limit_tokens
            || budget <= 1
        {
            return projected;
        }
        budget = shrink_token_budget(budget);
    }
}

fn truncate_tool_output_payload_to_item_limit(
    output: &FunctionCallOutputPayload,
    mut budget: usize,
    item_limit_tokens: i64,
    mut make_item: impl FnMut(FunctionCallOutputPayload) -> ResponseItem,
) -> FunctionCallOutputPayload {
    let mut payload = truncate_function_output_payload(output, TruncationPolicy::Tokens(budget));
    while estimate_response_item_token_count(&make_item(payload.clone())) > item_limit_tokens
        && budget > 1
    {
        budget = shrink_token_budget(budget);
        payload = truncate_function_output_payload(output, TruncationPolicy::Tokens(budget));
    }
    payload
}

fn project_tool_search_tools(
    call_id: &Option<String>,
    status: &str,
    execution: &str,
    tools: &[serde_json::Value],
    mut budget: usize,
    item_limit_tokens: i64,
) -> Vec<serde_json::Value> {
    let serialized = serde_json::to_string(tools).unwrap_or_else(|err| {
        format!("failed to serialize tool_search output for rolling projection: {err}")
    });
    let original_item = ResponseItem::ToolSearchOutput {
        call_id: call_id.clone(),
        status: status.to_string(),
        execution: execution.to_string(),
        tools: tools.to_vec(),
    };
    if approx_token_count(&serialized) <= budget
        && estimate_response_item_token_count(&original_item) <= item_limit_tokens
    {
        return tools.to_vec();
    }

    loop {
        let projected_tools = vec![serde_json::json!({
            "type": "truncated_tool_search_output",
            "original_tool_count": tools.len(),
            "tools_json": truncate_text(&serialized, TruncationPolicy::Tokens(budget)),
        })];
        let projected_item = ResponseItem::ToolSearchOutput {
            call_id: call_id.clone(),
            status: status.to_string(),
            execution: execution.to_string(),
            tools: projected_tools.clone(),
        };
        if estimate_response_item_token_count(&projected_item) <= item_limit_tokens || budget <= 1 {
            return projected_tools;
        }
        budget = shrink_token_budget(budget);
    }
}

fn truncate_text_to_item_limit(
    text: &str,
    mut budget: usize,
    item_limit_tokens: i64,
    mut make_item: impl FnMut(String) -> ResponseItem,
) -> String {
    let mut result = truncate_text(text, TruncationPolicy::Tokens(budget));
    while estimate_response_item_token_count(&make_item(result.clone())) > item_limit_tokens
        && budget > 1
    {
        budget = shrink_token_budget(budget);
        result = truncate_text(text, TruncationPolicy::Tokens(budget));
    }
    result
}

fn shrink_token_budget(budget: usize) -> usize {
    let next_budget = budget.saturating_mul(9).saturating_div(10).max(1);
    if next_budget == budget {
        budget.saturating_sub(1)
    } else {
        next_budget
    }
}

#[cfg(test)]
#[path = "rolling_projection_tests.rs"]
mod tests;
