use crate::compact_remote::should_keep_compacted_history_item;
use crate::context_manager::estimate_response_items_token_count;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TruncationPolicy;
use codex_utils_output_truncation::approx_token_count;
use codex_utils_output_truncation::truncate_text;

// Mirror the current /responses/compact retained-message default while the
// server-side path remains the reference implementation.
pub(crate) const REMOTE_COMPACTION_V2_RETAINED_MESSAGE_TOKEN_BUDGET: usize = 64_000;

pub(crate) fn build_v2_compacted_history(
    prompt_input: &[ResponseItem],
    compaction_output: ResponseItem,
) -> (Vec<ResponseItem>, usize) {
    let (mut retained, retained_image_count) = retained_messages_for_remote_compaction_v2(
        prompt_input,
        REMOTE_COMPACTION_V2_RETAINED_MESSAGE_TOKEN_BUDGET,
    );
    retained.push(compaction_output);
    (retained, retained_image_count)
}

pub(crate) fn retained_messages_for_remote_compaction_v2(
    prompt_input: &[ResponseItem],
    max_tokens: usize,
) -> (Vec<ResponseItem>, usize) {
    retained_messages_for_remote_compaction_v2_with_item_cap(
        prompt_input,
        max_tokens,
        /*max_item_tokens*/ usize::MAX,
    )
}

pub(crate) fn retained_messages_for_remote_compaction_v2_with_item_cap(
    prompt_input: &[ResponseItem],
    max_tokens: usize,
    max_item_tokens: usize,
) -> (Vec<ResponseItem>, usize) {
    let retained = prompt_input
        .iter()
        .filter(|item| is_retained_for_remote_compaction_v2(item))
        .filter(|item| should_keep_compacted_history_item(item))
        .cloned()
        .collect::<Vec<_>>();
    let retained =
        truncate_retained_messages_for_remote_compaction(retained, max_tokens, max_item_tokens);
    let retained_image_count = retained
        .iter()
        .map(retained_input_image_count)
        .sum::<usize>();
    (retained, retained_image_count)
}

fn is_retained_for_remote_compaction_v2(item: &ResponseItem) -> bool {
    let ResponseItem::Message { role, .. } = item else {
        return false;
    };

    matches!(role.as_str(), "user" | "developer" | "system")
}

fn retained_input_image_count(item: &ResponseItem) -> usize {
    let ResponseItem::Message { content, .. } = item else {
        return 0;
    };

    content
        .iter()
        .filter(|item| matches!(item, ContentItem::InputImage { .. }))
        .count()
}

fn truncate_retained_messages_for_remote_compaction(
    items: Vec<ResponseItem>,
    max_tokens: usize,
    max_item_tokens: usize,
) -> Vec<ResponseItem> {
    let mut remaining = max_tokens;
    let mut truncated_reversed = Vec::with_capacity(items.len());
    for item in items.into_iter().rev() {
        if remaining == 0 {
            continue;
        }

        let token_count = message_text_token_count(&item).max(1);
        let item_budget = remaining.min(max_item_tokens);
        if token_count <= item_budget
            && estimate_response_items_token_count(std::slice::from_ref(&item))
                <= i64::try_from(max_item_tokens).unwrap_or(i64::MAX)
        {
            truncated_reversed.push(item);
            remaining = remaining.saturating_sub(token_count);
        } else if max_item_tokens == usize::MAX {
            if let Some(truncated_item) =
                truncate_message_text_to_token_budget(item, /*max_tokens*/ remaining)
            {
                truncated_reversed.push(truncated_item);
                remaining = 0;
            }
        } else if let Some(truncated_item) =
            truncate_message_to_item_token_budget(item, /*max_tokens*/ item_budget)
        {
            let token_count = message_text_token_count(&truncated_item).max(1);
            truncated_reversed.push(truncated_item);
            remaining = remaining.saturating_sub(token_count);
        }
    }
    truncated_reversed.reverse();
    truncated_reversed
}

fn message_text_token_count(item: &ResponseItem) -> usize {
    let ResponseItem::Message { content, .. } = item else {
        return 0;
    };

    content
        .iter()
        .map(|item| match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                approx_token_count(text)
            }
            ContentItem::InputImage { .. } => 0,
        })
        .sum()
}

fn truncate_message_text_to_token_budget(
    item: ResponseItem,
    max_tokens: usize,
) -> Option<ResponseItem> {
    let ResponseItem::Message {
        id,
        role,
        content,
        phase,
        metadata,
    } = item
    else {
        return Some(item);
    };

    let mut remaining = max_tokens;
    let mut truncated_content = Vec::with_capacity(content.len());
    for mut content_item in content {
        match &mut content_item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                if remaining == 0 {
                    continue;
                }

                let token_count = approx_token_count(text);
                if token_count <= remaining {
                    remaining = remaining.saturating_sub(token_count);
                } else {
                    *text = truncate_text(text, TruncationPolicy::Tokens(remaining));
                    remaining = 0;
                }
                if !text.is_empty() {
                    truncated_content.push(content_item);
                }
            }
            ContentItem::InputImage { .. } => truncated_content.push(content_item),
        }
    }

    if truncated_content.is_empty() {
        return None;
    }

    Some(ResponseItem::Message {
        id,
        role,
        content: truncated_content,
        phase,
        metadata,
    })
}

fn truncate_message_to_item_token_budget(
    item: ResponseItem,
    max_tokens: usize,
) -> Option<ResponseItem> {
    let mut budget = max_tokens;
    loop {
        let truncated = truncate_message_text_to_token_budget(item.clone(), budget)?;
        let item_tokens = estimate_response_items_token_count(std::slice::from_ref(&truncated));
        if item_tokens <= i64::try_from(max_tokens).unwrap_or(i64::MAX) {
            return Some(truncated);
        }
        if budget <= 1 {
            return None;
        }
        budget = budget.saturating_mul(3).saturating_div(4).max(1);
    }
}

#[cfg(test)]
#[path = "compact_remote_v2_retention_tests.rs"]
mod tests;
