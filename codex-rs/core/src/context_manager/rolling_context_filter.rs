use super::updates::is_rolling_invariant_developer_content;
use crate::context::ContextualUserFragment;
use crate::context::EnvironmentContext;
use crate::context::ExtensionContextualUserFragment;
use crate::context::RollingInvariantDeveloperContext;
use crate::context::UserInstructions;
use crate::context::is_contextual_user_fragment;
use crate::event_mapping::is_contextual_dev_message_content;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;

pub(super) fn is_turn_context_group_start(
    item: &ResponseItem,
    invariant_prefix: &[ResponseItem],
) -> bool {
    if is_historical_context_item(item, invariant_prefix) {
        return false;
    }

    match item {
        ResponseItem::Message { role, .. } if role == "developer" => true,
        ResponseItem::Message { role, content, .. } if role == "user" => {
            is_turn_scoped_contextual_user_content(content)
        }
        _ => false,
    }
}

pub(super) fn is_turn_context_group_anchor(
    item: &ResponseItem,
    invariant_prefix: &[ResponseItem],
) -> bool {
    if is_historical_context_item(item, invariant_prefix) {
        return false;
    }

    matches!(
        item,
        ResponseItem::Message { role, .. } if role == "user" || role == "assistant"
    )
}

pub(super) fn is_historical_context_item_at(
    raw_items: &[ResponseItem],
    index: usize,
    invariant_prefix: &[ResponseItem],
) -> bool {
    is_historical_context_item(&raw_items[index], invariant_prefix)
        || is_legacy_developer_preamble_context_item(raw_items, index, invariant_prefix)
}

fn is_turn_scoped_contextual_user_content(content: &[ContentItem]) -> bool {
    !is_static_contextual_user_content(content)
        && !content.is_empty()
        && content.iter().all(is_contextual_user_fragment)
}

fn is_historical_context_item(item: &ResponseItem, invariant_prefix: &[ResponseItem]) -> bool {
    if invariant_prefix.contains(item)
        && !matches!(item, ResponseItem::Message { role, .. } if role == "user")
    {
        return true;
    }

    match item {
        ResponseItem::Message { role, content, .. } if role == "developer" => {
            is_rolling_invariant_developer_content(content)
                || is_contextual_dev_message_content(content)
                || is_current_invariant_developer_content(content, invariant_prefix)
        }
        ResponseItem::Message { role, content, .. } if role == "user" => {
            is_static_contextual_user_content(content)
        }
        _ => false,
    }
}

fn is_legacy_developer_preamble_context_item(
    raw_items: &[ResponseItem],
    index: usize,
    invariant_prefix: &[ResponseItem],
) -> bool {
    if !matches!(
        raw_items.get(index),
        Some(ResponseItem::Message { role, .. }) if role == "developer"
    ) {
        return false;
    }

    if !invariant_prefix
        .iter()
        .any(|item| matches!(item, ResponseItem::Message { role, .. } if role == "developer"))
    {
        return false;
    }

    !raw_items[..index].iter().any(is_conversation_anchor_item)
        && !raw_items[..index]
            .iter()
            .any(|item| is_historical_context_item(item, invariant_prefix))
}

fn is_conversation_anchor_item(item: &ResponseItem) -> bool {
    match item {
        ResponseItem::Message { role, content, .. } if role == "user" => {
            !is_static_contextual_user_content(content)
        }
        ResponseItem::Message { role, .. } if role == "assistant" => true,
        _ => false,
    }
}

fn is_static_contextual_user_content(content: &[ContentItem]) -> bool {
    !content.is_empty()
        && content.iter().all(|item| {
            let ContentItem::InputText { text } = item else {
                return false;
            };
            EnvironmentContext::matches_text(text)
                || ExtensionContextualUserFragment::matches_text(text)
                || UserInstructions::matches_text(text)
                || is_legacy_user_instructions_context(text)
        })
}

fn is_legacy_user_instructions_context(text: &str) -> bool {
    let start_marker = codex_protocol::protocol::USER_INSTRUCTIONS_OPEN_TAG;
    let end_marker = codex_protocol::protocol::USER_INSTRUCTIONS_CLOSE_TAG;
    let trimmed = text.trim_start();
    let starts_with_marker = trimmed
        .get(..start_marker.len())
        .is_some_and(|candidate| candidate.eq_ignore_ascii_case(start_marker));
    let trimmed = trimmed.trim_end();
    let ends_with_marker = trimmed
        .get(trimmed.len().saturating_sub(end_marker.len())..)
        .is_some_and(|candidate| candidate.eq_ignore_ascii_case(end_marker));
    starts_with_marker && ends_with_marker
}

fn is_current_invariant_developer_content(
    content: &[ContentItem],
    invariant_prefix: &[ResponseItem],
) -> bool {
    let Some(historical_sections) = developer_sections_without_rolling_marker(content) else {
        return false;
    };
    if historical_sections.is_empty() {
        return false;
    }

    invariant_prefix.iter().any(|item| {
        let ResponseItem::Message {
            role,
            content: current_content,
            ..
        } = item
        else {
            return false;
        };
        if role != "developer" {
            return false;
        }
        developer_sections_without_rolling_marker(current_content)
            .is_some_and(|current_sections| current_sections == historical_sections)
    })
}

fn developer_sections_without_rolling_marker(content: &[ContentItem]) -> Option<Vec<&str>> {
    let mut sections = Vec::new();
    for item in content {
        let ContentItem::InputText { text } = item else {
            return None;
        };
        if !RollingInvariantDeveloperContext::matches_text(text) {
            sections.push(text.as_str());
        }
    }
    Some(sections)
}
