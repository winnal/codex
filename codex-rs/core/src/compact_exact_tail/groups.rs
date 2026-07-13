use crate::compact_exact_tail::ExactTailItemClass;
use crate::compact_exact_tail::classify_exact_tail_history_item;
use crate::context_manager::HistoryItemProvenance;
use crate::context_manager::estimate_response_items_token_count;
use codex_protocol::models::ResponseItem;

#[derive(Clone, Debug)]
pub(super) struct ExactTailGroup {
    pub(super) id: usize,
    pub(super) items: Vec<ResponseItem>,
    pub(super) item_provenance: Vec<HistoryItemProvenance>,
    pub(super) tokens: i64,
}

pub(super) fn build_groups(
    history_items: &[ResponseItem],
    item_provenance: &[HistoryItemProvenance],
) -> (Vec<ExactTailGroup>, Vec<usize>, usize) {
    let mut eligible = Vec::new();
    let mut filtered_stale_groups = Vec::new();
    let mut filtered_context_item_count = 0usize;

    for (index, (item, provenance)) in history_items.iter().zip(item_provenance).enumerate() {
        match classify_exact_tail_history_item(item, *provenance) {
            ExactTailItemClass::StaleContextWrapper => {
                filtered_stale_groups.push(index);
                filtered_context_item_count += 1;
            }
            ExactTailItemClass::MixedDeveloperContext => {
                // Mixed developer bundles come from initial-context assembly. The caller rebuilds
                // and budgets current context separately, so the historical bundle is stale.
                filtered_stale_groups.push(index);
                filtered_context_item_count += 1;
            }
            ExactTailItemClass::ConversationOrProtocol
            | ExactTailItemClass::DependencyProtocol
            | ExactTailItemClass::Unsupported => {
                eligible.push((index, item.clone(), *provenance));
            }
        }
    }

    let mut groups = Vec::new();
    for (next_group_id, (start, end)) in group_intervals(&eligible).into_iter().enumerate() {
        let items = eligible[start..=end]
            .iter()
            .map(|(_, item, _)| item.clone())
            .collect::<Vec<_>>();
        let item_provenance = eligible[start..=end]
            .iter()
            .map(|(_, _, provenance)| *provenance)
            .collect::<Vec<_>>();
        let tokens = estimate_response_items_token_count(&items);
        groups.push(ExactTailGroup {
            id: next_group_id,
            items,
            item_provenance,
            tokens,
        });
    }

    (groups, filtered_stale_groups, filtered_context_item_count)
}

fn group_intervals(
    eligible: &[(usize, ResponseItem, HistoryItemProvenance)],
) -> Vec<(usize, usize)> {
    if eligible.is_empty() {
        return Vec::new();
    }

    let mut intervals = dependency_intervals(eligible);
    intervals.extend((0..eligible.len()).map(|index| (index, index)));
    merge_intervals(intervals)
}

fn dependency_intervals(
    eligible: &[(usize, ResponseItem, HistoryItemProvenance)],
) -> Vec<(usize, usize)> {
    let mut raw_intervals = Vec::new();
    for (index, (_, item, _)) in eligible.iter().enumerate() {
        let Some(key) = dependency_key(item) else {
            continue;
        };
        if let Some((_, _, end)) = raw_intervals
            .iter_mut()
            .find(|(existing_key, _, _)| existing_key == &key)
        {
            *end = index;
        } else {
            raw_intervals.push((key, index, index));
        }
    }

    raw_intervals
        .into_iter()
        .filter_map(|(_, start, end)| (start < end).then_some((start, end)))
        .collect()
}

fn merge_intervals(mut intervals: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    intervals.sort_unstable_by_key(|(start, _)| *start);

    let mut merged = Vec::<(usize, usize)>::new();
    for (start, end) in intervals {
        if let Some((_, last_end)) = merged.last_mut()
            && start <= *last_end
        {
            *last_end = (*last_end).max(end);
            continue;
        }
        merged.push((start, end));
    }
    merged
}

fn dependency_key(item: &ResponseItem) -> Option<String> {
    match item {
        ResponseItem::LocalShellCall {
            call_id: Some(call_id),
            ..
        }
        | ResponseItem::ToolSearchCall {
            call_id: Some(call_id),
            ..
        }
        | ResponseItem::ToolSearchOutput {
            call_id: Some(call_id),
            ..
        } => Some(call_id.clone()),
        ResponseItem::FunctionCall { call_id, .. }
        | ResponseItem::FunctionCallOutput { call_id, .. }
        | ResponseItem::CustomToolCall { call_id, .. }
        | ResponseItem::CustomToolCallOutput { call_id, .. } => Some(call_id.clone()),
        _ => None,
    }
}
