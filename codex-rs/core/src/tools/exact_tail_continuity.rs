use std::collections::BTreeSet;

use crate::context::ContextualUserFragment;
use crate::context::ExactTailToolSurfaceNotice;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::ExactTailToolSurfaceDiagnosticEvent;
use codex_tools::ToolName;
use serde_json::Value;

pub(crate) const EXACT_TAIL_TOOL_SURFACE_REFERENCE_LIMIT: usize = 32;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ExactTailToolSurfaceHint {
    pub(crate) references: Vec<ToolName>,
    pub(crate) hot_tool_call_count: usize,
    pub(crate) hot_tool_namespace_count: usize,
    pub(crate) hot_tool_reference_count: usize,
    pub(crate) hot_tool_reference_overflow_count: usize,
    pub(crate) out_of_scope_dependency_protocol_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PendingExactTailToolSurfaceHint {
    pub(crate) compaction_id: String,
    pub(crate) route: String,
    pub(crate) hint: ExactTailToolSurfaceHint,
    pub(crate) notice_already_emitted: bool,
}

impl PendingExactTailToolSurfaceHint {
    pub(crate) fn new(
        compaction_id: impl Into<String>,
        route: impl Into<String>,
        hint: ExactTailToolSurfaceHint,
    ) -> Self {
        Self {
            compaction_id: compaction_id.into(),
            route: route.into(),
            hint,
            notice_already_emitted: false,
        }
    }

    pub(crate) fn with_notice_already_emitted(mut self, value: bool) -> Self {
        self.notice_already_emitted = value;
        self
    }

    pub(crate) fn mark_notice_emitted(&mut self) {
        self.notice_already_emitted = true;
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ExactTailToolSurfaceOutcome {
    pub(crate) compaction_id: String,
    pub(crate) route: String,
    pub(crate) hot_tool_call_count: usize,
    pub(crate) hot_tool_namespace_count: usize,
    pub(crate) hot_tool_reference_count: usize,
    pub(crate) rehydrated_tool_count: usize,
    pub(crate) missing_hot_tool_count: usize,
    pub(crate) already_direct_tool_count: usize,
    pub(crate) discoverable_hot_tool_count: usize,
    pub(crate) missing_notice_emitted_count: usize,
    pub(crate) missing_no_path_count: usize,
    pub(crate) rehydrated_tool_references: Vec<String>,
    pub(crate) missing_tool_references: Vec<String>,
    pub(crate) rehydrated_tool_namespaces: Vec<String>,
    pub(crate) missing_tool_rejection_reasons: Vec<String>,
    pub(crate) hot_tool_reference_overflow_count: usize,
    pub(crate) hot_tool_reference_overflow_notice_emitted_count: usize,
    pub(crate) out_of_scope_dependency_protocol_count: usize,
    pub(crate) tool_surface_changed_after_compaction: bool,
    pub(crate) tool_surface_rehydration_failure_reason: Option<String>,
}

impl ExactTailToolSurfaceOutcome {
    pub(crate) fn event(
        &self,
        thread_id: String,
        turn_id: String,
    ) -> ExactTailToolSurfaceDiagnosticEvent {
        ExactTailToolSurfaceDiagnosticEvent {
            thread_id,
            turn_id,
            compaction_id: self.compaction_id.clone(),
            route: self.route.clone(),
            hot_tool_call_count: self.hot_tool_call_count,
            hot_tool_namespace_count: self.hot_tool_namespace_count,
            hot_tool_reference_count: self.hot_tool_reference_count,
            rehydrated_tool_count: self.rehydrated_tool_count,
            missing_hot_tool_count: self.missing_hot_tool_count,
            already_direct_tool_count: self.already_direct_tool_count,
            discoverable_hot_tool_count: self.discoverable_hot_tool_count,
            missing_notice_emitted_count: self.missing_notice_emitted_count,
            missing_no_path_count: self.missing_no_path_count,
            rehydrated_tool_references: self.rehydrated_tool_references.clone(),
            missing_tool_references: self.missing_tool_references.clone(),
            rehydrated_tool_namespaces: self.rehydrated_tool_namespaces.clone(),
            missing_tool_rejection_reasons: self.missing_tool_rejection_reasons.clone(),
            hot_tool_reference_overflow_count: self.hot_tool_reference_overflow_count,
            hot_tool_reference_overflow_notice_emitted_count: self
                .hot_tool_reference_overflow_notice_emitted_count,
            out_of_scope_dependency_protocol_count: self.out_of_scope_dependency_protocol_count,
            tool_surface_changed_after_compaction: self.tool_surface_changed_after_compaction,
            tool_surface_rehydration_failure_reason: self
                .tool_surface_rehydration_failure_reason
                .clone(),
        }
    }
}

pub(crate) fn derive_exact_tail_tool_surface_hint(
    hot_suffix: &[ResponseItem],
) -> ExactTailToolSurfaceHint {
    let mut collector = ToolSurfaceHintCollector::default();
    for item in hot_suffix.iter().rev() {
        collector.collect_item(item);
    }
    collector.finish()
}

pub(crate) fn exact_tail_tool_surface_notice_item(
    outcome: &ExactTailToolSurfaceOutcome,
) -> Option<ResponseItem> {
    let missing = outcome.missing_hot_tool_count;
    let overflow = outcome.hot_tool_reference_overflow_count;
    if outcome.missing_notice_emitted_count == 0
        && outcome.hot_tool_reference_overflow_notice_emitted_count == 0
    {
        return None;
    }

    Some(ContextualUserFragment::into(
        ExactTailToolSurfaceNotice::new(
            (outcome.missing_notice_emitted_count > 0)
                .then_some(missing)
                .unwrap_or(0),
            (outcome.hot_tool_reference_overflow_notice_emitted_count > 0)
                .then_some(overflow)
                .unwrap_or(0),
            outcome.missing_no_path_count,
            outcome.missing_tool_references.clone(),
        ),
    ))
}

#[derive(Default)]
struct ToolSurfaceHintCollector {
    references: Vec<ToolName>,
    seen_references: BTreeSet<ToolName>,
    namespaces: BTreeSet<String>,
    hot_tool_call_count: usize,
    hot_tool_reference_overflow_count: usize,
    out_of_scope_dependency_protocol_count: usize,
}

impl ToolSurfaceHintCollector {
    fn collect_item(&mut self, item: &ResponseItem) {
        match item {
            ResponseItem::FunctionCall {
                namespace, name, ..
            } => {
                self.hot_tool_call_count += 1;
                self.push_reference(ToolName::new(namespace.clone(), name.clone()));
            }
            ResponseItem::CustomToolCall { name, .. } => {
                self.hot_tool_call_count += 1;
                self.push_reference(ToolName::plain(name.clone()));
            }
            ResponseItem::ToolSearchCall { .. } => {
                self.hot_tool_call_count += 1;
            }
            ResponseItem::ToolSearchOutput { tools, .. } => {
                for tool in tools {
                    for tool_name in tool_search_output_tool_names(tool) {
                        self.push_reference(tool_name);
                    }
                }
            }
            ResponseItem::LocalShellCall { .. }
            | ResponseItem::WebSearchCall { .. }
            | ResponseItem::ImageGenerationCall { .. } => {
                self.hot_tool_call_count += 1;
                self.out_of_scope_dependency_protocol_count += 1;
            }
            ResponseItem::FunctionCallOutput { .. } | ResponseItem::CustomToolCallOutput { .. } => {
                self.out_of_scope_dependency_protocol_count += 1;
            }
            ResponseItem::Message { .. }
            | ResponseItem::AgentMessage { .. }
            | ResponseItem::Reasoning { .. }
            | ResponseItem::Compaction { .. }
            | ResponseItem::CompactionTrigger { .. }
            | ResponseItem::ContextCompaction { .. }
            | ResponseItem::Other => {}
        }
    }

    fn push_reference(&mut self, tool_name: ToolName) {
        if let Some(namespace) = &tool_name.namespace {
            self.namespaces.insert(namespace.clone());
        }
        if !self.seen_references.insert(tool_name.clone()) {
            return;
        }
        if self.references.len() < EXACT_TAIL_TOOL_SURFACE_REFERENCE_LIMIT {
            self.references.push(tool_name);
        } else {
            self.hot_tool_reference_overflow_count += 1;
        }
    }

    fn finish(self) -> ExactTailToolSurfaceHint {
        ExactTailToolSurfaceHint {
            hot_tool_reference_count: self
                .references
                .len()
                .saturating_add(self.hot_tool_reference_overflow_count),
            references: self.references,
            hot_tool_call_count: self.hot_tool_call_count,
            hot_tool_namespace_count: self.namespaces.len(),
            hot_tool_reference_overflow_count: self.hot_tool_reference_overflow_count,
            out_of_scope_dependency_protocol_count: self.out_of_scope_dependency_protocol_count,
        }
    }
}

fn tool_search_output_tool_names(value: &Value) -> Vec<ToolName> {
    match value.get("type").and_then(Value::as_str) {
        Some("function") => value
            .get("name")
            .and_then(Value::as_str)
            .map(ToolName::plain)
            .into_iter()
            .collect(),
        Some("namespace") => {
            let Some(namespace) = value.get("name").and_then(Value::as_str) else {
                return Vec::new();
            };
            value
                .get("tools")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("function"))
                .filter_map(|tool| tool.get("name").and_then(Value::as_str))
                .map(|name| ToolName::namespaced(namespace.to_string(), name.to_string()))
                .collect()
        }
        Some(_) | None => Vec::new(),
    }
}

#[cfg(test)]
#[path = "exact_tail_continuity_tests.rs"]
mod tests;
