use crate::compact::COMPACT_USER_MESSAGE_MAX_TOKENS;
use crate::compact::CompactedUserMessage;
use crate::compact::InitialContextInjection;
use crate::config::Config;
use crate::context_manager::HistoryItemProvenance;
use crate::context_manager::model_visible_tool_output_item_token_limit;
use crate::session::session::Session;
use crate::session::turn_context::TurnContext;
use crate::tools::exact_tail_continuity::ExactTailToolSurfaceHint;
use codex_analytics::CompactionTrigger;
use codex_protocol::error::CodexErr;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::ExactTailModelVisibleItemKind;
use std::sync::Arc;

pub(crate) const EXACT_TAIL_CONSERVATIVE_SUMMARY_BUDGET_TOKENS: i64 = 16_384;
pub(crate) const EXACT_TAIL_REPLACEMENT_OVERHEAD_MARGIN_TOKENS: i64 = 1_024;
pub(crate) const EXACT_TAIL_MIN_SAFETY_MARGIN_TOKENS: i64 = 4_096;
pub(crate) const EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS: i64 = 10_000;
pub(crate) const EXACT_TAIL_LOCAL_RETAINED_COLD_USER_MESSAGE_BUDGET_TOKENS: i64 =
    COMPACT_USER_MESSAGE_MAX_TOKENS as i64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompactionHistoryPolicy {
    Standard,
    PreserveRecentExact {
        target_tokens: i64,
        max_model_visible_item_tokens: i64,
    },
}

impl CompactionHistoryPolicy {
    pub(crate) fn from_config(config: &Config) -> Self {
        match config.compact_preserve_recent_tokens {
            Some(target_tokens) if target_tokens > 0 => Self::PreserveRecentExact {
                target_tokens,
                max_model_visible_item_tokens: max_model_visible_item_tokens(config),
            },
            Some(_) | None => Self::Standard,
        }
    }
}

fn max_model_visible_item_tokens(config: &Config) -> i64 {
    config
        .tool_output_token_limit
        .and_then(|tokens| i64::try_from(tokens).ok())
        .filter(|tokens| *tokens > 0)
        .map(model_visible_tool_output_item_token_limit)
        .unwrap_or(EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExactTailImplementation {
    Local,
    RemoteLegacy,
    RemoteV2,
    SemanticTranscript,
}

impl ExactTailImplementation {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::RemoteLegacy => "remote_legacy",
            Self::RemoteV2 => "remote_v2",
            Self::SemanticTranscript => "semantic_transcript",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExactTailFailReason {
    BudgetUnavailable,
    NoColdPrefix,
    MinimumHotSuffixTooLarge,
    ReplacementTooLarge,
    ColdInputMismatch,
    ColdInputTooLarge,
    NoUsableColdSummary,
    ModelVisibleItemTooLarge,
    BackendContextExceeded,
    HotSuffixItemIdMissing,
    HotSuffixMismatch,
}

impl ExactTailFailReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::BudgetUnavailable => "ExactTailBudgetUnavailable",
            Self::NoColdPrefix => "ExactTailNoColdPrefix",
            Self::MinimumHotSuffixTooLarge => "ExactTailMinimumHotSuffixTooLarge",
            Self::ReplacementTooLarge => "ExactTailReplacementTooLarge",
            Self::ColdInputMismatch => "ExactTailColdInputMismatch",
            Self::ColdInputTooLarge => "ExactTailColdInputTooLarge",
            Self::NoUsableColdSummary => "ExactTailNoUsableColdSummary",
            Self::ModelVisibleItemTooLarge => "ExactTailModelVisibleItemTooLarge",
            Self::BackendContextExceeded => "ExactTailBackendContextExceeded",
            Self::HotSuffixItemIdMissing => "ExactTailHotSuffixItemIdMissing",
            Self::HotSuffixMismatch => "ExactTailHotSuffixMismatch",
        }
    }
}

#[derive(Debug)]
pub(crate) struct ExactTailError {
    pub(crate) reason: ExactTailFailReason,
    message: String,
    pub(crate) model_visible_item_limit: Option<ExactTailModelVisibleItemLimit>,
}

impl ExactTailError {
    pub(crate) fn new(reason: ExactTailFailReason, message: impl Into<String>) -> Self {
        Self {
            reason,
            message: message.into(),
            model_visible_item_limit: None,
        }
    }

    pub(crate) fn with_model_visible_item_limit(
        mut self,
        limit: ExactTailModelVisibleItemLimit,
    ) -> Self {
        self.model_visible_item_limit = Some(limit);
        self
    }

    pub(crate) fn into_codex_err(self) -> CodexErr {
        let reason = self.reason.as_str();
        let message = if self.message.contains(reason) {
            self.message
        } else {
            format!("{reason}: {}", self.message)
        };
        CodexErr::ExactTailCompactionFailed(message)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExactTailModelVisibleItemLimit {
    pub(crate) item_kind: ExactTailModelVisibleItemKind,
    pub(crate) item_tokens: i64,
    pub(crate) max_item_tokens: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExactTailItemClass {
    ConversationOrProtocol,
    DependencyProtocol,
    StaleContextWrapper,
    MixedDeveloperContext,
    Unsupported,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExactTailCoverage {
    pub(crate) cold_covered_groups: Vec<usize>,
    pub(crate) hot_exact_groups: Vec<usize>,
    pub(crate) filtered_stale_groups: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExactTailDiagnostics {
    pub(crate) implementation: ExactTailImplementation,
    pub(crate) raw_item_count: usize,
    pub(crate) group_count: usize,
    pub(crate) cold_group_count: usize,
    pub(crate) hot_group_count: usize,
    pub(crate) filtered_context_item_count: usize,
    pub(crate) requested_hot_tokens: i64,
    pub(crate) actual_hot_tokens: i64,
    pub(crate) cold_tokens: i64,
    pub(crate) effective_replacement_budget: i64,
    pub(crate) conservative_summary_budget_tokens: i64,
    pub(crate) estimated_summary_scaffold_overhead_tokens: i64,
    pub(crate) retained_cold_user_message_budget_tokens: i64,
    pub(crate) replacement_overhead_margin_tokens: i64,
    pub(crate) conservative_cold_summary_budget: i64,
    pub(crate) required_current_context_budget: i64,
    pub(crate) final_replacement_extra_budget_tokens: i64,
    pub(crate) max_model_visible_item_tokens: i64,
    pub(crate) normalized_tool_output_count: usize,
    pub(crate) safety_margin: i64,
    pub(crate) available_for_hot: i64,
    pub(crate) largest_hot_item_tokens: i64,
    pub(crate) post_summary_cold_reserve_target_tokens: i64,
    pub(crate) post_summary_cold_reserve_tokens: i64,
    pub(crate) post_summary_cold_reserve_group_count: usize,
    pub(crate) actual_summary_tokens: Option<i64>,
    pub(crate) attempted_replacement_tokens_estimate: Option<i64>,
    pub(crate) attempted_final_replacement_tokens_estimate: Option<i64>,
    pub(crate) semantic_transcript_tokens: Option<i64>,
    pub(crate) semantic_transcript_item_count: Option<usize>,
    pub(crate) semantic_transcript_tool_observation_count: Option<usize>,
    pub(crate) raw_cold_tokens: Option<i64>,
    pub(crate) semantic_transcript_reduction_tokens: Option<i64>,
    pub(crate) retained_cold_message_tokens: Option<i64>,
    pub(crate) retained_cold_message_count: Option<usize>,
    pub(crate) hot_tool_call_count: usize,
    pub(crate) hot_tool_namespace_count: usize,
    pub(crate) hot_tool_reference_count: usize,
    pub(crate) hot_tool_reference_overflow_count: usize,
    pub(crate) out_of_scope_dependency_protocol_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExactTailBudgetReservation {
    pub(crate) conservative_cold_summary_budget: i64,
    pub(crate) safety_margin: i64,
    pub(crate) available_for_hot: i64,
}

#[derive(Clone, Debug)]
pub(crate) struct ExactTailPlan {
    pub(crate) cold_history: Vec<ResponseItem>,
    pub(crate) cold_history_provenance: Vec<HistoryItemProvenance>,
    pub(crate) hot_suffix: Vec<ResponseItem>,
    pub(crate) hot_suffix_provenance: Vec<HistoryItemProvenance>,
    pub(crate) cold_user_messages: Vec<CompactedUserMessage>,
    pub(crate) diagnostics: ExactTailDiagnostics,
    pub(crate) coverage: ExactTailCoverage,
    pub(crate) tool_surface_hint: ExactTailToolSurfaceHint,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedExactTailPlan {
    pub(crate) plan: ExactTailPlan,
    pub(crate) initial_context: Vec<ResponseItem>,
    pub(crate) model_context_window: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExactTailReplacement {
    pub(crate) replacement_history: Vec<ResponseItem>,
    pub(crate) item_provenance: Vec<HistoryItemProvenance>,
    pub(crate) diagnostics: ExactTailReplacementDiagnostics,
    pub(crate) tool_surface_hint: ExactTailToolSurfaceHint,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExactTailReplacementDiagnostics {
    pub(crate) actual_summary_tokens: i64,
    pub(crate) replacement_tokens_estimate: i64,
    pub(crate) final_replacement_tokens_estimate: i64,
    pub(crate) hot_suffix_exact_match: bool,
    pub(crate) planned_hot_suffix_item_count: usize,
    pub(crate) installed_hot_suffix_item_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExactTailHotSuffixProof {
    pub(crate) exact_match: bool,
    pub(crate) planned_item_count: usize,
    pub(crate) installed_item_count: usize,
}

pub(crate) struct ExactTailPlanInput<'a> {
    pub(crate) history_items: &'a [ResponseItem],
    pub(crate) target_tokens: i64,
    pub(crate) effective_replacement_budget: Option<i64>,
    pub(crate) required_current_context_budget: i64,
    pub(crate) final_replacement_extra_budget_tokens: i64,
    pub(crate) max_model_visible_item_tokens: i64,
    pub(crate) estimated_summary_scaffold_overhead_tokens: i64,
    pub(crate) retained_cold_user_message_budget_tokens: i64,
    pub(crate) implementation: ExactTailImplementation,
}

pub(crate) struct ExactTailPrepareInput<'a> {
    pub(crate) sess: &'a Arc<Session>,
    pub(crate) turn_context: &'a Arc<TurnContext>,
    pub(crate) history_items: &'a [ResponseItem],
    pub(crate) history_item_provenance: &'a [HistoryItemProvenance],
    pub(crate) base_instructions: &'a BaseInstructions,
    pub(crate) policy: CompactionHistoryPolicy,
    pub(crate) trigger: CompactionTrigger,
    pub(crate) initial_context_injection: &'a InitialContextInjection,
    pub(crate) estimated_summary_scaffold_overhead_tokens: i64,
    pub(crate) retained_cold_user_message_budget_tokens: i64,
    pub(crate) normalized_tool_output_count: usize,
    pub(crate) implementation: ExactTailImplementation,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ExactTailCompactInputExpectation<'a> {
    Source { derived_items: &'a [ResponseItem] },
    Derived { expected_items: &'a [ResponseItem] },
}
