use crate::compact::COMPACT_USER_MESSAGE_MAX_TOKENS;
use crate::compact::CompactedUserMessage;
use crate::compact::InitialContextInjection;
use crate::config::Config;
use crate::session::session::Session;
use crate::session::turn_context::TurnContext;
use codex_analytics::CompactionTrigger;
use codex_protocol::error::CodexErr;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ResponseItem;

pub(crate) const EXACT_TAIL_CONSERVATIVE_SUMMARY_BUDGET_TOKENS: i64 = 16_384;
pub(crate) const EXACT_TAIL_REPLACEMENT_OVERHEAD_MARGIN_TOKENS: i64 = 1_024;
pub(crate) const EXACT_TAIL_MIN_SAFETY_MARGIN_TOKENS: i64 = 4_096;
pub(crate) const EXACT_TAIL_AUTO_COMPACT_TRIGGER_MARGIN_TOKENS: i64 = 4_096;
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
        .unwrap_or(EXACT_TAIL_DEFAULT_MAX_MODEL_VISIBLE_ITEM_TOKENS)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExactTailImplementation {
    Local,
    RemoteLegacy,
    RemoteV2,
}

impl ExactTailImplementation {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::RemoteLegacy => "remote",
            Self::RemoteV2 => "remote_v2",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExactTailFailReason {
    BudgetUnavailable,
    NoColdPrefix,
    MinimumHotSuffixTooLarge,
    ReplacementTooLarge,
    ColdInputTooLarge,
    UnsupportedRemoteV2Ordering,
    NoUsableColdSummary,
    ModelVisibleItemTooLarge,
}

impl ExactTailFailReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::BudgetUnavailable => "ExactTailBudgetUnavailable",
            Self::NoColdPrefix => "ExactTailNoColdPrefix",
            Self::MinimumHotSuffixTooLarge => "ExactTailMinimumHotSuffixTooLarge",
            Self::ReplacementTooLarge => "ExactTailReplacementTooLarge",
            Self::ColdInputTooLarge => "ExactTailColdInputTooLarge",
            Self::UnsupportedRemoteV2Ordering => "ExactTailUnsupportedForRemoteV2Ordering",
            Self::NoUsableColdSummary => "ExactTailNoUsableColdSummary",
            Self::ModelVisibleItemTooLarge => "ExactTailModelVisibleItemTooLarge",
        }
    }
}

#[derive(Debug)]
pub(crate) struct ExactTailError {
    pub(super) reason: ExactTailFailReason,
    message: String,
}

impl ExactTailError {
    pub(crate) fn new(reason: ExactTailFailReason, message: impl Into<String>) -> Self {
        Self {
            reason,
            message: message.into(),
        }
    }

    pub(crate) fn into_codex_err(self) -> CodexErr {
        let reason = self.reason.as_str();
        let message = if self.message.contains(reason) {
            self.message
        } else {
            format!("{reason}: {}", self.message)
        };
        CodexErr::Stream(message, None)
    }
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
    pub(crate) safety_margin: i64,
    pub(crate) available_for_hot: i64,
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
    pub(crate) hot_suffix: Vec<ResponseItem>,
    pub(crate) cold_user_messages: Vec<CompactedUserMessage>,
    pub(crate) diagnostics: ExactTailDiagnostics,
    pub(crate) coverage: ExactTailCoverage,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedExactTailPlan {
    pub(crate) plan: ExactTailPlan,
    pub(crate) initial_context: Vec<ResponseItem>,
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
    pub(crate) sess: &'a Session,
    pub(crate) turn_context: &'a TurnContext,
    pub(crate) history_items: &'a [ResponseItem],
    pub(crate) base_instructions: &'a BaseInstructions,
    pub(crate) policy: CompactionHistoryPolicy,
    pub(crate) trigger: CompactionTrigger,
    pub(crate) initial_context_injection: InitialContextInjection,
    pub(crate) estimated_summary_scaffold_overhead_tokens: i64,
    pub(crate) retained_cold_user_message_budget_tokens: i64,
    pub(crate) implementation: ExactTailImplementation,
}
