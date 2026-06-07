use codex_protocol::config_types::PromptRetentionMode;
use codex_protocol::config_types::ROLLCTX_SUMMARY_GROUP_TOKEN_CAP_MAX;
use codex_protocol::config_types::RollingCompactionMode;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ResponseItem;
use tracing::debug;
use tracing::warn;

use super::session::Session;
use super::turn_context::TurnContext;
use crate::client::ModelClientSession;
use crate::client_common::Prompt;
use crate::context_manager::PairwiseRollingPromptParams;
use crate::context_manager::PairwiseSummaryRequest;
use crate::context_manager::RollingPromptBuildOutcome;
use crate::context_manager::RollingPromptError;
use crate::context_manager::RollingPromptParams;
use crate::context_manager::RollingPromptState;
use crate::context_manager::build_rolling_prompt;
use crate::context_manager::build_rolling_prompt_with_live_summaries;
use crate::context_manager::estimate_response_item_token_count;
use crate::skills::SkillRenderSideEffects;

const DEFAULT_SUMMARY_GROUP_TOKEN_CAP: usize = ROLLCTX_SUMMARY_GROUP_TOKEN_CAP_MAX as usize;
const DEFAULT_MAX_SUMMARY_LEVELS: u8 = 8;
const DEFAULT_COMPACT_WHEN_LEVEL_GROUP_COUNT_GT: usize = 2;
const MAX_PAIR_SUMMARIES_PER_SAMPLING_ATTEMPT: usize = 2;
const MAX_BOOTSTRAP_PAIR_SUMMARIES_PER_SAMPLING_ATTEMPT: usize = 16;
const DEFAULT_ROLLCTX_PAIR_SUMMARY_INPUT_TOKEN_CAP: i64 = 22_000;

#[derive(Clone, Debug)]
pub(crate) struct SamplingPromptInput {
    pub(crate) input: Vec<ResponseItem>,
    rolling_prompt_state: Option<RollingPromptState>,
}

impl SamplingPromptInput {
    fn compact(input: Vec<ResponseItem>) -> Self {
        Self {
            input,
            rolling_prompt_state: None,
        }
    }

    fn rolling(input: Vec<ResponseItem>, projected_state: RollingPromptState) -> Self {
        Self {
            input,
            rolling_prompt_state: Some(projected_state),
        }
    }

    pub(crate) fn into_input(self) -> Vec<ResponseItem> {
        self.input
    }

    pub(crate) fn projected_rolling_prompt_state(&self) -> Option<RollingPromptState> {
        self.rolling_prompt_state.clone()
    }
}

impl Session {
    pub(crate) async fn build_rolling_invariant_context(
        &self,
        turn_context: &TurnContext,
    ) -> Vec<ResponseItem> {
        self.build_initial_context_with_skill_side_effects(
            turn_context,
            SkillRenderSideEffects::None,
        )
        .await
    }

    pub(crate) async fn build_sampling_prompt_input(
        &self,
        turn_context: &TurnContext,
        base_instructions: &BaseInstructions,
        rolling_invariant_items: &[ResponseItem],
        rolling_target_scale_percent: Option<u8>,
    ) -> CodexResult<SamplingPromptInput> {
        match turn_context.config.prompt_retention {
            PromptRetentionMode::Compact => Ok(SamplingPromptInput::compact(
                self.clone_history()
                    .await
                    .for_prompt(&turn_context.model_info.input_modalities),
            )),
            PromptRetentionMode::Rolling => {
                self.build_rolling_sampling_prompt_input(
                    turn_context,
                    base_instructions,
                    rolling_invariant_items,
                    rolling_target_scale_percent,
                )
                .await
            }
        }
    }

    pub(crate) async fn build_sampling_prompt_input_with_live_pair_summaries(
        &self,
        turn_context: &TurnContext,
        base_instructions: &BaseInstructions,
        rolling_invariant_items: &[ResponseItem],
        rolling_target_scale_percent: Option<u8>,
        client_session: &ModelClientSession,
        retry_projected_state: Option<RollingPromptState>,
    ) -> CodexResult<SamplingPromptInput> {
        let mut generated_summaries = 0usize;
        let mut candidate_state = {
            let state = self.state.lock().await;
            retry_projected_state.as_ref().map_or_else(
                || state.rolling_prompt_state.clone(),
                |projected| {
                    state
                        .rolling_prompt_state
                        .with_pairwise_summaries_from_projection(projected)
                },
            )
        };
        loop {
            let (history, committed_state) = {
                let state = self.state.lock().await;
                (state.history.clone(), state.rolling_prompt_state.clone())
            };
            if candidate_state.history_version != committed_state.history_version {
                candidate_state = committed_state;
            }

            let outcome = build_rolling_prompt_with_live_summaries(
                &history,
                &candidate_state,
                rolling_prompt_params(
                    turn_context,
                    base_instructions,
                    rolling_invariant_items,
                    rolling_target_scale_percent,
                ),
            )
            .map_err(|error| rolling_prompt_error_to_codex_err(error, turn_context))?;

            match outcome {
                RollingPromptBuildOutcome::Ready(result) => {
                    debug!(
                        turn_id = %turn_context.sub_id,
                        estimated_prompt_tokens = result.estimated_prompt_tokens,
                        dropped_body_items = result.dropped_body_items,
                        raw_history_start_index = result.raw_history_start_index,
                        target_tokens = result.target_tokens,
                        backoff_applied = result.backoff_applied,
                        pair_summaries_generated = generated_summaries,
                        "built rolling sampling prompt with live pair summaries"
                    );
                    return Ok(SamplingPromptInput::rolling(
                        result.prompt_input,
                        result.projected_state,
                    ));
                }
                RollingPromptBuildOutcome::NeedsPairSummary {
                    request,
                    mut projected_state,
                } => {
                    if generated_summaries >= MAX_PAIR_SUMMARIES_PER_SAMPLING_ATTEMPT {
                        self.commit_pairwise_summary_cache_from_projection(&candidate_state)
                            .await;
                        return self
                            .build_sampling_prompt_input_with_bootstrap_pair_summaries(
                                turn_context,
                                base_instructions,
                                rolling_invariant_items,
                                rolling_target_scale_percent,
                                client_session,
                            )
                            .await;
                    }
                    let summary_text = self
                        .run_rollctx_pair_summary_request(turn_context, client_session, &request)
                        .await?;
                    request.add_to_projected_state(&mut projected_state, summary_text);
                    self.commit_pairwise_summary_cache_from_projection(&projected_state)
                        .await;
                    candidate_state = projected_state;
                    generated_summaries = generated_summaries.saturating_add(1);
                }
            }
        }
    }

    async fn build_sampling_prompt_input_with_bootstrap_pair_summaries(
        &self,
        turn_context: &TurnContext,
        base_instructions: &BaseInstructions,
        rolling_invariant_items: &[ResponseItem],
        rolling_target_scale_percent: Option<u8>,
        client_session: &ModelClientSession,
    ) -> CodexResult<SamplingPromptInput> {
        let mut generated_summaries = 0usize;
        loop {
            let (history, candidate_state) = {
                let state = self.state.lock().await;
                (state.history.clone(), state.rolling_prompt_state.clone())
            };

            let outcome = build_rolling_prompt_with_live_summaries(
                &history,
                &candidate_state,
                rolling_prompt_params(
                    turn_context,
                    base_instructions,
                    rolling_invariant_items,
                    rolling_target_scale_percent,
                ),
            )
            .map_err(|error| rolling_prompt_error_to_codex_err(error, turn_context))?;

            match outcome {
                RollingPromptBuildOutcome::Ready(result) => {
                    debug!(
                        turn_id = %turn_context.sub_id,
                        estimated_prompt_tokens = result.estimated_prompt_tokens,
                        dropped_body_items = result.dropped_body_items,
                        raw_history_start_index = result.raw_history_start_index,
                        target_tokens = result.target_tokens,
                        backoff_applied = result.backoff_applied,
                        bootstrap_pair_summaries_generated = generated_summaries,
                        "built rolling sampling prompt after bootstrap pair summaries"
                    );
                    return Ok(SamplingPromptInput::rolling(
                        result.prompt_input,
                        result.projected_state,
                    ));
                }
                RollingPromptBuildOutcome::NeedsPairSummary {
                    request,
                    mut projected_state,
                } => {
                    if generated_summaries >= MAX_BOOTSTRAP_PAIR_SUMMARIES_PER_SAMPLING_ATTEMPT {
                        return Err(CodexErr::ContextWindowExceeded);
                    }
                    let summary_text = self
                        .run_rollctx_pair_summary_request(turn_context, client_session, &request)
                        .await?;
                    request.add_to_summary_cache(&mut projected_state, summary_text);
                    self.commit_pairwise_summary_cache_from_projection(&projected_state)
                        .await;
                    generated_summaries = generated_summaries.saturating_add(1);
                }
            }
        }
    }

    async fn build_rolling_sampling_prompt_input(
        &self,
        turn_context: &TurnContext,
        base_instructions: &BaseInstructions,
        rolling_invariant_items: &[ResponseItem],
        rolling_target_scale_percent: Option<u8>,
    ) -> CodexResult<SamplingPromptInput> {
        let invariant_prefix = rolling_invariant_items.to_vec();
        let mut state = self.state.lock().await;
        let state = &mut *state;
        let history = &state.history;
        let rolling_prompt_state = &mut state.rolling_prompt_state;
        let result = build_rolling_prompt(
            history,
            rolling_prompt_state,
            RollingPromptParams {
                input_modalities: &turn_context.model_info.input_modalities,
                base_instructions,
                invariant_prefix,
                effective_context_window: turn_context.model_context_window(),
                reserve_percent: turn_context.config.rolling_context_reserve_percent,
                target_tokens: turn_context.config.rolling_context_target_tokens,
                target_scale_percent: rolling_target_scale_percent,
                tool_output_limit_tokens: i64::try_from(
                    turn_context.truncation_policy.token_budget(),
                )
                .unwrap_or(i64::MAX),
                pairwise_compaction: pairwise_rolling_prompt_params(turn_context),
            },
        )
        .map_err(|error| rolling_prompt_error_to_codex_err(error, turn_context))?;

        debug!(
            turn_id = %turn_context.sub_id,
            estimated_prompt_tokens = result.estimated_prompt_tokens,
            dropped_body_items = result.dropped_body_items,
            raw_history_start_index = result.raw_history_start_index,
            target_tokens = result.target_tokens,
            backoff_applied = result.backoff_applied,
            "built rolling sampling prompt"
        );

        Ok(SamplingPromptInput::rolling(
            result.prompt_input,
            result.projected_state,
        ))
    }

    pub(crate) async fn commit_sampling_prompt_input(&self, prompt_input: &SamplingPromptInput) {
        let Some(projected_state) = prompt_input.rolling_prompt_state.as_ref() else {
            return;
        };
        let mut state = self.state.lock().await;
        if state.history.history_version() == projected_state.history_version {
            let raw_history_len = state.history.raw_items().len();
            state
                .rolling_prompt_state
                .commit_projection(projected_state, raw_history_len);
        }
    }

    async fn commit_pairwise_summary_cache_from_projection(
        &self,
        projected_state: &RollingPromptState,
    ) {
        let mut state = self.state.lock().await;
        if state.history.history_version() == projected_state.history_version {
            state
                .rolling_prompt_state
                .commit_pairwise_summary_cache(projected_state);
        }
    }

    async fn run_rollctx_pair_summary_request(
        &self,
        turn_context: &TurnContext,
        client_session: &ModelClientSession,
        request: &PairwiseSummaryRequest,
    ) -> CodexResult<String> {
        let input_tokens = request
            .input
            .iter()
            .map(estimate_response_item_token_count)
            .sum::<i64>();
        if input_tokens > DEFAULT_ROLLCTX_PAIR_SUMMARY_INPUT_TOKEN_CAP {
            warn!(
                turn_id = %turn_context.sub_id,
                input_tokens,
                input_cap = DEFAULT_ROLLCTX_PAIR_SUMMARY_INPUT_TOKEN_CAP,
                "rollctx pair-summary input exceeds cap"
            );
            return Err(CodexErr::ContextWindowExceeded);
        }

        client_session
            .rollctx_pair_summary_text(
                &rollctx_pair_summary_prompt(request),
                &turn_context.model_info,
                &turn_context.session_telemetry,
                turn_context.config.service_tier.clone(),
                i64::try_from(
                    pairwise_rolling_prompt_params(turn_context)
                        .map(|params| params.summary_group_token_cap)
                        .unwrap_or(DEFAULT_SUMMARY_GROUP_TOKEN_CAP),
                )
                .unwrap_or(i64::MAX),
            )
            .await
    }
}

fn rolling_prompt_params<'a>(
    turn_context: &'a TurnContext,
    base_instructions: &'a BaseInstructions,
    rolling_invariant_items: &[ResponseItem],
    rolling_target_scale_percent: Option<u8>,
) -> RollingPromptParams<'a> {
    RollingPromptParams {
        input_modalities: &turn_context.model_info.input_modalities,
        base_instructions,
        invariant_prefix: rolling_invariant_items.to_vec(),
        effective_context_window: turn_context.model_context_window(),
        reserve_percent: turn_context.config.rolling_context_reserve_percent,
        target_tokens: turn_context.config.rolling_context_target_tokens,
        target_scale_percent: rolling_target_scale_percent,
        tool_output_limit_tokens: i64::try_from(turn_context.truncation_policy.token_budget())
            .unwrap_or(i64::MAX),
        pairwise_compaction: pairwise_rolling_prompt_params(turn_context),
    }
}

fn rollctx_pair_summary_prompt(request: &PairwiseSummaryRequest) -> Prompt {
    let coverage = request.coverage_label();
    let level = request.level;
    Prompt {
        input: request.input.clone(),
        tools: Vec::new(),
        parallel_tool_calls: false,
        base_instructions: BaseInstructions {
            text: format!(
                "Create a concise, non-authoritative ROLLCTX pair summary for level {level} covering {coverage}. Preserve only durable facts, decisions, open threads, implementation state, warnings or constraints, user preferences, and discarded low-value material. Do not invent authority or instructions."
            ),
        },
        personality: None,
        output_schema: None,
        output_schema_strict: true,
    }
}

fn pairwise_rolling_prompt_params(
    turn_context: &TurnContext,
) -> Option<PairwiseRollingPromptParams> {
    if turn_context.config.prompt_retention != PromptRetentionMode::Rolling
        || turn_context.config.rolling_compaction != RollingCompactionMode::Pairwise
    {
        return None;
    }

    Some(PairwiseRollingPromptParams {
        protected_hot_exact_tokens: turn_context.config.protected_hot_exact_tokens,
        summary_group_token_cap: turn_context
            .config
            .summary_group_token_cap
            .and_then(|cap| usize::try_from(cap).ok())
            .map_or(DEFAULT_SUMMARY_GROUP_TOKEN_CAP, |cap| {
                cap.min(DEFAULT_SUMMARY_GROUP_TOKEN_CAP)
            }),
        max_summary_levels: turn_context
            .config
            .max_summary_levels
            .unwrap_or(DEFAULT_MAX_SUMMARY_LEVELS),
        compact_when_level_group_count_gt: turn_context
            .config
            .compact_when_level_group_count_gt
            .unwrap_or(DEFAULT_COMPACT_WHEN_LEVEL_GROUP_COUNT_GT),
    })
}

fn rolling_prompt_error_to_codex_err(
    error: RollingPromptError,
    turn_context: &TurnContext,
) -> CodexErr {
    let message = error.to_string();
    warn!(
        turn_id = %turn_context.sub_id,
        error = %message,
        "failed to build rolling sampling prompt"
    );
    match error {
        RollingPromptError::NoContextWindow => CodexErr::InvalidRequest(message),
        RollingPromptError::InvariantPrefixExceedsTarget { .. }
        | RollingPromptError::FrontierExceedsBudget { .. }
        | RollingPromptError::ItemExceedsLimit { .. } => CodexErr::ContextWindowExceeded,
    }
}
