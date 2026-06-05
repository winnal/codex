use codex_protocol::config_types::PromptRetentionMode;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ResponseItem;
use tracing::debug;
use tracing::warn;

use super::session::Session;
use super::turn_context::TurnContext;
use crate::context_manager::RollingPromptError;
use crate::context_manager::RollingPromptParams;
use crate::context_manager::RollingPromptState;
use crate::context_manager::build_rolling_prompt;
use crate::skills::SkillRenderSideEffects;

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
