use std::sync::Arc;

use super::SessionTask;
use super::SessionTaskContext;
use super::SessionTaskResult;
use super::emit_compact_metric;
use crate::session::TurnInput;
use crate::session::turn_context::TurnContext;
use crate::state::TaskKind;
use codex_features::Feature;
use codex_protocol::error::CodexErr;
use codex_protocol::protocol::EventMsg;
use codex_protocol::user_input::UserInput;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, Default)]
pub(crate) struct CompactTask;

impl SessionTask for CompactTask {
    fn kind(&self) -> TaskKind {
        TaskKind::Compact
    }

    fn span_name(&self) -> &'static str {
        "session_task.compact"
    }

    async fn run(
        self: Arc<Self>,
        session: Arc<SessionTaskContext>,
        ctx: Arc<TurnContext>,
        _input: Vec<TurnInput>,
        _cancellation_token: CancellationToken,
    ) -> SessionTaskResult {
        let session = session.clone_session();
        if ctx.config.features.enabled(Feature::TokenBudget) {
            crate::compact_token_budget::run_manual_compact_task(session, ctx).await?;
            return Ok(None);
        }

        let route = match crate::compact::compact_route(&ctx) {
            Ok(route) => route,
            Err(err) => {
                session.track_turn_codex_error(&ctx, &err);
                session
                    .send_event(
                        &ctx,
                        EventMsg::Error(
                            err.to_error_event(Some("Error running compact task".to_string())),
                        ),
                    )
                    .await;
                return Ok(None);
            }
        };
        emit_compact_metric(
            &session.services.session_telemetry,
            route.metric_name(),
            /*manual*/ true,
        );
        let result = match route {
            crate::compact::CompactRoute::RemoteV2 => {
                crate::compact_remote_v2::run_remote_compact_task(session.clone(), ctx).await
            }
            crate::compact::CompactRoute::RemoteLegacy => {
                crate::compact_remote::run_remote_compact_task(session.clone(), ctx).await
            }
            crate::compact::CompactRoute::SemanticTranscript => {
                crate::compact_semantic_transcript::run_remote_compact_task(session.clone(), ctx)
                    .await
            }
            crate::compact::CompactRoute::Local => {
                let input = vec![UserInput::Text {
                    text: ctx
                        .config
                        .compact_prompt
                        .as_deref()
                        .unwrap_or(crate::compact::SUMMARIZATION_PROMPT)
                        .to_string(),
                    // Compaction prompt is synthesized; no UI element ranges to preserve.
                    text_elements: Vec::new(),
                }];
                crate::compact::run_compact_task(session.clone(), ctx, input).await
            }
        };
        if let Err(err @ CodexErr::TurnAborted) = result {
            return Err(err);
        }
        Ok(None)
    }
}
