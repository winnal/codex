use super::*;
use crate::context_manager::is_user_turn_boundary;
use crate::tools::exact_tail_continuity::ExactTailToolSurfaceHint;
use crate::tools::exact_tail_continuity::PendingExactTailToolSurfaceHint;
use crate::tools::exact_tail_continuity::derive_exact_tail_tool_surface_hint;
use codex_protocol::protocol::ExactTailCompactionDiagnosticEvent;
use codex_protocol::protocol::ExactTailCompactionFitResult;
use codex_protocol::protocol::ExactTailCompactionRoute;
use uuid::Uuid;

// Return value of `Session::reconstruct_history_from_rollout`, bundling the rebuilt history with
// the resume/fork hydration metadata derived from the same replay.
#[derive(Debug)]
pub(super) struct RolloutReconstruction {
    pub(super) history: Vec<ResponseItem>,
    pub(super) previous_turn_settings: Option<PreviousTurnSettings>,
    pub(super) reference_context_item: Option<TurnContextItem>,
    pub(super) window_number: u64,
    pub(super) first_window_id: Option<Uuid>,
    pub(super) previous_window_id: Option<Uuid>,
    pub(super) window_id: Option<Uuid>,
    pub(super) pending_exact_tail_tool_surface_hint: Option<PendingExactTailToolSurfaceHint>,
}

#[derive(Debug, Clone, Copy)]
struct ReconstructedWindow {
    number: u64,
    first_id: Option<Uuid>,
    previous_id: Option<Uuid>,
    id: Option<Uuid>,
}

#[derive(Debug, Default)]
enum TurnReferenceContextItem {
    /// No `TurnContextItem` has been seen for this replay span yet.
    ///
    /// This differs from `Cleared`: `NeverSet` means there is no evidence this turn ever
    /// established a baseline, while `Cleared` means a baseline existed and a later compaction
    /// invalidated it. Only the latter must emit an explicit clearing segment for resume/fork
    /// hydration.
    #[default]
    NeverSet,
    /// A previously established baseline was invalidated by later compaction.
    Cleared,
    /// The latest baseline established by this replay span.
    Latest(Box<TurnContextItem>),
}

#[derive(Debug, Default)]
struct ActiveReplaySegment<'a> {
    turn_id: Option<String>,
    counts_as_user_turn: bool,
    previous_turn_settings: Option<PreviousTurnSettings>,
    reference_context_item: TurnReferenceContextItem,
    base_replacement_history: Option<&'a [ResponseItem]>,
    exact_tail_compaction_source: Option<RecoveredExactTailCompactionSource>,
    window: Option<ReconstructedWindow>,
}

#[derive(Debug, Default)]
struct RolloutReconstructionState<'a> {
    base_replacement_history: Option<&'a [ResponseItem]>,
    previous_turn_settings: Option<PreviousTurnSettings>,
    reference_context_item: TurnReferenceContextItem,
    window: Option<ReconstructedWindow>,
    exact_tail_compaction_source: Option<RecoveredExactTailCompactionSource>,
    pending_rollback_turns: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RecoveredExactTailCompactionSource {
    compaction_id: String,
    route: String,
    installed_hot_suffix_item_count: Option<usize>,
}

impl RecoveredExactTailCompactionSource {
    fn from_success_diagnostic(event: &ExactTailCompactionDiagnosticEvent) -> Option<Self> {
        (event.fit_result == ExactTailCompactionFitResult::Success).then(|| Self {
            compaction_id: event.compaction_id.clone(),
            route: match event.route {
                ExactTailCompactionRoute::Local => "local",
                ExactTailCompactionRoute::RemoteLegacy => "remote_legacy",
                ExactTailCompactionRoute::RemoteV2 => "remote_v2",
                ExactTailCompactionRoute::SemanticTranscript => "semantic_transcript",
            }
            .to_string(),
            installed_hot_suffix_item_count: event.installed_hot_suffix_item_count,
        })
    }

    fn hot_suffix_from_replacement_history<'a>(
        &self,
        replacement_history: &'a [ResponseItem],
    ) -> Option<&'a [ResponseItem]> {
        if let Some(item_count) = self.installed_hot_suffix_item_count {
            let start = replacement_history.len().checked_sub(item_count)?;
            return replacement_history.get(start..);
        }

        let compaction_index = replacement_history
            .iter()
            .rposition(|item| matches!(item, ResponseItem::Compaction { .. }))?;
        replacement_history.get(compaction_index + 1..)
    }
}

fn turn_ids_are_compatible(active_turn_id: Option<&str>, item_turn_id: Option<&str>) -> bool {
    active_turn_id
        .is_none_or(|turn_id| item_turn_id.is_none_or(|item_turn_id| item_turn_id == turn_id))
}

fn finalize_active_segment<'a>(
    active_segment: ActiveReplaySegment<'a>,
    state: &mut RolloutReconstructionState<'a>,
) {
    // Thread rollback drops the newest surviving real user-message boundaries. In replay, that
    // means skipping the next finalized segments that contain a non-contextual
    // `EventMsg::UserMessage`.
    if state.pending_rollback_turns > 0 {
        if active_segment.counts_as_user_turn {
            state.pending_rollback_turns -= 1;
        }
        return;
    }

    // A surviving replacement-history checkpoint is a complete history base. Once we
    // know the newest surviving one, older rollout items do not affect rebuilt history.
    if state.base_replacement_history.is_none()
        && let Some(segment_base_replacement_history) = active_segment.base_replacement_history
    {
        state.base_replacement_history = Some(segment_base_replacement_history);
        state.exact_tail_compaction_source = active_segment.exact_tail_compaction_source;
    }

    if state.window.is_none() {
        state.window = active_segment.window;
    }

    // `previous_turn_settings` come from the newest surviving user turn that established them.
    if state.previous_turn_settings.is_none() && active_segment.counts_as_user_turn {
        state.previous_turn_settings = active_segment.previous_turn_settings;
    }

    // `reference_context_item` comes from the newest surviving user turn baseline, or
    // from a surviving compaction that explicitly cleared that baseline.
    if matches!(
        state.reference_context_item,
        TurnReferenceContextItem::NeverSet
    ) && (active_segment.counts_as_user_turn
        || matches!(
            active_segment.reference_context_item,
            TurnReferenceContextItem::Cleared
        ))
    {
        state.reference_context_item = active_segment.reference_context_item;
    }
}

impl Session {
    pub(super) async fn reconstruct_history_from_rollout(
        &self,
        turn_context: &TurnContext,
        rollout_items: &[RolloutItem],
    ) -> RolloutReconstruction {
        // Replay metadata should already match the shape of the future lazy reverse loader, even
        // while history materialization still uses an eager bridge. Scan newest-to-oldest,
        // stopping once a surviving replacement-history checkpoint and the required resume metadata
        // are both known; then replay only the buffered surviving tail forward to preserve exact
        // history semantics.
        let mut state = RolloutReconstructionState::default();
        // Borrowed suffix of rollout items newer than the newest surviving replacement-history
        // checkpoint. If no such checkpoint exists, this remains the full rollout.
        let mut rollout_suffix = rollout_items;
        // Reverse replay accumulates rollout items into the newest in-progress turn segment until
        // we hit its matching `TurnStarted`, at which point the segment can be finalized.
        let mut active_segment: Option<ActiveReplaySegment<'_>> = None;

        for (index, item) in rollout_items.iter().enumerate().rev() {
            match item {
                RolloutItem::Compacted(compacted) => {
                    let active_segment =
                        active_segment.get_or_insert_with(ActiveReplaySegment::default);
                    if active_segment.window.is_none()
                        && let Some(window_number) = compacted.window_number
                    {
                        active_segment.window = Some(ReconstructedWindow {
                            number: window_number,
                            first_id: compacted.first_window_id.as_deref().and_then(parse_uuid_v7),
                            previous_id: compacted
                                .previous_window_id
                                .as_deref()
                                .and_then(parse_uuid_v7),
                            id: compacted.window_id.as_deref().and_then(parse_uuid_v7),
                        });
                    }
                    // Looking backward, compaction clears any older baseline unless a newer
                    // `TurnContextItem` in this same segment has already re-established it.
                    if matches!(
                        active_segment.reference_context_item,
                        TurnReferenceContextItem::NeverSet
                    ) {
                        active_segment.reference_context_item = TurnReferenceContextItem::Cleared;
                    }
                    if active_segment.base_replacement_history.is_none()
                        && let Some(replacement_history) = &compacted.replacement_history
                    {
                        active_segment.base_replacement_history = Some(replacement_history);
                        rollout_suffix = &rollout_items[index + 1..];
                    }
                }
                RolloutItem::EventMsg(EventMsg::ThreadRolledBack(rollback)) => {
                    state.pending_rollback_turns = state
                        .pending_rollback_turns
                        .saturating_add(usize::try_from(rollback.num_turns).unwrap_or(usize::MAX));
                }
                RolloutItem::EventMsg(EventMsg::ExactTailCompactionDiagnostic(event)) => {
                    let active_segment =
                        active_segment.get_or_insert_with(ActiveReplaySegment::default);
                    if active_segment.exact_tail_compaction_source.is_none() {
                        active_segment.exact_tail_compaction_source =
                            RecoveredExactTailCompactionSource::from_success_diagnostic(
                                event.as_ref(),
                            );
                    }
                }
                RolloutItem::EventMsg(EventMsg::ExactTailToolSurfaceDiagnostic(_)) => {}
                RolloutItem::EventMsg(EventMsg::TurnComplete(event)) => {
                    let active_segment =
                        active_segment.get_or_insert_with(ActiveReplaySegment::default);
                    // Reverse replay often sees `TurnComplete` before any turn-scoped metadata.
                    // Capture the turn id early so later `TurnContext` / abort items can match it.
                    if active_segment.turn_id.is_none() {
                        active_segment.turn_id = Some(event.turn_id.clone());
                    }
                }
                RolloutItem::EventMsg(EventMsg::TurnAborted(event)) => {
                    if let Some(active_segment) = active_segment.as_mut() {
                        if active_segment.turn_id.is_none()
                            && let Some(turn_id) = &event.turn_id
                        {
                            active_segment.turn_id = Some(turn_id.clone());
                        }
                    } else if let Some(turn_id) = &event.turn_id {
                        active_segment = Some(ActiveReplaySegment {
                            turn_id: Some(turn_id.clone()),
                            ..Default::default()
                        });
                    }
                }
                RolloutItem::EventMsg(EventMsg::UserMessage(_)) => {
                    let active_segment =
                        active_segment.get_or_insert_with(ActiveReplaySegment::default);
                    active_segment.counts_as_user_turn = true;
                }
                RolloutItem::TurnContext(ctx) => {
                    let active_segment =
                        active_segment.get_or_insert_with(ActiveReplaySegment::default);
                    // `TurnContextItem` can attach metadata to an existing segment, but only a
                    // real `UserMessage` event should make the segment count as a user turn.
                    if active_segment.turn_id.is_none() {
                        active_segment.turn_id = ctx.turn_id.clone();
                    }
                    if turn_ids_are_compatible(
                        active_segment.turn_id.as_deref(),
                        ctx.turn_id.as_deref(),
                    ) {
                        active_segment.previous_turn_settings = Some(PreviousTurnSettings {
                            model: ctx.model.clone(),
                            comp_hash: ctx.comp_hash.clone(),
                            realtime_active: ctx.realtime_active,
                        });
                        if matches!(
                            active_segment.reference_context_item,
                            TurnReferenceContextItem::NeverSet
                        ) {
                            active_segment.reference_context_item =
                                TurnReferenceContextItem::Latest(Box::new(ctx.clone()));
                        }
                    }
                }
                RolloutItem::EventMsg(EventMsg::TurnStarted(event)) => {
                    // `TurnStarted` is the oldest boundary of the active reverse segment.
                    if active_segment.as_ref().is_some_and(|active_segment| {
                        turn_ids_are_compatible(
                            active_segment.turn_id.as_deref(),
                            Some(event.turn_id.as_str()),
                        )
                    }) && let Some(active_segment) = active_segment.take()
                    {
                        finalize_active_segment(active_segment, &mut state);
                    }
                }
                RolloutItem::ResponseItem(response_item) => {
                    let active_segment =
                        active_segment.get_or_insert_with(ActiveReplaySegment::default);
                    active_segment.counts_as_user_turn |= is_user_turn_boundary(response_item);
                }
                RolloutItem::InterAgentCommunication(_) => {
                    let active_segment =
                        active_segment.get_or_insert_with(ActiveReplaySegment::default);
                    active_segment.counts_as_user_turn = true;
                }
                RolloutItem::EventMsg(_) | RolloutItem::SessionMeta(_) => {}
            }

            if state.base_replacement_history.is_some()
                && state.previous_turn_settings.is_some()
                && !matches!(
                    state.reference_context_item,
                    TurnReferenceContextItem::NeverSet
                )
            {
                // At this point we have both eager resume metadata values and the replacement-
                // history base for the surviving tail, so older rollout items cannot affect this
                // result.
                break;
            }
        }

        if let Some(active_segment) = active_segment.take() {
            finalize_active_segment(active_segment, &mut state);
        }

        let fallback_window_number = u64::try_from(
            rollout_items
                .iter()
                .filter(|item| matches!(item, RolloutItem::Compacted(_)))
                .count(),
        )
        .unwrap_or(u64::MAX);

        let mut history = ContextManager::new();
        let mut saw_legacy_compaction_without_replacement_history = false;
        let recovered_exact_tail_compaction_source = state.exact_tail_compaction_source.as_ref();
        let pending_exact_tail_tool_surface_hint =
            state
                .base_replacement_history
                .and_then(|base_replacement_history| {
                    recover_pending_exact_tail_tool_surface_hint(
                        base_replacement_history,
                        recovered_exact_tail_compaction_source?,
                    )
                });
        if let Some(base_replacement_history) = state.base_replacement_history {
            history.replace(base_replacement_history.to_vec());
        }
        // Materialize exact history semantics from the replay-derived suffix. The eventual lazy
        // design should keep this same replay shape, but drive it from a resumable reverse source
        // instead of an eagerly loaded `&[RolloutItem]`.
        for item in rollout_suffix {
            match item {
                RolloutItem::ResponseItem(response_item) => {
                    history.record_items(
                        std::iter::once(response_item),
                        turn_context.model_info.truncation_policy.into(),
                    );
                }
                RolloutItem::InterAgentCommunication(communication) => {
                    let response_item = communication.to_model_input_item();
                    history.record_items(
                        std::iter::once(&response_item),
                        turn_context.model_info.truncation_policy.into(),
                    );
                }
                RolloutItem::Compacted(compacted) => {
                    if let Some(replacement_history) = &compacted.replacement_history {
                        // This should actually never happen, because the reverse loop above (to build rollout_suffix)
                        // should stop before any compaction that has Some replacement_history
                        history.replace(replacement_history.clone());
                    } else {
                        saw_legacy_compaction_without_replacement_history = true;
                        // Legacy rollouts without `replacement_history` should rebuild the
                        // historical TurnContext at the correct insertion point from persisted
                        // `TurnContextItem`s. These are rare enough that we currently just clear
                        // `reference_context_item`, reinject canonical context at the end of the
                        // resumed conversation, and accept the temporary out-of-distribution
                        // prompt shape.
                        // TODO(ccunningham): if we drop support for None replacement_history compaction items,
                        // we can get rid of this second loop entirely and just build `history` directly in the first loop.
                        let user_messages = compact::collect_user_messages(history.raw_items());
                        let rebuilt = compact::build_compacted_history(
                            Vec::new(),
                            &user_messages,
                            &compacted.message,
                        );
                        history.replace(rebuilt);
                    }
                }
                RolloutItem::EventMsg(EventMsg::ThreadRolledBack(rollback)) => {
                    history.drop_last_n_user_turns(rollback.num_turns);
                }
                RolloutItem::EventMsg(_)
                | RolloutItem::TurnContext(_)
                | RolloutItem::SessionMeta(_) => {}
            }
        }

        let reference_context_item = match state.reference_context_item {
            TurnReferenceContextItem::NeverSet | TurnReferenceContextItem::Cleared => None,
            TurnReferenceContextItem::Latest(turn_reference_context_item) => {
                Some(*turn_reference_context_item)
            }
        };
        let reference_context_item = if saw_legacy_compaction_without_replacement_history {
            None
        } else {
            reference_context_item
        };

        let window = state.window.unwrap_or(ReconstructedWindow {
            number: fallback_window_number,
            first_id: None,
            previous_id: None,
            id: None,
        });
        RolloutReconstruction {
            history: history.raw_items().to_vec(),
            previous_turn_settings: state.previous_turn_settings,
            reference_context_item,
            window_number: window.number,
            first_window_id: window.first_id,
            previous_window_id: window.previous_id,
            window_id: window.id,
            pending_exact_tail_tool_surface_hint,
        }
    }
}

fn recover_pending_exact_tail_tool_surface_hint(
    replacement_history: &[ResponseItem],
    source: &RecoveredExactTailCompactionSource,
) -> Option<PendingExactTailToolSurfaceHint> {
    let hot_suffix = source.hot_suffix_from_replacement_history(replacement_history)?;
    let hint = derive_exact_tail_tool_surface_hint(hot_suffix);
    exact_tail_tool_surface_hint_has_signal(&hint)
        .then(|| PendingExactTailToolSurfaceHint::new(&source.compaction_id, &source.route, hint))
}

fn exact_tail_tool_surface_hint_has_signal(hint: &ExactTailToolSurfaceHint) -> bool {
    !hint.references.is_empty()
        || hint.hot_tool_call_count > 0
        || hint.hot_tool_reference_overflow_count > 0
        || hint.out_of_scope_dependency_protocol_count > 0
}

fn parse_uuid_v7(value: &str) -> Option<Uuid> {
    Uuid::parse_str(value)
        .ok()
        .filter(|uuid| uuid.get_version_num() == 7)
}
