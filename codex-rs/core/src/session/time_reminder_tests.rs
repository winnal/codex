use super::*;
use codex_protocol::models::ContentItem;

#[test]
fn context_shaped_direct_source_enables_after_user_reminder() {
    let now = Utc::now();
    let mode = CurrentTimeReminderDeliveryMode::AfterUserOrToolOutput;
    let item = ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: "<environment_context>literal source</environment_context>".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    };
    let mut state = CurrentTimeReminderState::default();
    assert!(state.take_reminder_due("window", now, 0, mode));

    state.note_recorded_items(std::slice::from_ref(&item), HistoryItemProvenance::Other);
    assert!(!state.take_reminder_due("window", now, 0, mode));

    state.note_recorded_items(
        std::slice::from_ref(&item),
        HistoryItemProvenance::DirectUserSource,
    );
    assert!(state.take_reminder_due("window", now, 0, mode));
}
