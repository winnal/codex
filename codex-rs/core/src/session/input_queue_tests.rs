use super::*;
use codex_protocol::AgentPath;
use codex_protocol::models::FunctionCallOutputPayload;
use pretty_assertions::assert_eq;

fn make_mail(
    author: AgentPath,
    recipient: AgentPath,
    content: &str,
    trigger_turn: bool,
) -> InterAgentCommunication {
    InterAgentCommunication::new(
        author,
        recipient,
        Vec::new(),
        content.to_string(),
        trigger_turn,
    )
}

#[tokio::test]
async fn input_queue_notifies_mailbox_subscribers() {
    let input_queue = InputQueue::new();
    let mut mailbox_rx = input_queue.subscribe_mailbox().await;

    input_queue
        .enqueue_mailbox_communication(make_mail(
            AgentPath::root(),
            AgentPath::try_from("/root/worker").expect("agent path"),
            "one",
            /*trigger_turn*/ false,
        ))
        .await;
    input_queue
        .enqueue_mailbox_communication(make_mail(
            AgentPath::root(),
            AgentPath::try_from("/root/worker").expect("agent path"),
            "two",
            /*trigger_turn*/ false,
        ))
        .await;

    mailbox_rx.changed().await.expect("mailbox update");
}

#[tokio::test]
async fn input_queue_drains_mailbox_in_delivery_order() {
    let input_queue = InputQueue::new();
    let mail_one = make_mail(
        AgentPath::root(),
        AgentPath::try_from("/root/worker").expect("agent path"),
        "one",
        /*trigger_turn*/ false,
    );
    let mail_two = make_mail(
        AgentPath::try_from("/root/worker").expect("agent path"),
        AgentPath::root(),
        "two",
        /*trigger_turn*/ false,
    );

    input_queue
        .enqueue_mailbox_communication(mail_one.clone())
        .await;
    input_queue
        .enqueue_mailbox_communication(mail_two.clone())
        .await;

    assert_eq!(
        input_queue.drain_mailbox_input_items().await,
        vec![
            ResponseItem::from(mail_one.to_response_input_item()),
            ResponseItem::from(mail_two.to_response_input_item())
        ]
    );
    assert!(!input_queue.has_pending_mailbox_items().await);
}

#[tokio::test]
async fn input_queue_tracks_pending_trigger_turn_mail() {
    let input_queue = InputQueue::new();

    input_queue
        .enqueue_mailbox_communication(make_mail(
            AgentPath::root(),
            AgentPath::try_from("/root/worker").expect("agent path"),
            "queued",
            /*trigger_turn*/ false,
        ))
        .await;
    assert!(!input_queue.has_trigger_turn_mailbox_items().await);

    input_queue
        .enqueue_mailbox_communication(make_mail(
            AgentPath::root(),
            AgentPath::try_from("/root/worker").expect("agent path"),
            "wake",
            /*trigger_turn*/ true,
        ))
        .await;
    assert!(input_queue.has_trigger_turn_mailbox_items().await);
}

#[tokio::test]
async fn input_queue_drains_tool_continuations_without_draining_user_context_or_mailbox_input() {
    let input_queue = InputQueue::new();
    let active_turn = Mutex::new(Some(ActiveTurn::default()));
    let turn_state = {
        let active_turn = active_turn.lock().await;
        Arc::clone(&active_turn.as_ref().expect("active turn").turn_state)
    };
    let user_input = TurnInput::UserInput {
        content: vec![UserInput::Text {
            text: "queued user input".to_string(),
            text_elements: Vec::new(),
        }],
        client_id: None,
    };
    let context_input = TurnInput::ResponseItem(ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: Vec::new(),
        phase: None,
    });
    let tool_output = TurnInput::ResponseItem(ResponseItem::CustomToolCallOutput {
        call_id: "call-1".to_string(),
        name: Some("exec".to_string()),
        output: FunctionCallOutputPayload::from_text("notify".to_string()),
    });
    let mailbox = make_mail(
        AgentPath::root(),
        AgentPath::try_from("/root/worker").expect("agent path"),
        "mailbox input",
        /*trigger_turn*/ false,
    );

    input_queue
        .extend_pending_input_for_turn_state(
            turn_state.as_ref(),
            vec![
                user_input.clone(),
                context_input.clone(),
                tool_output.clone(),
            ],
        )
        .await;
    input_queue
        .enqueue_mailbox_communication(mailbox.clone())
        .await;

    assert_eq!(
        vec![tool_output],
        input_queue
            .get_pending_tool_continuation_items(&active_turn)
            .await
    );
    assert!(input_queue.has_pending_mailbox_items().await);
    assert_eq!(
        vec![
            user_input,
            context_input,
            TurnInput::ResponseItem(ResponseItem::from(mailbox.to_response_input_item())),
        ],
        input_queue.get_pending_input(&active_turn).await
    );
}
