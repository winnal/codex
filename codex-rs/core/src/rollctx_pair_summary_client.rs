use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use futures::StreamExt;

use crate::client_common::ResponseEvent;
use crate::client_common::ResponseStream;

pub(crate) async fn collect_rollctx_pair_summary_text(
    mut stream: ResponseStream,
) -> Result<String> {
    let mut saw_completed = false;
    let mut delta_text = String::new();
    let mut final_message_text = String::new();

    while let Some(event) = stream.next().await {
        match event? {
            ResponseEvent::OutputTextDelta(delta) => delta_text.push_str(&delta),
            ResponseEvent::OutputItemAdded(ResponseItem::Message { .. })
            | ResponseEvent::OutputItemAdded(ResponseItem::Reasoning { .. })
            | ResponseEvent::OutputItemDone(ResponseItem::Reasoning { .. }) => {}
            ResponseEvent::OutputItemDone(ResponseItem::Message { role, content, .. })
                if role == "assistant" =>
            {
                for item in content {
                    match item {
                        ContentItem::OutputText { text } | ContentItem::InputText { text } => {
                            final_message_text.push_str(&text);
                        }
                        ContentItem::InputImage { .. } => {}
                    }
                }
            }
            ResponseEvent::Completed { .. } => {
                saw_completed = true;
                break;
            }
            ResponseEvent::OutputItemAdded(item) | ResponseEvent::OutputItemDone(item) => {
                return Err(CodexErr::Fatal(format!(
                    "rollctx pair-summary received unexpected output item: {item:?}"
                )));
            }
            ResponseEvent::Created
            | ResponseEvent::ServerModel(_)
            | ResponseEvent::ModelVerifications(_)
            | ResponseEvent::TurnModerationMetadata(_)
            | ResponseEvent::ServerReasoningIncluded(_)
            | ResponseEvent::ToolCallInputDelta { .. }
            | ResponseEvent::ReasoningSummaryDelta { .. }
            | ResponseEvent::ReasoningContentDelta { .. }
            | ResponseEvent::ReasoningSummaryPartAdded { .. }
            | ResponseEvent::RateLimits(_)
            | ResponseEvent::ModelsEtag(_) => {}
        }
    }

    if !saw_completed {
        return Err(CodexErr::Stream(
            "rollctx pair-summary stream closed before response.completed".to_string(),
            None,
        ));
    }

    let text = if delta_text.is_empty() {
        final_message_text
    } else {
        delta_text
    };
    if text.trim().is_empty() {
        return Err(CodexErr::Fatal(
            "rollctx pair-summary produced empty assistant text".to_string(),
        ));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::models::ResponseItem;
    use pretty_assertions::assert_eq;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    fn response_stream(events: Vec<Result<ResponseEvent>>) -> ResponseStream {
        let (tx, rx_event) = mpsc::channel(events.len().max(1));
        for event in events {
            tx.try_send(event)
                .expect("test stream channel should accept event");
        }
        drop(tx);
        ResponseStream {
            rx_event,
            consumer_dropped: CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn returns_assistant_text() {
        let text = collect_rollctx_pair_summary_text(response_stream(vec![
            Ok(ResponseEvent::Created),
            Ok(ResponseEvent::OutputTextDelta("standing facts".to_string())),
            Ok(ResponseEvent::Completed {
                response_id: "summary-response".to_string(),
                token_usage: None,
                end_turn: Some(true),
            }),
        ]))
        .await
        .expect("summary text should be collected");

        assert_eq!(text, "standing facts");
    }

    #[tokio::test]
    async fn rejects_tool_items() {
        let err = collect_rollctx_pair_summary_text(response_stream(vec![
            Ok(ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
                id: None,
                name: "shell_command".to_string(),
                namespace: None,
                arguments: "{}".to_string(),
                call_id: "call-1".to_string(),
            })),
            Ok(ResponseEvent::Completed {
                response_id: "summary-response".to_string(),
                token_usage: None,
                end_turn: Some(true),
            }),
        ]))
        .await
        .expect_err("tool items must be rejected");

        assert!(
            err.to_string()
                .contains("rollctx pair-summary received unexpected output item"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn rejects_stream_without_completion() {
        let err = collect_rollctx_pair_summary_text(response_stream(vec![Ok(
            ResponseEvent::OutputTextDelta("partial".to_string()),
        )]))
        .await
        .expect_err("missing response.completed must fail");

        assert!(
            err.to_string()
                .contains("stream closed before response.completed"),
            "unexpected error: {err}"
        );
    }
}
