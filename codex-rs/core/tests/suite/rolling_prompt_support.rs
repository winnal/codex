use anyhow::Result;
use codex_core::CodexThread;
use codex_protocol::protocol::EventMsg;
use core_test_support::responses;
use serde_json::Value;

pub(super) fn turn_metadata(request: &responses::ResponsesRequest) -> Result<Value> {
    let Some(metadata) = request.header("x-codex-turn-metadata") else {
        anyhow::bail!("request should include turn metadata");
    };
    Ok(serde_json::from_str(&metadata)?)
}

pub(super) async fn wait_for_turn_complete(codex: &CodexThread, turn_id: &str) -> Result<()> {
    loop {
        let event = codex.next_event().await?;
        if event.id != turn_id {
            continue;
        }
        match event.msg {
            EventMsg::TurnComplete(_) => return Ok(()),
            EventMsg::TurnAborted(_) => anyhow::bail!("turn {turn_id} aborted"),
            EventMsg::Error(error) => anyhow::bail!("turn {turn_id} failed: {}", error.message),
            _ => {}
        }
    }
}
