use codex_protocol::AgentPath;
use codex_protocol::ThreadId;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::ExactTailToolSurfaceDiagnosticEvent;
use codex_protocol::protocol::ExecCommandBeginEvent;
use codex_protocol::protocol::ExecCommandEndEvent;
use codex_protocol::protocol::ExecCommandSource;
use codex_protocol::protocol::ExecCommandStatus;
use codex_protocol::protocol::SubAgentActivityEvent;
use codex_protocol::protocol::SubAgentActivityKind;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::time::Duration;

use super::ToolRuntimeTraceEvent;
use super::tool_runtime_trace_event;
use super::wrapped_protocol_event_type;
use crate::ExecutionStatus;

fn exact_tail_tool_surface_diagnostic_event() -> EventMsg {
    EventMsg::ExactTailToolSurfaceDiagnostic(ExactTailToolSurfaceDiagnosticEvent {
        thread_id: "thread".into(),
        turn_id: "turn".into(),
        compaction_id: "compact".into(),
        route: "remote_v2".into(),
        hot_tool_call_count: 1,
        hot_tool_namespace_count: 1,
        hot_tool_reference_count: 1,
        rehydrated_tool_count: 1,
        missing_hot_tool_count: 0,
        already_direct_tool_count: 0,
        discoverable_hot_tool_count: 1,
        missing_notice_emitted_count: 0,
        missing_no_path_count: 0,
        rehydrated_tool_references: Vec::new(),
        missing_tool_references: Vec::new(),
        rehydrated_tool_namespaces: Vec::new(),
        missing_tool_rejection_reasons: Vec::new(),
        hot_tool_reference_overflow_count: 0,
        hot_tool_reference_overflow_notice_emitted_count: 0,
        out_of_scope_dependency_protocol_count: 0,
        tool_surface_changed_after_compaction: true,
        tool_surface_rehydration_failure_reason: None,
    })
}

#[test]
fn exact_tail_tool_surface_diagnostic_is_protocol_event_not_tool_runtime_event() {
    let tool_surface = exact_tail_tool_surface_diagnostic_event();

    assert_eq!(
        wrapped_protocol_event_type(&tool_surface),
        Some("exact_tail_tool_surface_diagnostic")
    );
    assert!(tool_runtime_trace_event(&tool_surface).is_none());
}

#[test]
fn sub_agent_activity_is_a_terminal_tool_runtime_event() -> anyhow::Result<()> {
    let agent_thread_id = ThreadId::new();
    let event = EventMsg::SubAgentActivity(SubAgentActivityEvent {
        event_id: "call-spawn".to_string(),
        occurred_at_ms: 1234,
        agent_thread_id,
        agent_path: AgentPath::try_from("/root/reviewer").map_err(anyhow::Error::msg)?,
        kind: SubAgentActivityKind::Started,
    });

    let Some(ToolRuntimeTraceEvent::Ended {
        tool_call_id,
        status,
        payload,
    }) = tool_runtime_trace_event(&event)
    else {
        panic!("expected terminal tool runtime event");
    };

    assert_eq!(tool_call_id, "call-spawn");
    assert_eq!(status, ExecutionStatus::Completed);
    assert_eq!(
        serde_json::to_value(payload)?,
        json!({
            "event_id": "call-spawn",
            "occurred_at_ms": 1234,
            "agent_thread_id": agent_thread_id,
            "agent_path": "/root/reviewer",
            "kind": "started"
        })
    );
    Ok(())
}

#[test]
fn exec_command_trace_payloads_use_inferred_native_cwd() -> anyhow::Result<()> {
    // Convention inference depends on the URI spelling, not the test host, so exercise both
    // Windows and POSIX paths on every platform.
    let begin = EventMsg::ExecCommandBegin(ExecCommandBeginEvent {
        call_id: "call-begin".to_string(),
        process_id: Some("process-1".to_string()),
        turn_id: "turn-1".to_string(),
        started_at_ms: 1234,
        command: vec!["pwd".to_string()],
        cwd: "file:///C:/windows".parse()?,
        parsed_cmd: Vec::new(),
        source: ExecCommandSource::Agent,
        interaction_input: None,
    });
    let end = EventMsg::ExecCommandEnd(ExecCommandEndEvent {
        call_id: "call-end".to_string(),
        process_id: None,
        turn_id: "turn-1".to_string(),
        completed_at_ms: 2345,
        command: vec!["pwd".to_string()],
        cwd: "file:///workspace/project".parse()?,
        parsed_cmd: Vec::new(),
        source: ExecCommandSource::UnifiedExecInteraction,
        interaction_input: Some("input".to_string()),
        stdout: "output".to_string(),
        stderr: String::new(),
        aggregated_output: "output".to_string(),
        exit_code: 0,
        duration: Duration::from_millis(250),
        formatted_output: "output".to_string(),
        status: ExecCommandStatus::Completed,
    });

    let Some(ToolRuntimeTraceEvent::Started { payload, .. }) = tool_runtime_trace_event(&begin)
    else {
        panic!("expected started tool runtime event");
    };
    assert_eq!(
        serde_json::to_value(payload)?,
        json!({
            "call_id": "call-begin",
            "process_id": "process-1",
            "turn_id": "turn-1",
            "started_at_ms": 1234,
            "command": ["pwd"],
            "cwd": r"C:\windows",
            "parsed_cmd": [],
            "source": "agent"
        })
    );

    let Some(ToolRuntimeTraceEvent::Ended { payload, .. }) = tool_runtime_trace_event(&end) else {
        panic!("expected ended tool runtime event");
    };
    assert_eq!(
        serde_json::to_value(payload)?,
        json!({
            "call_id": "call-end",
            "turn_id": "turn-1",
            "completed_at_ms": 2345,
            "command": ["pwd"],
            "cwd": "/workspace/project",
            "parsed_cmd": [],
            "source": "unified_exec_interaction",
            "interaction_input": "input",
            "stdout": "output",
            "stderr": "",
            "aggregated_output": "output",
            "exit_code": 0,
            "duration": {"secs": 0, "nanos": 250000000},
            "formatted_output": "output",
            "status": "completed"
        })
    );
    Ok(())
}
