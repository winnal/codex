use super::ContextualUserFragment;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExactTailToolSurfaceNotice {
    missing_hot_tool_count: usize,
    hot_tool_reference_overflow_count: usize,
    missing_no_path_count: usize,
}

impl ExactTailToolSurfaceNotice {
    pub(crate) fn new(
        missing_hot_tool_count: usize,
        hot_tool_reference_overflow_count: usize,
        missing_no_path_count: usize,
    ) -> Self {
        Self {
            missing_hot_tool_count,
            hot_tool_reference_overflow_count,
            missing_no_path_count,
        }
    }
}

impl ContextualUserFragment for ExactTailToolSurfaceNotice {
    fn role(&self) -> &'static str {
        "user"
    }

    fn markers(&self) -> (&'static str, &'static str) {
        Self::type_markers()
    }

    fn type_markers() -> (&'static str, &'static str) {
        (
            "<exact_tail_tool_surface_notice>",
            "</exact_tail_tool_surface_notice>",
        )
    }

    fn body(&self) -> String {
        let mut lines = Vec::new();
        if self.missing_hot_tool_count > 0 {
            let missing = self.missing_hot_tool_count;
            lines.push(format!(
                "{missing} tool reference(s) from the preserved exact-tail history are not currently available as direct callable tools."
            ));
        }
        if self.hot_tool_reference_overflow_count > 0 {
            let overflow = self.hot_tool_reference_overflow_count;
            lines.push(format!(
                "{overflow} older exact-tail tool reference(s) were beyond the continuity hint cap and were not rehydrated."
            ));
        }
        let missing_with_discovery_path = self
            .missing_hot_tool_count
            .saturating_sub(self.missing_no_path_count);
        if missing_with_discovery_path > 0 {
            lines
                .push("Rediscover still-available deferred tools before calling them.".to_string());
        } else {
            lines.push("Do not assume missing historical tools remain callable.".to_string());
        }

        format!("\n{}\n", lines.join("\n"))
    }
}
