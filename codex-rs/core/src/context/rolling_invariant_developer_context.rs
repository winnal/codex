use super::ContextualUserFragment;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RollingInvariantDeveloperContext;

impl ContextualUserFragment for RollingInvariantDeveloperContext {
    fn role(&self) -> &'static str {
        "developer"
    }

    fn markers(&self) -> (&'static str, &'static str) {
        Self::type_markers()
    }

    fn type_markers() -> (&'static str, &'static str) {
        (
            "<codex_rollctx_invariant_developer_context>",
            "</codex_rollctx_invariant_developer_context>",
        )
    }

    fn body(&self) -> String {
        "\n".to_string()
    }
}
