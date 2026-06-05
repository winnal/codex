use super::ContextualUserFragment;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExtensionContextualUserFragment {
    text: String,
}

impl ExtensionContextualUserFragment {
    pub(crate) fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

impl ContextualUserFragment for ExtensionContextualUserFragment {
    fn role(&self) -> &'static str {
        "user"
    }

    fn markers(&self) -> (&'static str, &'static str) {
        Self::type_markers()
    }

    fn type_markers() -> (&'static str, &'static str) {
        (
            "<extension_contextual_user>",
            "</extension_contextual_user>",
        )
    }

    fn body(&self) -> String {
        format!("\n{}\n", self.text)
    }
}
