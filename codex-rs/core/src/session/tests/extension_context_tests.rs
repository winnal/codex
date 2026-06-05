use super::*;

struct ContextualUserPromptExtensionTestContributor;
struct ContextualUserPromptExtensionTestState {
    text: String,
}

impl codex_extension_api::ContextContributor for ContextualUserPromptExtensionTestContributor {
    fn contribute<'a>(
        &'a self,
        _session_store: &'a codex_extension_api::ExtensionData,
        thread_store: &'a codex_extension_api::ExtensionData,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Vec<codex_extension_api::PromptFragment>> + Send + 'a>,
    > {
        Box::pin(async move {
            thread_store
                .get::<ContextualUserPromptExtensionTestState>()
                .map(|state| {
                    codex_extension_api::PromptFragment::new(
                        codex_extension_api::PromptSlot::ContextualUser,
                        state.text.clone(),
                    )
                })
                .into_iter()
                .collect()
        })
    }
}

fn contextual_user_prompt_extension_test_registry()
-> Arc<codex_extension_api::ExtensionRegistry<crate::config::Config>> {
    let mut builder = codex_extension_api::ExtensionRegistryBuilder::new();
    builder.prompt_contributor(Arc::new(ContextualUserPromptExtensionTestContributor));
    Arc::new(builder.build())
}

#[tokio::test]
async fn build_initial_context_preserves_large_contextual_user_prompt_fragment() {
    let (mut session, turn_context) = make_session_and_context().await;
    let source_text = format!(
        "{}TAIL_EXTENSION_CONTEXTUAL_USER_SENTINEL",
        "extension contextual user ".repeat(30_000)
    );
    session.services.extensions = contextual_user_prompt_extension_test_registry();
    session
        .services
        .thread_extension_data
        .insert(ContextualUserPromptExtensionTestState {
            text: source_text.clone(),
        });

    let initial_context = session.build_initial_context(&turn_context).await;
    let user_texts = user_input_texts(&initial_context);
    let rendered = user_texts
        .iter()
        .find(|text| text.contains("<extension_contextual_user>"))
        .expect("extension contextual-user fragment should render as user context");

    assert!(
        rendered.contains(&source_text),
        "extension contextual-user rendering must preserve the full source text"
    );
    assert!(
        !rendered.contains("tokens truncated"),
        "extension contextual-user rendering must not truncate before rolling projection"
    );
}
