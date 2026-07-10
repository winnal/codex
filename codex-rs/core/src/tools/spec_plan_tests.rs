use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use codex_features::Feature;
use codex_login::AuthManager;
use codex_login::CodexAuth;
use codex_mcp::ToolInfo;
use codex_model_provider::ModelProvider;
use codex_model_provider::ModelProviderFuture;
use codex_model_provider::ProviderAccountResult;
use codex_model_provider::ProviderAccountState;
use codex_model_provider::ProviderCapabilities;
use codex_model_provider::SharedModelProvider;
use codex_model_provider::create_model_provider;
use codex_model_provider_info::AMAZON_BEDROCK_PROVIDER_ID;
use codex_model_provider_info::ModelProviderInfo;
use codex_models_manager::manager::SharedModelsManager;
use codex_models_manager::manager::StaticModelsManager;
use codex_protocol::config_types::WebSearchMode;
use codex_protocol::dynamic_tools::DynamicToolSpec;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ApplyPatchToolType;
use codex_protocol::openai_models::ConfigShellToolType;
use codex_protocol::openai_models::InputModality;
use codex_protocol::openai_models::ModelsResponse;
use codex_protocol::openai_models::ToolMode;
use codex_protocol::openai_models::WebSearchToolType;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentSource;
use codex_tools::DiscoverablePluginInfo;
use codex_tools::DiscoverableTool;
use codex_tools::ResponsesApiNamespace;
use codex_tools::ResponsesApiNamespaceTool;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolCall as ExtensionToolCall;
use codex_tools::ToolExecutor;
use codex_tools::ToolExposure;
use codex_tools::ToolName;
use codex_tools::ToolOutput;
use codex_tools::ToolSearchInfo;
use codex_tools::ToolSpec;
use pretty_assertions::assert_eq;
use serde_json::json;

use crate::config::CurrentTimeReminderConfig;
use crate::session::step_context::StepContext;
use crate::session::tests::make_session_and_context;
use crate::session::turn_context::TurnContext;
use crate::tools::exact_tail_continuity::ExactTailToolSurfaceHint;
use crate::tools::exact_tail_continuity::ExactTailToolSurfaceOutcome;
use crate::tools::exact_tail_continuity::PendingExactTailToolSurfaceHint;
use crate::tools::exact_tail_continuity::derive_exact_tail_tool_surface_hint;
use crate::tools::exact_tail_continuity::exact_tail_tool_surface_notice_item;
use crate::tools::handlers::ToolSearchHandlerCache;
use crate::tools::handlers::multi_agents_spec::MULTI_AGENT_V1_NAMESPACE;
use crate::tools::router::ToolRouter;
use crate::tools::router::ToolRouterParams;
use crate::tools::router::ToolSuggestCandidates;
use crate::tools::router::ToolSuggestPresentation;

use super::EXACT_TAIL_PROMOTED_TOOL_SPEC_TOKEN_LIMIT;
use super::EXACT_TAIL_PROMOTED_TOOL_SPECS_TOTAL_TOKEN_LIMIT;
const MULTI_AGENT_V2_NAMESPACE: &str = "collaboration";

#[derive(Default)]
struct ToolPlanInputs {
    mcp_tools: Option<Vec<ToolInfo>>,
    deferred_mcp_tools: Option<Vec<ToolInfo>>,
    tool_suggest_candidates: Option<ToolSuggestCandidates>,
    extension_tool_executors: Vec<Arc<dyn ToolExecutor<ExtensionToolCall>>>,
    dynamic_tools: Vec<DynamicToolSpec>,
    exact_tail_tool_surface_hint: Option<PendingExactTailToolSurfaceHint>,
}

struct ToolPlanProbe {
    visible_specs: Vec<ToolSpec>,
    visible_names: Vec<String>,
    namespace_functions: BTreeMap<String, Vec<String>>,
    registered_names: Vec<String>,
    exposures: BTreeMap<String, ToolExposure>,
    exact_tail_tool_surface_outcome: Option<ExactTailToolSurfaceOutcome>,
}

#[derive(Debug)]
struct CapabilityProvider {
    info: ModelProviderInfo,
    capabilities: ProviderCapabilities,
}

impl ModelProvider for CapabilityProvider {
    fn info(&self) -> &ModelProviderInfo {
        &self.info
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities
    }

    fn auth_manager(&self) -> Option<Arc<AuthManager>> {
        None
    }

    fn auth(&self) -> ModelProviderFuture<'_, Option<CodexAuth>> {
        Box::pin(async { None })
    }

    fn account_state(&self) -> ProviderAccountResult {
        Ok(ProviderAccountState {
            account: None,
            requires_openai_auth: false,
        })
    }

    fn models_manager(
        &self,
        _codex_home: PathBuf,
        config_model_catalog: Option<ModelsResponse>,
    ) -> SharedModelsManager {
        Arc::new(StaticModelsManager::new(
            None,
            config_model_catalog.unwrap_or_default(),
        ))
    }
}

impl ToolPlanProbe {
    fn from_router(router: ToolRouter) -> Self {
        let visible_specs = router.model_visible_specs();
        let visible_names = visible_specs
            .iter()
            .map(|spec| spec.name().to_string())
            .collect::<Vec<_>>();
        let namespace_functions = visible_specs
            .iter()
            .filter_map(|spec| match spec {
                ToolSpec::Namespace(namespace) => Some((
                    namespace.name.clone(),
                    namespace
                        .tools
                        .iter()
                        .map(|tool| match tool {
                            ResponsesApiNamespaceTool::Function(tool) => tool.name.clone(),
                        })
                        .collect::<Vec<_>>(),
                )),
                ToolSpec::Function(_)
                | ToolSpec::ToolSearch { .. }
                | ToolSpec::WebSearch { .. }
                | ToolSpec::Freeform(_) => None,
            })
            .collect::<BTreeMap<_, _>>();
        let registered_tool_names = router.registered_tool_names_for_test();
        let registered_names = registered_tool_names
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        let exposures = registered_tool_names
            .iter()
            .filter_map(|name| {
                router
                    .tool_exposure_for_test(name)
                    .map(|exposure| (name.to_string(), exposure))
            })
            .collect::<BTreeMap<_, _>>();

        Self {
            visible_specs,
            visible_names,
            namespace_functions,
            registered_names,
            exposures,
            exact_tail_tool_surface_outcome: router.exact_tail_tool_surface_outcome().cloned(),
        }
    }

    fn assert_visible_contains(&self, expected: &[&str]) {
        for name in expected {
            assert!(
                self.visible_names.iter().any(|visible| visible == name),
                "expected visible tool `{name}` in {:?}",
                self.visible_names
            );
        }
    }

    fn assert_visible_lacks(&self, expected_absent: &[&str]) {
        for name in expected_absent {
            assert!(
                !self.visible_names.iter().any(|visible| visible == name),
                "expected visible tool `{name}` to be absent from {:?}",
                self.visible_names
            );
        }
    }

    fn assert_registered_contains(&self, expected: &[&str]) {
        for name in expected {
            assert!(
                self.registered_names
                    .iter()
                    .any(|registered| registered == name),
                "expected registered tool `{name}` in {:?}",
                self.registered_names
            );
        }
    }

    fn assert_registered_lacks(&self, expected_absent: &[&str]) {
        for name in expected_absent {
            assert!(
                !self
                    .registered_names
                    .iter()
                    .any(|registered| registered == name),
                "expected registered tool `{name}` to be absent from {:?}",
                self.registered_names
            );
        }
    }

    fn namespace_function_names(&self, namespace: &str) -> &[String] {
        self.namespace_functions
            .get(namespace)
            .map_or(&[], Vec::as_slice)
    }

    fn visible_spec(&self, name: &str) -> &ToolSpec {
        self.visible_specs
            .iter()
            .find(|spec| spec.name() == name)
            .unwrap_or_else(|| panic!("expected visible spec `{name}` in {:?}", self.visible_names))
    }

    fn exposure(&self, name: &str) -> ToolExposure {
        *self
            .exposures
            .get(name)
            .unwrap_or_else(|| panic!("expected registered tool `{name}`"))
    }
}

async fn probe_with(
    configure_turn: impl FnOnce(&mut TurnContext),
    inputs: ToolPlanInputs,
) -> ToolPlanProbe {
    let (_session, mut turn) = make_session_and_context().await;
    configure_turn(&mut turn);
    let turn = Arc::new(turn);
    let step_context = StepContext::for_test(Arc::clone(&turn));
    let router = ToolRouter::from_context(
        step_context.as_ref(),
        ToolRouterParams {
            tool_suggest_candidates: inputs.tool_suggest_candidates,
            mcp_tools: inputs.mcp_tools,
            deferred_mcp_tools: inputs.deferred_mcp_tools,
            extension_tool_executors: inputs.extension_tool_executors,
            dynamic_tools: inputs.dynamic_tools.as_slice(),
            exact_tail_tool_surface_hint: inputs.exact_tail_tool_surface_hint,
        },
        &Default::default(),
    );
    ToolPlanProbe::from_router(router)
}

async fn probe(configure_turn: impl FnOnce(&mut TurnContext)) -> ToolPlanProbe {
    probe_with(configure_turn, ToolPlanInputs::default()).await
}

fn set_feature(turn: &mut TurnContext, feature: Feature, enabled: bool) {
    let mut config = (*turn.config).clone();
    if enabled {
        config
            .features
            .enable(feature)
            .expect("test feature should be enableable in config");
    } else {
        config
            .features
            .disable(feature)
            .expect("test feature should be disableable in config");
    }
    turn.multi_agent_version = config.multi_agent_version_from_features();
    turn.config = Arc::new(config);
}

fn set_features(turn: &mut TurnContext, features: &[Feature]) {
    for feature in features {
        set_feature(turn, *feature, /*enabled*/ true);
    }
}

fn zsh_fork_config_for_spec_plan_tests() -> codex_tools::ZshForkConfig {
    let placeholder_exe = codex_utils_absolute_path::AbsolutePathBuf::try_from(
        std::env::current_exe().expect("current exe path"),
    )
    .expect("current exe should be absolute");

    // Spec planning only checks whether the shell mode is ZshFork. These paths
    // are never executed, so use a stable absolute placeholder instead of
    // depending on packaged zsh-fork artifacts in schema tests.
    codex_tools::ZshForkConfig {
        shell_zsh_path: placeholder_exe.clone(),
        main_execve_wrapper_exe: placeholder_exe,
    }
}

fn update_config(turn: &mut TurnContext, update: impl FnOnce(&mut crate::config::Config)) {
    let mut config = (*turn.config).clone();
    update(&mut config);
    turn.config = Arc::new(config);
}

fn set_web_search_mode(turn: &mut TurnContext, mode: WebSearchMode) {
    update_config(turn, |config| {
        config
            .web_search_mode
            .set(mode)
            .expect("test web search mode should be accepted");
    });
}

fn use_chatgpt_auth(turn: &mut TurnContext) {
    turn.auth_manager = Some(AuthManager::from_auth_for_testing(
        CodexAuth::create_dummy_chatgpt_auth_for_testing(),
    ));
    turn.provider = create_model_provider(
        turn.config.model_provider.clone(),
        turn.auth_manager.clone(),
    );
}

fn use_bedrock_provider(turn: &mut TurnContext) {
    let provider_info = ModelProviderInfo::create_amazon_bedrock_provider(/*aws*/ None);
    update_config(turn, |config| {
        config.model_provider_id = AMAZON_BEDROCK_PROVIDER_ID.to_string();
        config.model_provider = provider_info.clone();
    });
    turn.provider = create_model_provider(provider_info, turn.auth_manager.clone());
}

struct TestNamespaceExtensionTool {
    namespace: &'static str,
    tool_name: &'static str,
}

impl ToolExecutor<ExtensionToolCall> for TestNamespaceExtensionTool {
    fn tool_name(&self) -> ToolName {
        ToolName::namespaced(self.namespace, self.tool_name)
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec::Namespace(codex_tools::ResponsesApiNamespace {
            name: self.namespace.to_string(),
            description: "Test namespace.".to_string(),
            tools: vec![ResponsesApiNamespaceTool::Function(ResponsesApiTool {
                name: self.tool_name.to_string(),
                description: "Test namespace tool.".to_string(),
                strict: false,
                defer_loading: None,
                parameters: codex_tools::JsonSchema::default(),
                output_schema: None,
            })],
        })
    }

    fn handle(&self, _call: ExtensionToolCall) -> codex_tools::ToolExecutorFuture<'_> {
        Box::pin(async {
            Ok(Box::new(codex_tools::JsonToolOutput::new(json!({}))) as Box<dyn ToolOutput>)
        })
    }
}

struct DeferredExtensionTool;

impl ToolExecutor<ExtensionToolCall> for DeferredExtensionTool {
    fn tool_name(&self) -> ToolName {
        ToolName::plain("extension_echo")
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec::Function(ResponsesApiTool {
            name: "extension_echo".to_string(),
            description: "Echoes arguments through an extension tool.".to_string(),
            strict: true,
            defer_loading: None,
            parameters: codex_tools::JsonSchema::object(
                BTreeMap::from([(
                    "message".to_string(),
                    codex_tools::JsonSchema::string(/*description*/ None),
                )]),
                Some(vec!["message".to_string()]),
                Some(false.into()),
            ),
            output_schema: None,
        })
    }

    fn exposure(&self) -> ToolExposure {
        ToolExposure::Deferred
    }

    fn handle(&self, _call: ExtensionToolCall) -> codex_tools::ToolExecutorFuture<'_> {
        Box::pin(async { panic!("spec planning should not execute extension tools") })
    }
}

struct NonSearchableDeferredExtensionTool;

impl ToolExecutor<ExtensionToolCall> for NonSearchableDeferredExtensionTool {
    fn tool_name(&self) -> ToolName {
        ToolName::plain("extension_echo")
    }

    fn spec(&self) -> ToolSpec {
        DeferredExtensionTool.spec()
    }

    fn exposure(&self) -> ToolExposure {
        ToolExposure::Deferred
    }

    fn search_info(&self) -> Option<ToolSearchInfo> {
        None
    }

    fn handle(&self, _call: ExtensionToolCall) -> codex_tools::ToolExecutorFuture<'_> {
        Box::pin(async { panic!("spec planning should not execute extension tools") })
    }
}

fn duplicate_primary_environment(turn: &mut TurnContext) {
    let mut second_environment = turn.environments.turn_environments[0].clone();
    second_environment.environment_id = "secondary".to_string();
    turn.environments.turn_environments.push(second_environment);
}

fn mcp_tool(server: &str, namespace: &str, name: &str) -> ToolInfo {
    ToolInfo {
        server_name: server.to_string(),
        supports_parallel_tool_calls: false,
        server_origin: None,
        callable_name: name.to_string(),
        callable_namespace: namespace.to_string(),
        namespace_description: Some(format!("Tools from {server}.")),
        tool: rmcp::model::Tool::new(
            name.to_string(),
            format!("{name} test tool"),
            Arc::new(rmcp::model::object(json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false,
            }))),
        ),
        connector_id: None,
        connector_name: None,
        plugin_display_names: Vec::new(),
    }
}

fn invalid_mcp_tool(server: &str, namespace: &str, name: &str) -> ToolInfo {
    let mut tool = mcp_tool(server, namespace, name);
    tool.tool.input_schema = Arc::new(rmcp::model::object(json!({
        "type": "null",
    })));
    tool
}

fn dynamic_tool(namespace: Option<&str>, name: &str, defer_loading: bool) -> DynamicToolSpec {
    dynamic_tool_with_description(
        namespace,
        name,
        defer_loading,
        format!("{name} dynamic tool"),
    )
}

fn dynamic_tool_with_description(
    namespace: Option<&str>,
    name: &str,
    defer_loading: bool,
    description: String,
) -> DynamicToolSpec {
    let function = codex_protocol::dynamic_tools::DynamicToolFunctionSpec {
        name: name.to_string(),
        description,
        input_schema: json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false,
        }),
        defer_loading,
    };
    match namespace {
        Some(namespace) => {
            DynamicToolSpec::Namespace(codex_protocol::dynamic_tools::DynamicToolNamespaceSpec {
                name: namespace.to_string(),
                description: format!("{namespace} dynamic tools"),
                tools: vec![
                    codex_protocol::dynamic_tools::DynamicToolNamespaceTool::Function(function),
                ],
            })
        }
        None => DynamicToolSpec::Function(function),
    }
}

fn sized_dynamic_tool_description_for_namespace_merge_cap() -> String {
    for repeat_count in 1..2_000 {
        let description = "merge cap ".repeat(repeat_count);
        let one_tool_tokens = super::serialized_tool_spec_token_count(&namespace_tool_spec(
            "codex_app",
            "tool_one",
            &description,
            &[],
        ));
        let two_tool_tokens = super::serialized_tool_spec_token_count(&namespace_tool_spec(
            "codex_app",
            "tool_one",
            &description,
            &["tool_two"],
        ));
        if one_tool_tokens <= EXACT_TAIL_PROMOTED_TOOL_SPEC_TOKEN_LIMIT
            && two_tool_tokens > EXACT_TAIL_PROMOTED_TOOL_SPEC_TOKEN_LIMIT
        {
            return description;
        }
    }
    panic!("failed to find deterministic namespace merge cap fixture size");
}

fn sized_dynamic_tool_description_for_aggregate_cap() -> (String, usize) {
    for repeat_count in 1..2_000 {
        let description = "aggregate cap ".repeat(repeat_count);
        let tokens = super::serialized_tool_spec_token_count(&function_tool_spec(
            "aggregate_probe",
            &description,
        ));
        if tokens > EXACT_TAIL_PROMOTED_TOOL_SPEC_TOKEN_LIMIT / 2
            && tokens <= EXACT_TAIL_PROMOTED_TOOL_SPEC_TOKEN_LIMIT
        {
            return (description, tokens);
        }
    }
    panic!("failed to find deterministic aggregate cap fixture size");
}

fn namespace_tool_spec(
    namespace: &str,
    first_tool_name: &str,
    description: &str,
    additional_tool_names: &[&str],
) -> ToolSpec {
    let mut tools = vec![ResponsesApiNamespaceTool::Function(ResponsesApiTool {
        name: first_tool_name.to_string(),
        description: description.to_string(),
        strict: false,
        defer_loading: None,
        parameters: codex_tools::JsonSchema::object(
            BTreeMap::new(),
            None,
            Some(codex_tools::AdditionalProperties::Boolean(false)),
        ),
        output_schema: None,
    })];
    tools.extend(additional_tool_names.iter().map(|tool_name| {
        ResponsesApiNamespaceTool::Function(ResponsesApiTool {
            name: (*tool_name).to_string(),
            description: description.to_string(),
            strict: false,
            defer_loading: None,
            parameters: codex_tools::JsonSchema::object(
                BTreeMap::new(),
                None,
                Some(codex_tools::AdditionalProperties::Boolean(false)),
            ),
            output_schema: None,
        })
    }));
    ToolSpec::Namespace(ResponsesApiNamespace {
        name: namespace.to_string(),
        description: format!("{namespace} dynamic tools"),
        tools,
    })
}

fn function_tool_spec(name: &str, description: &str) -> ToolSpec {
    ToolSpec::Function(ResponsesApiTool {
        name: name.to_string(),
        description: description.to_string(),
        strict: false,
        defer_loading: None,
        parameters: codex_tools::JsonSchema::object(
            BTreeMap::new(),
            None,
            Some(codex_tools::AdditionalProperties::Boolean(false)),
        ),
        output_schema: None,
    })
}

fn provider_with_capabilities(
    info: ModelProviderInfo,
    capabilities: ProviderCapabilities,
) -> SharedModelProvider {
    Arc::new(CapabilityProvider { info, capabilities })
}

fn pending_exact_tail_hint(references: Vec<ToolName>) -> PendingExactTailToolSurfaceHint {
    let namespaces = references
        .iter()
        .filter_map(|reference| reference.namespace.clone())
        .collect::<BTreeSet<_>>();
    pending_exact_tail_hint_from_hint(ExactTailToolSurfaceHint {
        hot_tool_reference_count: references.len(),
        references,
        hot_tool_call_count: 1,
        hot_tool_namespace_count: namespaces.len(),
        hot_tool_reference_overflow_count: 0,
        out_of_scope_dependency_protocol_count: 0,
    })
}

fn pending_exact_tail_hint_from_hint(
    hint: ExactTailToolSurfaceHint,
) -> PendingExactTailToolSurfaceHint {
    PendingExactTailToolSurfaceHint::new("compact-test", "remote_v2", hint)
}

fn notice_text(item: ResponseItem) -> String {
    let ResponseItem::Message { content, .. } = item else {
        panic!("expected exact-tail notice message");
    };
    content
        .into_iter()
        .filter_map(|item| match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => Some(text),
            ContentItem::InputImage { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn tool_reference_label(reference: &ToolName) -> String {
    match &reference.namespace {
        Some(namespace) => format!("{namespace}.{}", reference.name),
        None => reference.name.clone(),
    }
}

fn expected_exact_tail_outcome(
    hot_tool_reference_count: usize,
    hot_tool_namespace_count: usize,
) -> ExactTailToolSurfaceOutcome {
    ExactTailToolSurfaceOutcome {
        compaction_id: "compact-test".to_string(),
        route: "remote_v2".to_string(),
        hot_tool_call_count: 1,
        hot_tool_namespace_count,
        hot_tool_reference_count,
        ..ExactTailToolSurfaceOutcome::default()
    }
}

fn plugin_candidates(presentation: ToolSuggestPresentation) -> ToolSuggestCandidates {
    ToolSuggestCandidates {
        tools: vec![DiscoverableTool::Plugin(Box::new(DiscoverablePluginInfo {
            id: "github@openai-curated-remote".to_string(),
            remote_plugin_id: None,
            name: "GitHub".to_string(),
            description: Some("Work with GitHub repositories".to_string()),
            has_skills: true,
            mcp_server_names: Vec::new(),
            app_connector_ids: Vec::new(),
        }))],
        presentation,
    }
}

fn has_parameter(spec: &ToolSpec, parameter_name: &str) -> bool {
    serde_json::to_value(spec)
        .expect("tool spec should serialize")
        .pointer(&format!("/parameters/properties/{parameter_name}"))
        .is_some()
}

fn apply_patch_accepts_environment_id(spec: &ToolSpec) -> bool {
    match spec {
        ToolSpec::Freeform(tool) if tool.name == "apply_patch" => {
            tool.format.definition.contains("Environment ID")
        }
        _ => false,
    }
}

#[tokio::test]
async fn request_user_input_tool_respects_experimental_config_gate() {
    let enabled = probe(|_| {}).await;
    enabled.assert_visible_contains(&["request_user_input"]);
    enabled.assert_registered_contains(&["request_user_input"]);
    assert_eq!(
        enabled.exposure("request_user_input"),
        ToolExposure::DirectModelOnly
    );

    let disabled = probe(|turn| {
        update_config(turn, |config| {
            config.experimental_request_user_input_enabled = false;
        });
    })
    .await;
    disabled.assert_visible_lacks(&["request_user_input"]);
    disabled.assert_registered_lacks(&["request_user_input"]);
}

#[tokio::test]
async fn request_user_input_stays_direct_in_code_mode_only() {
    let plan = probe(|turn| {
        set_features(turn, &[Feature::CodeMode, Feature::CodeModeOnly]);
    })
    .await;

    plan.assert_visible_contains(&[
        "request_user_input",
        codex_code_mode::PUBLIC_TOOL_NAME,
        codex_code_mode::WAIT_TOOL_NAME,
    ]);
    plan.assert_registered_contains(&["request_user_input"]);
    assert_eq!(
        plan.exposure("request_user_input"),
        ToolExposure::DirectModelOnly
    );

    let ToolSpec::Freeform(exec) = plan.visible_spec(codex_code_mode::PUBLIC_TOOL_NAME) else {
        panic!("expected code mode exec tool");
    };
    assert!(!exec.description.contains("request_user_input"));
}

#[tokio::test]
async fn shell_family_registers_visible_unified_exec_and_hidden_legacy_shell() {
    let plan = probe(|turn| {
        set_features(turn, &[Feature::ShellTool, Feature::UnifiedExec]);
        set_feature(turn, Feature::ShellZshFork, /*enabled*/ false);
        turn.model_info.shell_type = ConfigShellToolType::ShellCommand;
    })
    .await;

    plan.assert_visible_contains(&["exec_command", "write_stdin"]);
    plan.assert_visible_lacks(&["shell_command"]);
    plan.assert_registered_contains(&["exec_command", "write_stdin", "shell_command"]);
    assert_eq!(plan.exposure("shell_command"), ToolExposure::Hidden);
    assert!(has_parameter(plan.visible_spec("exec_command"), "shell"));
}

#[tokio::test]
async fn shell_zsh_fork_stays_standalone_until_unified_exec_composition_is_enabled() {
    let standalone = probe(|turn| {
        set_features(turn, &[Feature::ShellTool, Feature::UnifiedExec]);
        set_feature(turn, Feature::ShellZshFork, /*enabled*/ true);
        set_feature(turn, Feature::UnifiedExecZshFork, /*enabled*/ false);
        turn.model_info.shell_type = ConfigShellToolType::ShellCommand;
    })
    .await;

    standalone.assert_visible_contains(&["shell_command"]);
    standalone.assert_visible_lacks(&["exec_command", "write_stdin"]);
    standalone.assert_registered_contains(&["shell_command"]);
    standalone.assert_registered_lacks(&["exec_command", "write_stdin"]);

    let composed = probe(|turn| {
        set_features(
            turn,
            &[
                Feature::ShellTool,
                Feature::UnifiedExec,
                Feature::ShellZshFork,
                Feature::UnifiedExecZshFork,
            ],
        );
        turn.model_info.shell_type = ConfigShellToolType::ShellCommand;
    })
    .await;

    if codex_utils_pty::conpty_supported() {
        composed.assert_visible_contains(&["exec_command", "write_stdin"]);
        composed.assert_visible_lacks(&["shell_command"]);
        composed.assert_registered_contains(&["exec_command", "write_stdin", "shell_command"]);
        assert_eq!(composed.exposure("shell_command"), ToolExposure::Hidden);
    } else {
        composed.assert_visible_contains(&["shell_command"]);
        composed.assert_visible_lacks(&["exec_command", "write_stdin"]);
    }
}

#[tokio::test]
async fn zsh_fork_unified_exec_hides_shell_parameter() {
    if !codex_utils_pty::conpty_supported() {
        return;
    }

    let plan = probe(|turn| {
        set_features(
            turn,
            &[
                Feature::ShellTool,
                Feature::UnifiedExec,
                Feature::ShellZshFork,
                Feature::UnifiedExecZshFork,
            ],
        );
        turn.unified_exec_shell_mode =
            codex_tools::UnifiedExecShellMode::ZshFork(zsh_fork_config_for_spec_plan_tests());
    })
    .await;

    plan.assert_visible_contains(&["exec_command", "write_stdin"]);
    assert!(!has_parameter(plan.visible_spec("exec_command"), "shell"));
}

#[tokio::test]
async fn zsh_fork_unified_exec_keeps_shell_parameter_when_remote_environment_available() {
    if !codex_utils_pty::conpty_supported() {
        return;
    }

    let plan = probe(|turn| {
        set_features(
            turn,
            &[
                Feature::ShellTool,
                Feature::UnifiedExec,
                Feature::ShellZshFork,
                Feature::UnifiedExecZshFork,
            ],
        );
        turn.unified_exec_shell_mode =
            codex_tools::UnifiedExecShellMode::ZshFork(zsh_fork_config_for_spec_plan_tests());
        let remote_cwd = turn
            .environments
            .primary()
            .expect("primary environment")
            .cwd()
            .clone();
        turn.environments.turn_environments.push(
            crate::session::turn_context::TurnEnvironment::new(
                "remote".to_string(),
                Arc::new(
                    codex_exec_server::Environment::create_for_tests(Some(
                        "ws://127.0.0.1:1/remote-exec-server".to_string(),
                    ))
                    .expect("remote test environment"),
                ),
                remote_cwd,
                /*shell*/ None,
            ),
        );
    })
    .await;

    plan.assert_visible_contains(&["exec_command", "write_stdin"]);
    assert!(has_parameter(plan.visible_spec("exec_command"), "shell"));
    assert!(has_parameter(
        plan.visible_spec("exec_command"),
        "environment_id"
    ));
}

#[tokio::test]
async fn environment_count_controls_environment_backed_tools() {
    let no_environment = probe(|turn| {
        turn.environments.turn_environments.clear();
        set_feature(turn, Feature::ShellTool, /*enabled*/ true);
        set_feature(turn, Feature::RequestPermissionsTool, /*enabled*/ true);
        turn.model_info.apply_patch_tool_type = Some(ApplyPatchToolType::Freeform);
    })
    .await;
    no_environment.assert_visible_lacks(&[
        "shell_command",
        "exec_command",
        "apply_patch",
        "view_image",
        "request_permissions",
    ]);
    no_environment.assert_registered_lacks(&[
        "shell_command",
        "exec_command",
        "apply_patch",
        "view_image",
        "request_permissions",
    ]);

    let multiple_environments = probe(|turn| {
        duplicate_primary_environment(turn);
        set_feature(turn, Feature::ShellTool, /*enabled*/ true);
        set_feature(turn, Feature::UnifiedExec, /*enabled*/ true);
        set_feature(turn, Feature::RequestPermissionsTool, /*enabled*/ true);
        turn.model_info.apply_patch_tool_type = Some(ApplyPatchToolType::Freeform);
    })
    .await;
    multiple_environments.assert_visible_contains(&[
        "exec_command",
        "apply_patch",
        "view_image",
        "request_permissions",
    ]);
    assert!(has_parameter(
        multiple_environments.visible_spec("exec_command"),
        "environment_id"
    ));
    assert!(apply_patch_accepts_environment_id(
        multiple_environments.visible_spec("apply_patch")
    ));
    assert!(has_parameter(
        multiple_environments.visible_spec("view_image"),
        "environment_id"
    ));
}

#[tokio::test]
async fn environment_tools_follow_the_step_context() {
    let (_session, mut turn) = make_session_and_context().await;
    set_feature(&mut turn, Feature::UnifiedExec, /*enabled*/ true);
    turn.model_info.apply_patch_tool_type = Some(ApplyPatchToolType::Freeform);

    let environments = turn.environments.clone();
    turn.environments.turn_environments.clear();
    let turn = Arc::new(turn);
    let step_context = Arc::new(StepContext::new(
        Arc::clone(&turn),
        environments,
        Vec::new(),
        crate::session::McpRuntimeSnapshot::new_uninitialized_for_test(&turn.config),
        /*loaded_agents_md*/ None,
    ));

    let plan = ToolPlanProbe::from_router(ToolRouter::from_context(
        step_context.as_ref(),
        ToolRouterParams {
            mcp_tools: None,
            deferred_mcp_tools: None,
            tool_suggest_candidates: None,
            extension_tool_executors: Vec::new(),
            dynamic_tools: &[],
            exact_tail_tool_surface_hint: None,
        },
        &Default::default(),
    ));

    plan.assert_visible_contains(&["exec_command", "apply_patch", "view_image"]);
}

#[tokio::test]
async fn host_context_gates_agent_job_tools() {
    let normal_agent_job = probe(|turn| {
        set_feature(turn, Feature::SpawnCsv, /*enabled*/ true);
    })
    .await;
    normal_agent_job.assert_visible_contains(&["spawn_agents_on_csv"]);
    normal_agent_job.assert_visible_lacks(&["report_agent_job_result"]);

    let worker_agent_job = probe(|turn| {
        set_feature(turn, Feature::SpawnCsv, /*enabled*/ true);
        turn.session_source =
            SessionSource::SubAgent(SubAgentSource::Other("agent_job:42".to_string()));
    })
    .await;
    worker_agent_job.assert_visible_contains(&["spawn_agents_on_csv", "report_agent_job_result"]);
}

#[tokio::test]
async fn sleep_tool_follows_current_time_config() {
    let disabled = probe(|turn| {
        set_feature(turn, Feature::CurrentTimeReminder, /*enabled*/ true);
    })
    .await;
    assert_eq!(disabled.namespace_function_names("clock"), ["curr_time"]);

    let enabled = probe(|turn| {
        set_feature(turn, Feature::CurrentTimeReminder, /*enabled*/ true);
        let mut config = (*turn.config).clone();
        config.current_time_reminder = Some(CurrentTimeReminderConfig {
            sleep_tool: true,
            ..CurrentTimeReminderConfig::default()
        });
        turn.config = Arc::new(config);
    })
    .await;
    assert_eq!(
        enabled.namespace_function_names("clock"),
        ["curr_time", "sleep"]
    );
}

#[tokio::test]
async fn mcp_and_tool_search_follow_direct_and_deferred_tool_exposure() {
    let direct_mcp = probe_with(
        |_| {},
        ToolPlanInputs {
            mcp_tools: Some(vec![mcp_tool("direct", "mcp__direct", "lookup")]),
            ..ToolPlanInputs::default()
        },
    )
    .await;
    direct_mcp.assert_visible_contains(&[
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
    ]);
    assert_eq!(
        direct_mcp.namespace_function_names("mcp__direct"),
        &["lookup".to_string()]
    );

    let searchable_mcp = ToolPlanInputs {
        deferred_mcp_tools: Some(vec![mcp_tool("searchable", "mcp__searchable", "lookup")]),
        ..ToolPlanInputs::default()
    };

    let missing_model_capability = probe_with(
        |turn| {
            turn.model_info.supports_search_tool = false;
        },
        ToolPlanInputs {
            deferred_mcp_tools: searchable_mcp.deferred_mcp_tools.clone(),
            ..ToolPlanInputs::default()
        },
    )
    .await;
    missing_model_capability.assert_visible_lacks(&["tool_search"]);

    let missing_deferred_tools = probe(|turn| {
        set_feature(turn, Feature::Collab, /*enabled*/ false);
        turn.model_info.supports_search_tool = true;
    })
    .await;
    missing_deferred_tools.assert_visible_lacks(&["tool_search"]);
    missing_deferred_tools.assert_visible_lacks(&[
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
    ]);

    let bedrock_namespace_capability = probe_with(
        |turn| {
            turn.model_info.supports_search_tool = true;
            use_bedrock_provider(turn);
        },
        ToolPlanInputs {
            deferred_mcp_tools: searchable_mcp.deferred_mcp_tools.clone(),
            ..ToolPlanInputs::default()
        },
    )
    .await;
    bedrock_namespace_capability.assert_visible_contains(&["tool_search"]);

    let enabled = probe_with(
        |turn| {
            turn.model_info.supports_search_tool = true;
        },
        searchable_mcp,
    )
    .await;
    enabled.assert_visible_contains(&["tool_search"]);
    enabled.assert_registered_contains(&[
        "tool_search",
        &ToolName::namespaced("mcp__searchable", "lookup").to_string(),
    ]);
}

#[tokio::test]
async fn deferred_extension_tools_are_discoverable_with_tool_search() {
    let plan = probe_with(
        |turn| {
            turn.model_info.supports_search_tool = true;
        },
        ToolPlanInputs {
            extension_tool_executors: vec![Arc::new(DeferredExtensionTool)],
            ..ToolPlanInputs::default()
        },
    )
    .await;

    plan.assert_visible_contains(&["tool_search"]);
    plan.assert_visible_lacks(&["extension_echo"]);
    plan.assert_registered_contains(&["extension_echo"]);
    assert_eq!(plan.exposure("extension_echo"), ToolExposure::Deferred);
}

#[tokio::test]
async fn exact_tail_hint_rehydrates_current_deferred_dynamic_tool() {
    let tool_name = ToolName::namespaced("codex_app", "restored_tool");
    let mut expected = expected_exact_tail_outcome(
        /*hot_tool_reference_count*/ 1, /*hot_tool_namespace_count*/ 1,
    );
    expected.rehydrated_tool_count = 1;
    expected.discoverable_hot_tool_count = 1;
    expected.rehydrated_tool_references = vec![tool_reference_label(&tool_name)];
    expected.rehydrated_tool_namespaces = vec!["codex_app".to_string()];
    expected.tool_surface_changed_after_compaction = true;
    let plan = probe_with(
        |turn| {
            turn.model_info.supports_search_tool = true;
        },
        ToolPlanInputs {
            dynamic_tools: vec![dynamic_tool(
                Some("codex_app"),
                "restored_tool",
                /*defer_loading*/ true,
            )],
            exact_tail_tool_surface_hint: Some(pending_exact_tail_hint(vec![tool_name.clone()])),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    assert_eq!(
        plan.namespace_function_names("codex_app"),
        &["restored_tool".to_string()]
    );
    assert_eq!(plan.exposure(&tool_name.to_string()), ToolExposure::Direct);
    assert_eq!(plan.exact_tail_tool_surface_outcome, Some(expected));
}

#[tokio::test]
async fn exact_tail_hint_rehydrates_multi_agent_v1_namespace_atomically() {
    let references = vec![
        ToolName::namespaced(MULTI_AGENT_V1_NAMESPACE, "close_agent"),
        ToolName::namespaced(MULTI_AGENT_V1_NAMESPACE, "wait_agent"),
        ToolName::namespaced(MULTI_AGENT_V1_NAMESPACE, "spawn_agent"),
    ];
    let mut expected = expected_exact_tail_outcome(
        /*hot_tool_reference_count*/ references.len(),
        /*hot_tool_namespace_count*/ 1,
    );
    expected.rehydrated_tool_count = references.len();
    expected.discoverable_hot_tool_count = references.len();
    expected.rehydrated_tool_references = references.iter().map(tool_reference_label).collect();
    expected.rehydrated_tool_namespaces = vec![MULTI_AGENT_V1_NAMESPACE.to_string()];
    expected.tool_surface_changed_after_compaction = true;
    let plan = probe_with(
        |turn| {
            turn.model_info.supports_search_tool = true;
            set_feature(turn, Feature::Collab, /*enabled*/ true);
            set_feature(turn, Feature::MultiAgentV2, /*enabled*/ false);
        },
        ToolPlanInputs {
            exact_tail_tool_surface_hint: Some(pending_exact_tail_hint(references.clone())),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    plan.assert_visible_contains(&[MULTI_AGENT_V1_NAMESPACE]);
    assert_eq!(
        plan.namespace_function_names(MULTI_AGENT_V1_NAMESPACE),
        &[
            "close_agent".to_string(),
            "resume_agent".to_string(),
            "send_input".to_string(),
            "spawn_agent".to_string(),
            "wait_agent".to_string(),
        ]
    );
    for tool_name in [
        "close_agent",
        "resume_agent",
        "send_input",
        "spawn_agent",
        "wait_agent",
    ] {
        let namespaced_tool_name = ToolName::namespaced(MULTI_AGENT_V1_NAMESPACE, tool_name);
        assert_eq!(
            plan.exposure(&namespaced_tool_name.to_string()),
            ToolExposure::Direct,
            "expected {tool_name} to be promoted with its namespace family"
        );
    }
    assert_eq!(plan.exact_tail_tool_surface_outcome, Some(expected));
}

#[tokio::test]
async fn exact_tail_hint_preserves_current_schema_authority_over_historical_reference() {
    let tool_name = ToolName::namespaced("codex_app", "current_tool");
    let historical_hint = derive_exact_tail_tool_surface_hint(&[ResponseItem::ToolSearchOutput {
        id: None,
        call_id: Some("historical-search".to_string()),
        status: "completed".to_string(),
        execution: "search".to_string(),
        tools: vec![json!({
            "type": "namespace",
            "name": "codex_app",
            "description": "stale historical namespace description",
            "tools": [{
                "type": "function",
                "name": "current_tool",
                "description": "stale historical tool description",
                "parameters": {
                    "type": "object",
                    "properties": {"stale": {"type": "string"}},
                    "required": ["stale"],
                    "additionalProperties": true
                }
            }]
        })],
        internal_chat_message_metadata_passthrough: None,
    }]);
    assert_eq!(historical_hint.references, vec![tool_name.clone()]);
    let plan = probe_with(
        |turn| {
            turn.model_info.supports_search_tool = true;
        },
        ToolPlanInputs {
            dynamic_tools: vec![dynamic_tool(
                Some("codex_app"),
                "current_tool",
                /*defer_loading*/ true,
            )],
            exact_tail_tool_surface_hint: Some(pending_exact_tail_hint_from_hint(historical_hint)),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    let ToolSpec::Namespace(namespace) = plan.visible_spec("codex_app") else {
        panic!("expected current dynamic namespace");
    };
    assert_eq!(namespace.description, "codex_app dynamic tools");
    assert_eq!(
        namespace.tools,
        vec![ResponsesApiNamespaceTool::Function(ResponsesApiTool {
            name: "current_tool".to_string(),
            description: "current_tool dynamic tool".to_string(),
            strict: false,
            defer_loading: None,
            parameters: codex_tools::JsonSchema::object(
                BTreeMap::new(),
                None,
                Some(codex_tools::AdditionalProperties::Boolean(false)),
            ),
            output_schema: None,
        })]
    );
}

#[tokio::test]
async fn exact_tail_hint_notice_avoids_tool_search_instruction() {
    let restored_tool_name = ToolName::namespaced("codex_app", "restored_tool");
    let stale_tool_name = ToolName::namespaced("codex_app", "stale_tool");
    let mut expected = expected_exact_tail_outcome(
        /*hot_tool_reference_count*/ 2, /*hot_tool_namespace_count*/ 1,
    );
    expected.rehydrated_tool_count = 1;
    expected.missing_hot_tool_count = 1;
    expected.discoverable_hot_tool_count = 1;
    expected.missing_notice_emitted_count = 1;
    expected.missing_no_path_count = 1;
    expected.rehydrated_tool_references = vec![tool_reference_label(&restored_tool_name)];
    expected.missing_tool_references = vec![tool_reference_label(&stale_tool_name)];
    expected.rehydrated_tool_namespaces = vec!["codex_app".to_string()];
    expected.missing_tool_rejection_reasons = vec!["no_runtime".to_string()];
    expected.tool_surface_changed_after_compaction = true;
    expected.tool_surface_rehydration_failure_reason =
        Some("hot_tool_references_not_model_visible".to_string());
    let plan = probe_with(
        |turn| {
            turn.model_info.supports_search_tool = true;
        },
        ToolPlanInputs {
            dynamic_tools: vec![dynamic_tool(
                Some("codex_app"),
                "restored_tool",
                /*defer_loading*/ true,
            )],
            exact_tail_tool_surface_hint: Some(pending_exact_tail_hint(vec![
                restored_tool_name.clone(),
                stale_tool_name.clone(),
            ])),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    plan.assert_visible_contains(&["codex_app"]);
    assert_eq!(
        plan.exposure(&restored_tool_name.to_string()),
        ToolExposure::Direct
    );
    assert!(!plan.registered_names.contains(&stale_tool_name.to_string()));
    let notice = notice_text(
        exact_tail_tool_surface_notice_item(
            plan.exact_tail_tool_surface_outcome
                .as_ref()
                .expect("exact-tail outcome"),
        )
        .expect("missing reference should emit notice"),
    );
    assert!(notice.contains("Do not assume missing historical tools remain callable."));
    assert!(notice.contains("Missing direct tool reference(s): codex_app.stale_tool."));
    assert!(!notice.contains("tool_search"));
    assert_eq!(plan.exact_tail_tool_surface_outcome, Some(expected));
}

#[tokio::test]
async fn exact_tail_hint_reports_stale_reference_without_resurrecting_it() {
    let stale_tool_name = ToolName::namespaced("codex_app", "stale_tool");
    let mut expected = expected_exact_tail_outcome(
        /*hot_tool_reference_count*/ 1, /*hot_tool_namespace_count*/ 1,
    );
    expected.missing_hot_tool_count = 1;
    expected.missing_notice_emitted_count = 1;
    expected.missing_no_path_count = 1;
    expected.missing_tool_references = vec![tool_reference_label(&stale_tool_name)];
    expected.missing_tool_rejection_reasons = vec!["no_runtime".to_string()];
    expected.tool_surface_changed_after_compaction = true;
    expected.tool_surface_rehydration_failure_reason =
        Some("hot_tool_references_not_model_visible".to_string());
    let plan = probe_with(
        |turn| {
            turn.model_info.supports_search_tool = true;
        },
        ToolPlanInputs {
            dynamic_tools: vec![dynamic_tool(
                Some("codex_app"),
                "current_tool",
                /*defer_loading*/ true,
            )],
            exact_tail_tool_surface_hint: Some(pending_exact_tail_hint(vec![
                stale_tool_name.clone(),
            ])),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    assert_eq!(plan.namespace_function_names("codex_app"), &[] as &[String]);
    assert!(
        exact_tail_tool_surface_notice_item(
            plan.exact_tail_tool_surface_outcome
                .as_ref()
                .expect("exact-tail outcome")
        )
        .is_some()
    );
    assert_eq!(plan.exact_tail_tool_surface_outcome, Some(expected));
    assert!(!plan.registered_names.contains(&stale_tool_name.to_string()));
}

#[tokio::test]
async fn exact_tail_hint_overflow_emits_bounded_notice() {
    let mut expected = expected_exact_tail_outcome(
        /*hot_tool_reference_count*/ 2, /*hot_tool_namespace_count*/ 0,
    );
    expected.missing_notice_emitted_count = 1;
    expected.hot_tool_reference_overflow_count = 2;
    expected.hot_tool_reference_overflow_notice_emitted_count = 1;
    expected.tool_surface_changed_after_compaction = true;
    let plan = probe_with(
        |turn| {
            turn.model_info.supports_search_tool = true;
        },
        ToolPlanInputs {
            exact_tail_tool_surface_hint: Some(pending_exact_tail_hint_from_hint(
                ExactTailToolSurfaceHint {
                    hot_tool_reference_count: 2,
                    references: Vec::new(),
                    hot_tool_call_count: 1,
                    hot_tool_namespace_count: 0,
                    hot_tool_reference_overflow_count: 2,
                    out_of_scope_dependency_protocol_count: 0,
                },
            )),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    let notice = notice_text(
        exact_tail_tool_surface_notice_item(
            plan.exact_tail_tool_surface_outcome
                .as_ref()
                .expect("exact-tail outcome"),
        )
        .expect("overflow should emit notice"),
    );
    assert!(notice.contains("2 older exact-tail tool reference(s)"));
    assert!(!notice.contains("tool_search"));
    assert_eq!(plan.exact_tail_tool_surface_outcome, Some(expected));
}

#[tokio::test]
async fn exact_tail_hint_does_not_rehydrate_non_searchable_deferred_tool() {
    let tool_name = ToolName::plain("extension_echo");
    let mut expected = expected_exact_tail_outcome(
        /*hot_tool_reference_count*/ 1, /*hot_tool_namespace_count*/ 0,
    );
    expected.missing_hot_tool_count = 1;
    expected.missing_notice_emitted_count = 1;
    expected.missing_no_path_count = 1;
    expected.missing_tool_references = vec![tool_reference_label(&tool_name)];
    expected.missing_tool_rejection_reasons = vec!["not_model_visible".to_string()];
    expected.tool_surface_changed_after_compaction = true;
    expected.tool_surface_rehydration_failure_reason =
        Some("hot_tool_references_not_model_visible".to_string());
    let plan = probe_with(
        |turn| {
            turn.model_info.supports_search_tool = true;
        },
        ToolPlanInputs {
            extension_tool_executors: vec![Arc::new(NonSearchableDeferredExtensionTool)],
            exact_tail_tool_surface_hint: Some(pending_exact_tail_hint(vec![tool_name.clone()])),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    plan.assert_visible_lacks(&["extension_echo"]);
    assert_eq!(plan.exposure("extension_echo"), ToolExposure::Deferred);
    assert_eq!(plan.exact_tail_tool_surface_outcome, Some(expected));
}

#[tokio::test]
async fn exact_tail_hint_already_direct_tool_does_not_change_surface() {
    let tool_name = ToolName::namespaced("codex_app", "visible_tool");
    let mut expected = expected_exact_tail_outcome(
        /*hot_tool_reference_count*/ 1, /*hot_tool_namespace_count*/ 1,
    );
    expected.already_direct_tool_count = 1;
    let plan = probe_with(
        |_| {},
        ToolPlanInputs {
            dynamic_tools: vec![dynamic_tool(
                Some("codex_app"),
                "visible_tool",
                /*defer_loading*/ false,
            )],
            exact_tail_tool_surface_hint: Some(pending_exact_tail_hint(vec![tool_name.clone()])),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    assert_eq!(plan.exact_tail_tool_surface_outcome, Some(expected));
}

#[tokio::test]
async fn exact_tail_hint_already_direct_tool_respects_final_namespace_filter() {
    let tool_name = ToolName::namespaced("codex_app", "filtered_direct_tool");
    let mut expected = expected_exact_tail_outcome(
        /*hot_tool_reference_count*/ 1, /*hot_tool_namespace_count*/ 1,
    );
    expected.missing_hot_tool_count = 1;
    expected.missing_notice_emitted_count = 1;
    expected.missing_no_path_count = 1;
    expected.missing_tool_references = vec![tool_reference_label(&tool_name)];
    expected.missing_tool_rejection_reasons = vec!["not_model_visible".to_string()];
    expected.tool_surface_changed_after_compaction = true;
    expected.tool_surface_rehydration_failure_reason =
        Some("hot_tool_references_not_model_visible".to_string());
    let plan = probe_with(
        |turn| {
            turn.provider = provider_with_capabilities(
                turn.provider.info().clone(),
                ProviderCapabilities {
                    namespace_tools: false,
                    ..ProviderCapabilities::default()
                },
            );
        },
        ToolPlanInputs {
            dynamic_tools: vec![dynamic_tool(
                Some("codex_app"),
                "filtered_direct_tool",
                /*defer_loading*/ false,
            )],
            exact_tail_tool_surface_hint: Some(pending_exact_tail_hint(vec![tool_name.clone()])),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    plan.assert_visible_lacks(&["codex_app"]);
    assert_eq!(plan.exposure(&tool_name.to_string()), ToolExposure::Direct);
    assert_eq!(plan.exact_tail_tool_surface_outcome, Some(expected));
}

#[tokio::test]
async fn exact_tail_hint_does_not_rehydrate_namespace_filtered_tool() {
    let tool_name = ToolName::namespaced("codex_app", "filtered_tool");
    let mut expected = expected_exact_tail_outcome(
        /*hot_tool_reference_count*/ 1, /*hot_tool_namespace_count*/ 1,
    );
    expected.missing_hot_tool_count = 1;
    expected.missing_notice_emitted_count = 1;
    expected.missing_no_path_count = 1;
    expected.missing_tool_references = vec![tool_reference_label(&tool_name)];
    expected.missing_tool_rejection_reasons = vec!["not_model_visible".to_string()];
    expected.tool_surface_changed_after_compaction = true;
    expected.tool_surface_rehydration_failure_reason =
        Some("hot_tool_references_not_model_visible".to_string());
    let plan = probe_with(
        |turn| {
            turn.provider = provider_with_capabilities(
                turn.provider.info().clone(),
                ProviderCapabilities {
                    namespace_tools: false,
                    ..ProviderCapabilities::default()
                },
            );
            turn.model_info.supports_search_tool = true;
        },
        ToolPlanInputs {
            dynamic_tools: vec![dynamic_tool(
                Some("codex_app"),
                "filtered_tool",
                /*defer_loading*/ true,
            )],
            exact_tail_tool_surface_hint: Some(pending_exact_tail_hint(vec![tool_name.clone()])),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    plan.assert_visible_lacks(&["codex_app", "tool_search"]);
    assert_eq!(
        plan.exposure(&tool_name.to_string()),
        ToolExposure::Deferred
    );
    assert_eq!(plan.exact_tail_tool_surface_outcome, Some(expected));
}

#[tokio::test]
async fn exact_tail_hint_does_not_promote_when_final_namespace_merge_exceeds_cap() {
    let description = sized_dynamic_tool_description_for_namespace_merge_cap();
    let direct_tool_name = ToolName::namespaced("codex_app", "direct_merge_probe");
    let deferred_tool_name = ToolName::namespaced("codex_app", "deferred_merge_probe");
    let mut expected = expected_exact_tail_outcome(
        /*hot_tool_reference_count*/ 1, /*hot_tool_namespace_count*/ 1,
    );
    expected.missing_hot_tool_count = 1;
    expected.discoverable_hot_tool_count = 1;
    expected.missing_notice_emitted_count = 1;
    expected.missing_tool_references = vec![tool_reference_label(&deferred_tool_name)];
    expected.missing_tool_rejection_reasons = vec!["spec_budget".to_string()];
    expected.tool_surface_changed_after_compaction = true;
    expected.tool_surface_rehydration_failure_reason =
        Some("hot_tool_references_not_model_visible".to_string());
    let plan = probe_with(
        |turn| {
            turn.model_info.supports_search_tool = true;
        },
        ToolPlanInputs {
            dynamic_tools: vec![
                dynamic_tool_with_description(
                    Some("codex_app"),
                    "direct_merge_probe",
                    /*defer_loading*/ false,
                    description.clone(),
                ),
                dynamic_tool_with_description(
                    Some("codex_app"),
                    "deferred_merge_probe",
                    /*defer_loading*/ true,
                    description,
                ),
            ],
            exact_tail_tool_surface_hint: Some(pending_exact_tail_hint(vec![
                deferred_tool_name.clone(),
            ])),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    assert_eq!(
        plan.namespace_function_names("codex_app"),
        &[direct_tool_name.name]
    );
    assert_eq!(
        plan.exposure(&deferred_tool_name.to_string()),
        ToolExposure::Deferred
    );
    assert_eq!(plan.exact_tail_tool_surface_outcome, Some(expected));
}

#[tokio::test]
async fn exact_tail_hint_does_not_rehydrate_oversized_tool_spec() {
    let tool_name = ToolName::namespaced("codex_app", "large_tool");
    let mut expected = expected_exact_tail_outcome(
        /*hot_tool_reference_count*/ 1, /*hot_tool_namespace_count*/ 1,
    );
    expected.missing_hot_tool_count = 1;
    expected.discoverable_hot_tool_count = 1;
    expected.missing_notice_emitted_count = 1;
    expected.missing_tool_references = vec![tool_reference_label(&tool_name)];
    expected.missing_tool_rejection_reasons = vec!["spec_budget".to_string()];
    expected.tool_surface_changed_after_compaction = true;
    expected.tool_surface_rehydration_failure_reason =
        Some("hot_tool_references_not_model_visible".to_string());
    let plan = probe_with(
        |turn| {
            turn.model_info.supports_search_tool = true;
        },
        ToolPlanInputs {
            dynamic_tools: vec![dynamic_tool_with_description(
                Some("codex_app"),
                "large_tool",
                /*defer_loading*/ true,
                "large spec ".repeat(5_000),
            )],
            exact_tail_tool_surface_hint: Some(pending_exact_tail_hint(vec![tool_name.clone()])),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    plan.assert_visible_contains(&["tool_search"]);
    plan.assert_visible_lacks(&["codex_app"]);
    assert_eq!(
        plan.exposure(&tool_name.to_string()),
        ToolExposure::Deferred
    );
    assert_eq!(plan.exact_tail_tool_surface_outcome, Some(expected));
}

#[tokio::test]
async fn exact_tail_hint_oversized_tool_has_no_discovery_path_without_tool_search() {
    let tool_name = ToolName::namespaced("codex_app", "large_tool_without_search");
    let mut expected = expected_exact_tail_outcome(
        /*hot_tool_reference_count*/ 1, /*hot_tool_namespace_count*/ 1,
    );
    expected.missing_hot_tool_count = 1;
    expected.missing_notice_emitted_count = 1;
    expected.missing_no_path_count = 1;
    expected.missing_tool_references = vec![tool_reference_label(&tool_name)];
    expected.missing_tool_rejection_reasons = vec!["not_model_visible".to_string()];
    expected.tool_surface_changed_after_compaction = true;
    expected.tool_surface_rehydration_failure_reason =
        Some("hot_tool_references_not_model_visible".to_string());
    let plan = probe_with(
        |turn| {
            turn.model_info.supports_search_tool = false;
        },
        ToolPlanInputs {
            dynamic_tools: vec![dynamic_tool_with_description(
                Some("codex_app"),
                "large_tool_without_search",
                /*defer_loading*/ true,
                "large spec ".repeat(5_000),
            )],
            exact_tail_tool_surface_hint: Some(pending_exact_tail_hint(vec![tool_name.clone()])),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    let notice = notice_text(
        exact_tail_tool_surface_notice_item(
            plan.exact_tail_tool_surface_outcome
                .as_ref()
                .expect("exact-tail outcome"),
        )
        .expect("missing reference should emit notice"),
    );
    assert!(notice.contains("Do not assume missing historical tools remain callable."));
    assert!(!notice.contains("Rediscover"));
    plan.assert_visible_lacks(&["codex_app", "tool_search"]);
    assert_eq!(
        plan.exposure(&tool_name.to_string()),
        ToolExposure::Deferred
    );
    assert_eq!(plan.exact_tail_tool_surface_outcome, Some(expected));
}

#[tokio::test]
async fn exact_tail_hint_caps_aggregate_promoted_tool_specs() {
    let (description, per_tool_tokens) = sized_dynamic_tool_description_for_aggregate_cap();
    let tool_count = (EXACT_TAIL_PROMOTED_TOOL_SPECS_TOTAL_TOKEN_LIMIT / per_tool_tokens) + 2;
    let references = (0..tool_count)
        .map(|index| ToolName::plain(format!("aggregate_probe_{index}")))
        .collect::<Vec<_>>();
    let expected_rehydrated = EXACT_TAIL_PROMOTED_TOOL_SPECS_TOTAL_TOKEN_LIMIT / per_tool_tokens;
    assert!(expected_rehydrated < tool_count);

    let mut expected = expected_exact_tail_outcome(
        /*hot_tool_reference_count*/ tool_count, /*hot_tool_namespace_count*/ 0,
    );
    expected.rehydrated_tool_count = expected_rehydrated;
    expected.missing_hot_tool_count = tool_count - expected_rehydrated;
    expected.discoverable_hot_tool_count = tool_count;
    expected.missing_notice_emitted_count = 1;
    expected.rehydrated_tool_references = references
        .iter()
        .take(expected_rehydrated)
        .map(tool_reference_label)
        .collect();
    expected.missing_tool_references = references
        .iter()
        .skip(expected_rehydrated)
        .map(tool_reference_label)
        .collect();
    expected.missing_tool_rejection_reasons = vec!["spec_budget".to_string()];
    expected.tool_surface_changed_after_compaction = true;
    expected.tool_surface_rehydration_failure_reason =
        Some("hot_tool_references_not_model_visible".to_string());

    let dynamic_tools = (0..tool_count)
        .map(|index| {
            dynamic_tool_with_description(
                None,
                &format!("aggregate_probe_{index}"),
                /*defer_loading*/ true,
                description.clone(),
            )
        })
        .collect::<Vec<_>>();
    let plan = probe_with(
        |turn| {
            turn.model_info.supports_search_tool = true;
        },
        ToolPlanInputs {
            dynamic_tools,
            exact_tail_tool_surface_hint: Some(pending_exact_tail_hint(references.clone())),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    for tool_name in references.iter().take(expected_rehydrated) {
        plan.assert_visible_contains(&[&tool_name.name]);
        assert_eq!(plan.exposure(&tool_name.to_string()), ToolExposure::Direct);
    }
    for tool_name in references.iter().skip(expected_rehydrated) {
        plan.assert_visible_lacks(&[&tool_name.name]);
        assert_eq!(
            plan.exposure(&tool_name.to_string()),
            ToolExposure::Deferred
        );
    }
    assert_eq!(plan.exact_tail_tool_surface_outcome, Some(expected));
}

#[tokio::test]
async fn tool_search_cache_rebuilds_when_deferred_sources_change() {
    let cache = ToolSearchHandlerCache::default();

    let (_session, mut first_turn) = make_session_and_context().await;
    first_turn.model_info.supports_search_tool = true;
    let first_turn = Arc::new(first_turn);
    let first_step_context = StepContext::for_test(Arc::clone(&first_turn));
    let first_router = ToolRouter::from_context(
        first_step_context.as_ref(),
        ToolRouterParams {
            mcp_tools: None,
            deferred_mcp_tools: Some(vec![mcp_tool("first", "mcp__first", "lookup")]),
            tool_suggest_candidates: None,
            extension_tool_executors: Vec::new(),
            dynamic_tools: &[],
            exact_tail_tool_surface_hint: None,
        },
        &cache,
    );
    let first_plan = ToolPlanProbe::from_router(first_router);

    let (_session, mut second_turn) = make_session_and_context().await;
    second_turn.model_info.supports_search_tool = true;
    let second_turn = Arc::new(second_turn);
    let second_step_context = StepContext::for_test(Arc::clone(&second_turn));
    let second_router = ToolRouter::from_context(
        second_step_context.as_ref(),
        ToolRouterParams {
            mcp_tools: None,
            deferred_mcp_tools: Some(vec![mcp_tool("second", "mcp__second", "lookup")]),
            tool_suggest_candidates: None,
            extension_tool_executors: Vec::new(),
            dynamic_tools: &[],
            exact_tail_tool_surface_hint: None,
        },
        &cache,
    );
    let second_plan = ToolPlanProbe::from_router(second_router);

    let ToolSpec::ToolSearch {
        description: first_description,
        ..
    } = first_plan.visible_spec("tool_search")
    else {
        panic!("expected first tool_search spec");
    };
    assert!(first_description.contains("- first: Tools from first."));
    assert!(!first_description.contains("- second: Tools from second."));

    let ToolSpec::ToolSearch {
        description: second_description,
        ..
    } = second_plan.visible_spec("tool_search")
    else {
        panic!("expected second tool_search spec");
    };
    assert!(second_description.contains("- second: Tools from second."));
    assert!(!second_description.contains("- first: Tools from first."));
}

#[tokio::test]
async fn invalid_mcp_tools_are_not_registered() {
    let plan = probe_with(
        |_| {},
        ToolPlanInputs {
            mcp_tools: Some(vec![invalid_mcp_tool("invalid", "mcp__invalid", "lookup")]),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    plan.assert_visible_lacks(&["mcp__invalid"]);
    plan.assert_registered_lacks(&[&ToolName::namespaced("mcp__invalid", "lookup").to_string()]);
}

#[tokio::test]
async fn request_plugin_install_requires_all_discovery_features() {
    for disabled_feature in [Feature::ToolSuggest, Feature::Apps, Feature::Plugins] {
        let plan = probe_with(
            |turn| {
                set_features(
                    turn,
                    &[Feature::ToolSuggest, Feature::Apps, Feature::Plugins],
                );
                set_feature(turn, disabled_feature, /*enabled*/ false);
            },
            ToolPlanInputs {
                tool_suggest_candidates: Some(plugin_candidates(ToolSuggestPresentation::ListTool)),
                ..ToolPlanInputs::default()
            },
        )
        .await;
        plan.assert_visible_lacks(&[
            "list_available_plugins_to_install",
            "request_plugin_install",
        ]);
    }

    for tool_suggest_candidates in [
        None,
        Some(ToolSuggestCandidates {
            tools: Vec::new(),
            presentation: ToolSuggestPresentation::RecommendationContext,
        }),
    ] {
        let plan = probe_with(
            |turn| {
                set_features(
                    turn,
                    &[Feature::ToolSuggest, Feature::Apps, Feature::Plugins],
                );
            },
            ToolPlanInputs {
                tool_suggest_candidates,
                ..ToolPlanInputs::default()
            },
        )
        .await;
        plan.assert_visible_lacks(&[
            "list_available_plugins_to_install",
            "request_plugin_install",
        ]);
    }

    let enabled = probe_with(
        |turn| {
            set_features(
                turn,
                &[Feature::ToolSuggest, Feature::Apps, Feature::Plugins],
            );
        },
        ToolPlanInputs {
            tool_suggest_candidates: Some(plugin_candidates(ToolSuggestPresentation::ListTool)),
            ..ToolPlanInputs::default()
        },
    )
    .await;
    enabled.assert_visible_contains(&[
        "list_available_plugins_to_install",
        "request_plugin_install",
    ]);
}

#[tokio::test]
async fn request_plugin_install_stays_visible_without_tool_search() {
    let plan = probe_with(
        |turn| {
            turn.model_info.supports_search_tool = false;
            set_features(
                turn,
                &[Feature::ToolSuggest, Feature::Apps, Feature::Plugins],
            );
        },
        ToolPlanInputs {
            tool_suggest_candidates: Some(plugin_candidates(ToolSuggestPresentation::ListTool)),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    plan.assert_visible_contains(&[
        "list_available_plugins_to_install",
        "request_plugin_install",
    ]);
    plan.assert_visible_lacks(&["tool_search"]);
}

#[tokio::test]
async fn request_plugin_install_description_refers_to_recommended_plugins_hint() {
    let plan = probe_with(
        |turn| {
            set_features(
                turn,
                &[Feature::ToolSuggest, Feature::Apps, Feature::Plugins],
            );
        },
        ToolPlanInputs {
            tool_suggest_candidates: Some(plugin_candidates(
                ToolSuggestPresentation::RecommendationContext,
            )),
            ..ToolPlanInputs::default()
        },
    )
    .await;

    let request_spec = plan.visible_spec("request_plugin_install");
    let ToolSpec::Function(ResponsesApiTool {
        description: request_description,
        ..
    }) = request_spec
    else {
        panic!("expected request_plugin_install function spec");
    };
    assert!(request_description.contains("the `<recommended_plugins>` list"));
    assert!(!request_description.contains("list_available_plugins_to_install"));
    assert!(!request_description.contains("github"));
    assert!(has_parameter(request_spec, "plugin_id"));
    assert!(has_parameter(request_spec, "suggest_reason"));
    assert!(!has_parameter(request_spec, "tool_id"));
    assert!(!has_parameter(request_spec, "tool_type"));
    assert!(!has_parameter(request_spec, "action_type"));
    plan.assert_visible_lacks(&["list_available_plugins_to_install"]);
    plan.assert_registered_lacks(&["list_available_plugins_to_install"]);
}

#[tokio::test]
async fn code_mode_only_exposes_code_executor_and_hides_nested_tools() {
    let input = ToolPlanInputs {
        dynamic_tools: vec![dynamic_tool(
            Some("codex_app"),
            "lookup",
            /*defer_loading*/ false,
        )],
        ..ToolPlanInputs::default()
    };
    let plain = probe_with(|_| {}, input).await;
    assert_eq!(
        plain.namespace_function_names("codex_app"),
        &["lookup".to_string()]
    );
    plain.assert_visible_lacks(&[
        codex_code_mode::PUBLIC_TOOL_NAME,
        codex_code_mode::WAIT_TOOL_NAME,
    ]);

    let code_mode_only = probe_with(
        |turn| {
            set_features(turn, &[Feature::CodeMode, Feature::CodeModeOnly]);
        },
        ToolPlanInputs {
            dynamic_tools: vec![dynamic_tool(
                Some("codex_app"),
                "lookup",
                /*defer_loading*/ false,
            )],
            ..ToolPlanInputs::default()
        },
    )
    .await;
    code_mode_only.assert_visible_contains(&[
        codex_code_mode::PUBLIC_TOOL_NAME,
        codex_code_mode::WAIT_TOOL_NAME,
    ]);
    assert_eq!(
        code_mode_only.namespace_function_names("codex_app"),
        Vec::<String>::new().as_slice()
    );
}

#[tokio::test]
async fn code_mode_only_exposes_configured_dynamic_namespace_directly() {
    let plan = probe_with(
        |turn| {
            set_features(turn, &[Feature::CodeMode, Feature::CodeModeOnly]);
            turn.model_info.supports_search_tool = true;
            update_config(turn, |config| {
                config.code_mode.direct_only_tool_namespaces = vec!["direct_only".to_string()];
            });
        },
        ToolPlanInputs {
            dynamic_tools: vec![dynamic_tool(
                Some("direct_only"),
                "lookup",
                /*defer_loading*/ true,
            )],
            ..ToolPlanInputs::default()
        },
    )
    .await;

    plan.assert_visible_contains(&[
        codex_code_mode::PUBLIC_TOOL_NAME,
        codex_code_mode::WAIT_TOOL_NAME,
        "direct_only",
    ]);
    plan.assert_visible_lacks(&["tool_search"]);
    assert_eq!(
        plan.exposure(&ToolName::namespaced("direct_only", "lookup").to_string()),
        ToolExposure::DirectModelOnly
    );
    let ToolSpec::Namespace(namespace) = plan.visible_spec("direct_only") else {
        panic!("expected direct-only namespace spec");
    };
    let ResponsesApiNamespaceTool::Function(tool) = &namespace.tools[0];
    assert_eq!(tool.defer_loading, None);
    let ToolSpec::Freeform(exec) = plan.visible_spec(codex_code_mode::PUBLIC_TOOL_NAME) else {
        panic!("expected code mode exec tool");
    };
    assert!(!exec.description.contains("direct_only_lookup(args:"));
}

#[tokio::test]
async fn excluded_deferred_namespaces_do_not_enable_nested_tool_guidance() {
    let plan = probe_with(
        |turn| {
            set_features(turn, &[Feature::CodeMode, Feature::CodeModeOnly]);
            set_feature(turn, Feature::Collab, /*enabled*/ false);
            turn.model_info.supports_search_tool = true;
            update_config(turn, |config| {
                config.code_mode.excluded_tool_namespaces = vec!["excluded".to_string()];
            });
        },
        ToolPlanInputs {
            dynamic_tools: vec![dynamic_tool(
                Some("excluded"),
                "lookup",
                /*defer_loading*/ true,
            )],
            ..ToolPlanInputs::default()
        },
    )
    .await;

    let ToolSpec::Freeform(exec) = plan.visible_spec(codex_code_mode::PUBLIC_TOOL_NAME) else {
        panic!("expected code mode exec tool");
    };
    assert!(
        !exec
            .description
            .contains("Some deferred nested tools may be omitted")
    );
    plan.assert_registered_contains(&[
        &ToolName::namespaced("excluded", "lookup").to_string(),
        "tool_search",
    ]);
}

#[tokio::test]
async fn multi_agent_feature_selects_one_agent_tool_family() {
    let v1 = probe(|turn| {
        set_feature(turn, Feature::Collab, /*enabled*/ true);
        set_feature(turn, Feature::MultiAgentV2, /*enabled*/ false);
    })
    .await;
    v1.assert_visible_contains(&[MULTI_AGENT_V1_NAMESPACE]);
    v1.assert_visible_lacks(&[
        "spawn_agent",
        "send_input",
        "resume_agent",
        "wait_agent",
        "close_agent",
        "interrupt_agent",
        "send_message",
        "followup_task",
        "assign_task",
        "list_agents",
    ]);
    assert_eq!(
        v1.namespace_function_names(MULTI_AGENT_V1_NAMESPACE),
        &[
            "close_agent".to_string(),
            "resume_agent".to_string(),
            "send_input".to_string(),
            "spawn_agent".to_string(),
            "wait_agent".to_string(),
        ]
    );
    let ToolSpec::Namespace(namespace) = v1.visible_spec(MULTI_AGENT_V1_NAMESPACE) else {
        panic!("expected v1 multi-agent namespace");
    };
    let Some(ResponsesApiNamespaceTool::Function(spawn_agent)) =
        namespace.tools.iter().find(|tool| {
            matches!(
                tool,
                ResponsesApiNamespaceTool::Function(tool) if tool.name == "spawn_agent"
            )
        })
    else {
        panic!("expected v1 spawn_agent function");
    };
    let properties = spawn_agent
        .parameters
        .properties
        .as_ref()
        .expect("spawn_agent should use object params");
    for property in ["agent_type", "model", "reasoning_effort", "service_tier"] {
        assert!(
            properties.contains_key(property),
            "expected v1 spawn_agent to expose `{property}`"
        );
    }

    let v2 = probe(|turn| {
        set_feature(turn, Feature::MultiAgentV2, /*enabled*/ true);
        update_config(turn, |config| {
            config.multi_agent_v2.max_concurrent_threads_per_session = 17;
        });
    })
    .await;
    v2.assert_visible_contains(&[MULTI_AGENT_V2_NAMESPACE]);
    v2.assert_visible_lacks(&[
        "spawn_agent",
        "send_message",
        "followup_task",
        "wait_agent",
        "interrupt_agent",
        "list_agents",
        "send_input",
        "resume_agent",
        "assign_task",
        "close_agent",
    ]);
    for tool_name in [
        "spawn_agent",
        "send_message",
        "followup_task",
        "wait_agent",
        "interrupt_agent",
        "list_agents",
    ] {
        assert!(
            v2.namespace_function_names(MULTI_AGENT_V2_NAMESPACE)
                .iter()
                .any(|name| name == tool_name),
            "expected {tool_name} in {MULTI_AGENT_V2_NAMESPACE} namespace"
        );
    }
    let ToolSpec::Namespace(namespace) = v2.visible_spec(MULTI_AGENT_V2_NAMESPACE) else {
        panic!("expected {MULTI_AGENT_V2_NAMESPACE} namespace");
    };
    let Some(ResponsesApiNamespaceTool::Function(spawn_agent)) =
        namespace.tools.iter().find(|tool| {
            matches!(
                tool,
                ResponsesApiNamespaceTool::Function(tool) if tool.name == "spawn_agent"
            )
        })
    else {
        panic!("expected spawn_agent in {MULTI_AGENT_V2_NAMESPACE} namespace");
    };
    let spawn_agent_description = spawn_agent.description.as_str();
    assert!(!spawn_agent_description.contains("max_concurrent_threads_per_session"));
    assert!(spawn_agent_description.contains(
        "Note that passing `fork_turns=\"none\"` will not pass any surrounding context to the spawned subagent"
    ));

    let direct_model_only = probe(|turn| {
        set_features(
            turn,
            &[
                Feature::CodeMode,
                Feature::CodeModeOnly,
                Feature::MultiAgentV2,
            ],
        );
        update_config(turn, |config| {
            config.multi_agent_v2.non_code_mode_only = true;
        });
    })
    .await;
    direct_model_only.assert_visible_contains(&[MULTI_AGENT_V2_NAMESPACE]);
    direct_model_only.assert_visible_lacks(&["spawn_agent", "send_message", "wait_agent"]);
    assert_eq!(
        direct_model_only
            .exposure(&ToolName::namespaced(MULTI_AGENT_V2_NAMESPACE, "spawn_agent").to_string()),
        ToolExposure::DirectModelOnly
    );
}

#[tokio::test]
async fn multi_agent_v2_message_schemas_are_encrypted() {
    let plan = probe(|turn| {
        set_feature(turn, Feature::MultiAgentV2, /*enabled*/ true);
    })
    .await;
    let ToolSpec::Namespace(namespace) = plan.visible_spec(MULTI_AGENT_V2_NAMESPACE) else {
        panic!("expected {MULTI_AGENT_V2_NAMESPACE} namespace");
    };
    for tool_name in ["spawn_agent", "send_message", "followup_task"] {
        let Some(ResponsesApiNamespaceTool::Function(tool)) = namespace.tools.iter().find(|tool| {
            matches!(
                tool,
                ResponsesApiNamespaceTool::Function(tool) if tool.name == tool_name
            )
        }) else {
            panic!("expected {tool_name} in {MULTI_AGENT_V2_NAMESPACE} namespace");
        };
        let properties = tool
            .parameters
            .properties
            .as_ref()
            .expect("tool should use object params");
        assert_eq!(
            properties
                .get("message")
                .and_then(|schema| schema.encrypted),
            Some(true)
        );
    }
}

#[tokio::test]
async fn tool_mode_selector_overrides_feature_flags() {
    let direct = probe(|turn| {
        set_features(turn, &[Feature::CodeMode, Feature::CodeModeOnly]);
        turn.model_info.tool_mode = Some(ToolMode::Direct);
    })
    .await;
    direct.assert_visible_lacks(&[
        codex_code_mode::PUBLIC_TOOL_NAME,
        codex_code_mode::WAIT_TOOL_NAME,
    ]);
}

#[tokio::test]
async fn v1_multi_agent_tools_defer_when_tool_search_available() {
    let plan = probe(|turn| {
        turn.model_info.supports_search_tool = true;
        set_feature(turn, Feature::Collab, /*enabled*/ true);
        set_feature(turn, Feature::MultiAgentV2, /*enabled*/ false);
    })
    .await;

    plan.assert_visible_contains(&["tool_search"]);
    plan.assert_visible_lacks(&[
        "spawn_agent",
        "send_input",
        "resume_agent",
        "wait_agent",
        "close_agent",
        "interrupt_agent",
    ]);
    for tool_name in [
        "spawn_agent",
        "send_input",
        "resume_agent",
        "wait_agent",
        "close_agent",
    ] {
        let namespaced_tool_name = ToolName::namespaced(MULTI_AGENT_V1_NAMESPACE, tool_name);
        let namespaced_tool_name = namespaced_tool_name.to_string();
        assert!(
            plan.registered_names.contains(&namespaced_tool_name),
            "expected namespaced runtime for {tool_name}"
        );
        assert!(
            !plan
                .registered_names
                .contains(&ToolName::plain(tool_name).to_string()),
            "expected no plain runtime for deferred {tool_name}"
        );
        assert_eq!(plan.exposure(&namespaced_tool_name), ToolExposure::Deferred);
    }
    let ToolSpec::ToolSearch { description, .. } = plan.visible_spec("tool_search") else {
        panic!("expected visible tool_search spec");
    };
    assert!(description.contains("- Multi-agent tools: Spawn and manage sub-agents."));
}

#[tokio::test]
async fn multi_agent_v2_can_use_configured_tool_namespace() {
    let namespaced = probe(|turn| {
        set_feature(turn, Feature::MultiAgentV2, /*enabled*/ true);
        update_config(turn, |config| {
            config.multi_agent_v2.tool_namespace = Some("agents".to_string());
        });
    })
    .await;

    namespaced.assert_visible_contains(&["agents"]);
    namespaced.assert_visible_lacks(&["assign_task"]);
    assert!(
        !namespaced
            .registered_names
            .contains(&ToolName::namespaced("agents", "assign_task").to_string()),
        "expected no namespaced runtime for assign_task"
    );
    assert!(
        !namespaced
            .namespace_function_names("agents")
            .iter()
            .any(|name| name == "assign_task"),
        "expected assign_task to be absent from agents namespace"
    );
    for tool_name in [
        "spawn_agent",
        "send_message",
        "followup_task",
        "wait_agent",
        "interrupt_agent",
        "list_agents",
    ] {
        namespaced.assert_visible_lacks(&[tool_name]);
        assert!(
            namespaced
                .registered_names
                .contains(&ToolName::namespaced("agents", tool_name).to_string()),
            "expected namespaced runtime for {tool_name}"
        );
        assert!(
            !namespaced
                .registered_names
                .contains(&ToolName::plain(tool_name).to_string()),
            "expected no plain runtime for {tool_name}"
        );
        assert!(
            namespaced
                .namespace_function_names("agents")
                .iter()
                .any(|name| name == tool_name),
            "expected {tool_name} in agents namespace"
        );
    }
}

#[tokio::test]
async fn multi_agent_v2_namespace_is_supported_by_bedrock_provider() {
    let plan = probe(|turn| {
        set_feature(turn, Feature::MultiAgentV2, /*enabled*/ true);
        update_config(turn, |config| {
            config.multi_agent_v2.tool_namespace = Some("agents".to_string());
        });
        use_bedrock_provider(turn);
    })
    .await;

    plan.assert_visible_contains(&["agents"]);
    plan.assert_visible_lacks(&["spawn_agent", "send_message", "list_agents"]);
    assert!(
        !plan
            .registered_names
            .contains(&ToolName::plain("spawn_agent").to_string())
    );
    assert!(
        plan.registered_names
            .contains(&ToolName::namespaced("agents", "spawn_agent").to_string())
    );
}

#[tokio::test]
async fn code_mode_only_can_expose_namespaced_multi_agent_v2_as_normal_tools() {
    let plan = probe(|turn| {
        set_features(
            turn,
            &[
                Feature::CodeMode,
                Feature::CodeModeOnly,
                Feature::MultiAgentV2,
            ],
        );
        update_config(turn, |config| {
            config.multi_agent_v2.non_code_mode_only = true;
            config.multi_agent_v2.tool_namespace = Some("agents".to_string());
        });
    })
    .await;

    assert_eq!(
        plan.visible_names,
        vec![
            "exec",
            "wait",
            "request_user_input",
            "agents",
            // Hosted Responses tool.
            "web_search",
        ]
    );
    assert!(
        !plan
            .namespace_function_names("agents")
            .iter()
            .any(|name| name == "assign_task"),
        "expected assign_task to be absent from agents namespace"
    );
    for tool_name in [
        "spawn_agent",
        "send_message",
        "followup_task",
        "wait_agent",
        "interrupt_agent",
        "list_agents",
    ] {
        assert!(
            plan.namespace_function_names("agents")
                .iter()
                .any(|name| name == tool_name),
            "expected {tool_name} in agents namespace"
        );
    }
}

#[tokio::test]
async fn hosted_web_search_and_standalone_image_generation_follow_runtime_gates() {
    let image_generation_tool = Arc::new(TestNamespaceExtensionTool {
        namespace: "image_gen",
        tool_name: "imagegen",
    });
    let image_generation = probe_with(
        |turn| {
            use_chatgpt_auth(turn);
            turn.model_info.input_modalities = vec![InputModality::Image];
        },
        ToolPlanInputs {
            extension_tool_executors: vec![image_generation_tool.clone()],
            ..Default::default()
        },
    )
    .await;
    image_generation.assert_visible_contains(&["image_gen"]);

    let extension_disabled = probe_with(
        |turn| {
            use_chatgpt_auth(turn);
            set_feature(turn, Feature::ImageGeneration, /*enabled*/ false);
            turn.model_info.input_modalities = vec![InputModality::Image];
        },
        ToolPlanInputs {
            extension_tool_executors: vec![image_generation_tool.clone()],
            ..Default::default()
        },
    )
    .await;
    extension_disabled.assert_visible_lacks(&["image_gen"]);

    let text_only_model = probe_with(
        |turn| {
            use_chatgpt_auth(turn);
            turn.model_info.input_modalities = vec![];
        },
        ToolPlanInputs {
            extension_tool_executors: vec![image_generation_tool.clone()],
            ..Default::default()
        },
    )
    .await;
    text_only_model.assert_visible_lacks(&["image_gen"]);

    let unsupported_provider = probe_with(
        |turn| {
            use_bedrock_provider(turn);
            turn.model_info.input_modalities = vec![InputModality::Image];
        },
        ToolPlanInputs {
            extension_tool_executors: vec![image_generation_tool],
            ..Default::default()
        },
    )
    .await;
    unsupported_provider.assert_visible_lacks(&["image_gen"]);

    let live_web_search = probe(|turn| {
        set_web_search_mode(turn, WebSearchMode::Live);
        turn.model_info.web_search_tool_type = WebSearchToolType::TextAndImage;
    })
    .await;
    assert_eq!(
        live_web_search.visible_spec("web_search"),
        &ToolSpec::WebSearch {
            external_web_access: Some(true),
            indexed_web_access: None,
            filters: None,
            user_location: None,
            search_context_size: None,
            search_content_types: Some(vec!["text".to_string(), "image".to_string()]),
        }
    );

    let code_mode_only = probe(|turn| {
        use_chatgpt_auth(turn);
        set_features(turn, &[Feature::CodeModeOnly, Feature::MultiAgentV2]);
        set_web_search_mode(turn, WebSearchMode::Live);
        turn.model_info.input_modalities = vec![InputModality::Image];
    })
    .await;
    assert_eq!(
        code_mode_only.visible_names,
        vec![
            // Code-mode entrypoints.
            codex_code_mode::PUBLIC_TOOL_NAME,
            codex_code_mode::WAIT_TOOL_NAME,
            "request_user_input",
            // Multi-agent v2 tools.
            MULTI_AGENT_V2_NAMESPACE,
            // Hosted Responses tools.
            "web_search",
        ]
    );

    let standalone_web_search_without_web_run = probe(|turn| {
        set_feature(turn, Feature::StandaloneWebSearch, /*enabled*/ true);
        set_web_search_mode(turn, WebSearchMode::Live);
    })
    .await;
    standalone_web_search_without_web_run.assert_visible_contains(&["web_search"]);

    let standalone_web_search = probe_with(
        |turn| {
            set_feature(turn, Feature::StandaloneWebSearch, /*enabled*/ true);
            set_web_search_mode(turn, WebSearchMode::Live);
        },
        ToolPlanInputs {
            extension_tool_executors: vec![Arc::new(TestNamespaceExtensionTool {
                namespace: "web",
                tool_name: "run",
            })],
            ..Default::default()
        },
    )
    .await;
    standalone_web_search.assert_visible_lacks(&["web_search"]);

    let unsupported_provider = probe(|turn| {
        set_web_search_mode(turn, WebSearchMode::Live);
        use_bedrock_provider(turn);
    })
    .await;
    unsupported_provider.assert_visible_lacks(&["web_search"]);
}
