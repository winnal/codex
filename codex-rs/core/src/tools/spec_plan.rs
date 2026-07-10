use crate::agent::exceeds_thread_spawn_depth_limit;
use crate::agent::next_thread_spawn_depth;
use crate::session::step_context::StepContext;
use crate::session::turn_context::TurnContext;
use crate::tools::code_mode::execute_spec::create_code_mode_tool;
use crate::tools::context::ToolInvocation;
use crate::tools::effective_tool_mode;
use crate::tools::exact_tail_continuity::ExactTailToolSurfaceOutcome;
use crate::tools::exact_tail_continuity::PendingExactTailToolSurfaceHint;
use crate::tools::handlers::ApplyPatchHandler;
use crate::tools::handlers::CodeModeExecuteHandler;
use crate::tools::handlers::CodeModeWaitHandler;
use crate::tools::handlers::CurrentTimeHandler;
use crate::tools::handlers::DynamicToolHandler;
use crate::tools::handlers::ExecCommandHandler;
use crate::tools::handlers::ExecCommandHandlerOptions;
use crate::tools::handlers::GetContextRemainingHandler;
use crate::tools::handlers::ListAvailablePluginsToInstallHandler;
use crate::tools::handlers::ListMcpResourceTemplatesHandler;
use crate::tools::handlers::ListMcpResourcesHandler;
use crate::tools::handlers::McpHandler;
use crate::tools::handlers::NewContextWindowHandler;
use crate::tools::handlers::PlanHandler;
use crate::tools::handlers::ReadMcpResourceHandler;
use crate::tools::handlers::RequestPermissionsHandler;
use crate::tools::handlers::RequestPluginInstallHandler;
use crate::tools::handlers::RequestUserInputHandler;
use crate::tools::handlers::ShellCommandHandler;
use crate::tools::handlers::ShellCommandHandlerOptions;
use crate::tools::handlers::SleepHandler;
use crate::tools::handlers::TestSyncHandler;
use crate::tools::handlers::ToolSearchHandlerCache;
use crate::tools::handlers::ViewImageHandler;
use crate::tools::handlers::WaitForEnvironmentHandler;
use crate::tools::handlers::WriteStdinHandler;
use crate::tools::handlers::agent_jobs::ReportAgentJobResultHandler;
use crate::tools::handlers::agent_jobs::SpawnAgentsOnCsvHandler;
use crate::tools::handlers::extension_tools::ExtensionToolAdapter;
use crate::tools::handlers::multi_agents::CloseAgentHandler;
use crate::tools::handlers::multi_agents::ResumeAgentHandler;
use crate::tools::handlers::multi_agents::SendInputHandler;
use crate::tools::handlers::multi_agents::SpawnAgentHandler;
use crate::tools::handlers::multi_agents::WaitAgentHandler;
use crate::tools::handlers::multi_agents_common::DEFAULT_WAIT_TIMEOUT_MS;
use crate::tools::handlers::multi_agents_common::MAX_WAIT_TIMEOUT_MS;
use crate::tools::handlers::multi_agents_common::MIN_WAIT_TIMEOUT_MS;
use crate::tools::handlers::multi_agents_spec::MULTI_AGENT_V1_NAMESPACE;
use crate::tools::handlers::multi_agents_spec::SpawnAgentToolOptions;
use crate::tools::handlers::multi_agents_spec::WaitAgentTimeoutOptions;
use crate::tools::handlers::multi_agents_v2::FollowupTaskHandler as FollowupTaskHandlerV2;
use crate::tools::handlers::multi_agents_v2::InterruptAgentHandler;
use crate::tools::handlers::multi_agents_v2::ListAgentsHandler as ListAgentsHandlerV2;
use crate::tools::handlers::multi_agents_v2::SendMessageHandler as SendMessageHandlerV2;
use crate::tools::handlers::multi_agents_v2::SpawnAgentHandler as SpawnAgentHandlerV2;
use crate::tools::handlers::multi_agents_v2::WaitAgentHandler as WaitAgentHandlerV2;
use crate::tools::handlers::view_image_spec::ViewImageToolOptions;
use crate::tools::hosted_spec::WebSearchToolOptions;
use crate::tools::hosted_spec::create_web_search_tool;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::ToolExposure;
use crate::tools::registry::ToolRegistry;
use crate::tools::registry::override_tool_exposure;
use crate::tools::router::ToolRouter;
use crate::tools::router::ToolRouterParams;
use codex_features::Feature;
use codex_login::AuthManager;
use codex_mcp::ToolInfo;
use codex_protocol::config_types::WebSearchMode;
use codex_protocol::dynamic_tools::DynamicToolNamespaceTool;
use codex_protocol::dynamic_tools::DynamicToolSpec;
use codex_protocol::openai_models::ConfigShellToolType;
use codex_protocol::openai_models::InputModality;
use codex_protocol::openai_models::ToolMode;
use codex_protocol::protocol::MultiAgentVersion;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentSource;
use codex_tools::ResponsesApiNamespace;
use codex_tools::ResponsesApiNamespaceTool;
use codex_tools::TOOL_SEARCH_TOOL_NAME;
use codex_tools::ToolCall as ExtensionToolCall;
use codex_tools::ToolEnvironmentMode;
use codex_tools::ToolExecutor;
use codex_tools::ToolName;
use codex_tools::ToolSearchInfo;
use codex_tools::ToolSpec;
use codex_tools::UnifiedExecShellMode;
use codex_tools::can_request_original_image_detail;
use codex_tools::collect_code_mode_exec_prompt_tool_definitions;
use codex_tools::collect_request_plugin_install_entries;
use codex_tools::default_namespace_description;
use codex_tools::request_user_input_available_modes;
use codex_tools::shell_command_backend_for_features;
use codex_tools::shell_type_for_model_and_features;
use codex_utils_string::approx_token_count;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashSet;
use std::sync::Arc;
use tracing::instrument;
use tracing::warn;

const MULTI_AGENT_V2_NAMESPACE_DESCRIPTION: &str = "Tools for spawning and managing sub-agents.";
const IMAGE_GEN_NAMESPACE: &str = "image_gen";
const IMAGEGEN_TOOL_NAME: &str = "imagegen";
const EXACT_TAIL_PROMOTED_TOOL_SPEC_TOKEN_LIMIT: usize = 1_000;
const EXACT_TAIL_PROMOTED_NAMESPACE_SPEC_TOKEN_LIMIT: usize = 8_000;
const EXACT_TAIL_PROMOTED_TOOL_SPECS_TOTAL_TOKEN_LIMIT: usize = 12_000;

type PlannedRuntime = Arc<dyn CoreToolRuntime>;

#[derive(Default)]
struct PlannedTools {
    runtimes: Vec<PlannedRuntime>,
    hosted_specs: Vec<ToolSpec>,
}

impl PlannedTools {
    fn add<T>(&mut self, handler: T)
    where
        T: CoreToolRuntime + 'static,
    {
        self.runtimes.push(Arc::new(handler));
    }

    fn add_arc(&mut self, handler: PlannedRuntime) {
        self.runtimes.push(handler);
    }

    fn add_with_exposure<T>(&mut self, handler: T, exposure: ToolExposure)
    where
        T: CoreToolRuntime + 'static,
    {
        self.runtimes
            .push(override_tool_exposure(Arc::new(handler), exposure));
    }

    fn add_dispatch_only<T>(&mut self, handler: T)
    where
        T: CoreToolRuntime + 'static,
    {
        self.add_with_exposure(handler, ToolExposure::Hidden);
    }

    fn add_hosted_spec(&mut self, spec: ToolSpec) {
        self.hosted_specs.push(spec);
    }

    fn runtimes(&self) -> &[PlannedRuntime] {
        &self.runtimes
    }
}

#[derive(Clone, Copy)]
struct CoreToolPlanContext<'a> {
    step_context: &'a StepContext,
    mcp_tools: Option<&'a [ToolInfo]>,
    deferred_mcp_tools: Option<&'a [ToolInfo]>,
    tool_suggest_candidates: Option<&'a crate::tools::router::ToolSuggestCandidates>,
    extension_tool_executors: &'a [Arc<dyn ToolExecutor<ExtensionToolCall>>],
    dynamic_tools: &'a [DynamicToolSpec],
    exact_tail_tool_surface_hint: Option<&'a PendingExactTailToolSurfaceHint>,
    tool_search_handler_cache: &'a ToolSearchHandlerCache,
    default_agent_type_description: &'a str,
    wait_agent_timeouts: WaitAgentTimeoutOptions,
}

#[instrument(level = "trace", skip_all)]
pub(crate) fn build_tool_router(
    step_context: &StepContext,
    params: ToolRouterParams<'_>,
    tool_search_handler_cache: &ToolSearchHandlerCache,
) -> ToolRouter {
    let (model_visible_specs, registry, exact_tail_tool_surface_outcome) =
        build_tool_specs_and_registry(step_context, params, tool_search_handler_cache);
    ToolRouter::from_parts_with_exact_tail_tool_surface_outcome(
        registry,
        model_visible_specs,
        exact_tail_tool_surface_outcome,
    )
}

#[instrument(level = "trace", skip_all)]
fn build_tool_specs_and_registry(
    step_context: &StepContext,
    params: ToolRouterParams<'_>,
    tool_search_handler_cache: &ToolSearchHandlerCache,
) -> (
    Vec<ToolSpec>,
    ToolRegistry,
    Option<ExactTailToolSurfaceOutcome>,
) {
    let turn_context = step_context.turn.as_ref();
    let ToolRouterParams {
        mcp_tools,
        deferred_mcp_tools,
        tool_suggest_candidates,
        extension_tool_executors,
        dynamic_tools,
        exact_tail_tool_surface_hint,
    } = params;
    let default_agent_type_description =
        crate::agent::role::spawn_tool_spec::build(&std::collections::BTreeMap::new());
    let context = CoreToolPlanContext {
        step_context,
        mcp_tools: mcp_tools.as_deref(),
        deferred_mcp_tools: deferred_mcp_tools.as_deref(),
        tool_suggest_candidates: tool_suggest_candidates.as_ref(),
        extension_tool_executors: &extension_tool_executors,
        dynamic_tools,
        exact_tail_tool_surface_hint: exact_tail_tool_surface_hint.as_ref(),
        tool_search_handler_cache,
        default_agent_type_description: &default_agent_type_description,
        wait_agent_timeouts: wait_agent_timeout_options(turn_context),
    };
    let mut planned_tools = PlannedTools::default();
    add_tool_sources(&context, &mut planned_tools);
    apply_direct_model_only_namespace_overrides(turn_context, &mut planned_tools);
    let exact_tail_tool_surface_outcome =
        apply_exact_tail_tool_surface_continuity(&context, &mut planned_tools);
    append_tool_search_executor(&context, &mut planned_tools);
    prepend_code_mode_executors(&context, &mut planned_tools);
    let (model_visible_specs, registry) =
        build_model_visible_specs_and_registry(turn_context, planned_tools);
    (
        model_visible_specs,
        registry,
        exact_tail_tool_surface_outcome,
    )
}

fn apply_exact_tail_tool_surface_continuity(
    context: &CoreToolPlanContext<'_>,
    planned_tools: &mut PlannedTools,
) -> Option<ExactTailToolSurfaceOutcome> {
    let pending = context.exact_tail_tool_surface_hint?;
    let mut state = ExactTailToolSurfaceContinuityState::default();
    let mut processed_references = BTreeSet::new();
    let tool_search_visible = search_tool_enabled(context.step_context.turn.as_ref());

    for namespace in exact_tail_atomic_namespaces(&pending.hint.references) {
        apply_exact_tail_atomic_namespace_continuity(
            context,
            planned_tools,
            &namespace,
            tool_search_visible,
            &mut state,
            &mut processed_references,
        );
    }

    for reference in &pending.hint.references {
        if processed_references.contains(reference) {
            continue;
        }

        apply_exact_tail_reference_continuity(
            context,
            planned_tools,
            reference,
            tool_search_visible,
            &mut state,
        );
    }

    let should_emit_notice = !pending.notice_already_emitted
        && (state.missing_hot_tool_count > 0 || pending.hint.hot_tool_reference_overflow_count > 0);
    let missing_notice_emitted_count = usize::from(should_emit_notice);
    let hot_tool_reference_overflow_notice_emitted_count =
        usize::from(should_emit_notice && pending.hint.hot_tool_reference_overflow_count > 0);
    let tool_surface_changed_after_compaction =
        state.rehydrated_tool_count > 0 || missing_notice_emitted_count > 0;
    Some(ExactTailToolSurfaceOutcome {
        compaction_id: pending.compaction_id.clone(),
        route: pending.route.clone(),
        hot_tool_call_count: pending.hint.hot_tool_call_count,
        hot_tool_namespace_count: pending.hint.hot_tool_namespace_count,
        hot_tool_reference_count: pending.hint.hot_tool_reference_count,
        rehydrated_tool_count: state.rehydrated_tool_count,
        missing_hot_tool_count: state.missing_hot_tool_count,
        already_direct_tool_count: state.already_direct_tool_count,
        discoverable_hot_tool_count: state.discoverable_hot_tool_count,
        missing_notice_emitted_count,
        missing_no_path_count: state.missing_no_path_count,
        rehydrated_tool_references: state.rehydrated_tool_references,
        missing_tool_references: state.missing_tool_references,
        rehydrated_tool_namespaces: state
            .rehydrated_tool_namespaces
            .into_iter()
            .collect::<Vec<_>>(),
        missing_tool_rejection_reasons: state
            .missing_tool_rejection_reasons
            .into_iter()
            .collect::<Vec<_>>(),
        hot_tool_reference_overflow_count: pending.hint.hot_tool_reference_overflow_count,
        hot_tool_reference_overflow_notice_emitted_count,
        out_of_scope_dependency_protocol_count: pending.hint.out_of_scope_dependency_protocol_count,
        tool_surface_changed_after_compaction,
        tool_surface_rehydration_failure_reason: (state.missing_hot_tool_count > 0)
            .then(|| "hot_tool_references_not_model_visible".to_string()),
    })
}

#[derive(Default)]
struct ExactTailToolSurfaceContinuityState {
    rehydrated_tool_count: usize,
    missing_hot_tool_count: usize,
    already_direct_tool_count: usize,
    discoverable_hot_tool_count: usize,
    missing_no_path_count: usize,
    rehydrated_tool_spec_tokens: usize,
    rehydrated_tool_references: Vec<String>,
    missing_tool_references: Vec<String>,
    rehydrated_tool_namespaces: BTreeSet<String>,
    missing_tool_rejection_reasons: BTreeSet<String>,
}

impl ExactTailToolSurfaceContinuityState {
    fn record_already_direct(&mut self) {
        self.already_direct_tool_count += 1;
    }

    fn record_discoverable(&mut self) {
        self.discoverable_hot_tool_count += 1;
    }

    fn record_rehydrated(&mut self, reference: &ToolName, spec_tokens: usize) {
        self.rehydrated_tool_count += 1;
        self.rehydrated_tool_spec_tokens =
            self.rehydrated_tool_spec_tokens.saturating_add(spec_tokens);
        self.rehydrated_tool_references
            .push(tool_reference_label(reference));
        if let Some(namespace) = &reference.namespace {
            self.rehydrated_tool_namespaces.insert(namespace.clone());
        }
    }

    fn record_rehydrated_namespace(
        &mut self,
        namespace: &str,
        references: &[ToolName],
        spec_tokens: usize,
    ) {
        self.rehydrated_tool_count += references.len();
        self.rehydrated_tool_spec_tokens =
            self.rehydrated_tool_spec_tokens.saturating_add(spec_tokens);
        self.rehydrated_tool_references
            .extend(references.iter().map(tool_reference_label));
        self.rehydrated_tool_namespaces
            .insert(namespace.to_string());
    }

    fn record_missing(
        &mut self,
        reference: &ToolName,
        reason: ExactTailMissingToolReason,
        has_discovery_path: bool,
    ) {
        self.missing_hot_tool_count += 1;
        if !has_discovery_path {
            self.missing_no_path_count += 1;
        }
        self.missing_tool_references
            .push(tool_reference_label(reference));
        self.missing_tool_rejection_reasons
            .insert(reason.as_str().to_string());
    }
}

#[derive(Clone, Copy)]
enum ExactTailMissingToolReason {
    NoRuntime,
    NotModelVisible,
    SpecBudget,
}

impl ExactTailMissingToolReason {
    fn as_str(self) -> &'static str {
        match self {
            ExactTailMissingToolReason::NoRuntime => "no_runtime",
            ExactTailMissingToolReason::NotModelVisible => "not_model_visible",
            ExactTailMissingToolReason::SpecBudget => "spec_budget",
        }
    }
}

fn exact_tail_atomic_namespaces(references: &[ToolName]) -> BTreeSet<String> {
    references
        .iter()
        .filter_map(|reference| reference.namespace.as_deref())
        .filter(|namespace| exact_tail_namespace_is_atomic(namespace))
        .map(ToString::to_string)
        .collect()
}

fn exact_tail_namespace_is_atomic(namespace: &str) -> bool {
    namespace == MULTI_AGENT_V1_NAMESPACE
}

fn apply_exact_tail_atomic_namespace_continuity(
    context: &CoreToolPlanContext<'_>,
    planned_tools: &mut PlannedTools,
    namespace: &str,
    tool_search_visible: bool,
    state: &mut ExactTailToolSurfaceContinuityState,
    processed_references: &mut BTreeSet<ToolName>,
) {
    let Some(pending) = context.exact_tail_tool_surface_hint else {
        return;
    };
    let references = pending
        .hint
        .references
        .iter()
        .filter(|reference| reference.namespace.as_deref() == Some(namespace))
        .cloned()
        .collect::<Vec<_>>();
    if references.is_empty() {
        return;
    }
    processed_references.extend(references.iter().cloned());

    let namespace_runtime_indices = planned_tools
        .runtimes
        .iter()
        .enumerate()
        .filter_map(|(index, runtime)| {
            (runtime.tool_name().namespace.as_deref() == Some(namespace)).then_some(index)
        })
        .collect::<Vec<_>>();
    if namespace_runtime_indices.is_empty() {
        for reference in &references {
            state.record_missing(
                reference,
                ExactTailMissingToolReason::NoRuntime,
                /*has_discovery_path*/ false,
            );
        }
        return;
    }

    let mut reference_runtime_indices = Vec::new();
    for reference in &references {
        let Some(index) = namespace_runtime_indices
            .iter()
            .copied()
            .find(|index| planned_tools.runtimes[*index].tool_name() == *reference)
        else {
            state.record_missing(
                reference,
                ExactTailMissingToolReason::NoRuntime,
                /*has_discovery_path*/ false,
            );
            continue;
        };
        reference_runtime_indices.push((reference.clone(), index));
    }
    if reference_runtime_indices.len() != references.len() {
        return;
    }

    let deferred_references = reference_runtime_indices
        .iter()
        .filter(|&(_reference, index)| {
            planned_tools.runtimes[*index].exposure() == ToolExposure::Deferred
        })
        .map(|(reference, _index)| reference.clone())
        .collect::<Vec<_>>();
    if deferred_references.is_empty() {
        record_exact_tail_atomic_namespace_without_promotion(
            context,
            planned_tools,
            &reference_runtime_indices,
            tool_search_visible,
            state,
            ExactTailMissingToolReason::NotModelVisible,
        );
        return;
    }

    if reference_runtime_indices.iter().any(|(reference, index)| {
        let runtime = &planned_tools.runtimes[*index];
        runtime.exposure() == ToolExposure::Deferred
            && (runtime.search_info().is_none()
                || !tool_search_visible
                || is_hidden_by_code_mode_only(
                    context.step_context.turn.as_ref(),
                    reference,
                    ToolExposure::Direct,
                ))
    }) {
        record_exact_tail_atomic_namespace_without_promotion(
            context,
            planned_tools,
            &reference_runtime_indices,
            tool_search_visible,
            state,
            ExactTailMissingToolReason::NotModelVisible,
        );
        return;
    }

    match exact_tail_namespace_promotion_eligibility(
        context.step_context.turn.as_ref(),
        planned_tools,
        &namespace_runtime_indices,
        &references,
        state.rehydrated_tool_spec_tokens,
    ) {
        ExactTailPromotionEligibility::Eligible { spec_tokens } => {
            for (reference, index) in &reference_runtime_indices {
                match planned_tools.runtimes[*index].exposure() {
                    ToolExposure::Deferred => {
                        if planned_tools.runtimes[*index].search_info().is_some()
                            && tool_search_visible
                        {
                            state.record_discoverable();
                        }
                    }
                    exposure
                        if exact_tail_runtime_is_model_visible(
                            context.step_context.turn.as_ref(),
                            planned_tools,
                            *index,
                            exposure,
                        ) =>
                    {
                        state.record_already_direct();
                    }
                    ToolExposure::Direct | ToolExposure::DirectModelOnly | ToolExposure::Hidden => {
                        state.record_missing(
                            reference,
                            ExactTailMissingToolReason::NotModelVisible,
                            /*has_discovery_path*/ false,
                        );
                    }
                }
            }
            for index in namespace_runtime_indices {
                if planned_tools.runtimes[index].exposure() == ToolExposure::Deferred {
                    let runtime = Arc::clone(&planned_tools.runtimes[index]);
                    planned_tools.runtimes[index] =
                        override_tool_exposure(runtime, ToolExposure::Direct);
                }
            }
            state.record_rehydrated_namespace(namespace, &deferred_references, spec_tokens);
        }
        ExactTailPromotionEligibility::NotModelVisible => {
            record_exact_tail_atomic_namespace_without_promotion(
                context,
                planned_tools,
                &reference_runtime_indices,
                tool_search_visible,
                state,
                ExactTailMissingToolReason::NotModelVisible,
            );
        }
        ExactTailPromotionEligibility::ExceedsSpecBudget => {
            record_exact_tail_atomic_namespace_without_promotion(
                context,
                planned_tools,
                &reference_runtime_indices,
                tool_search_visible,
                state,
                ExactTailMissingToolReason::SpecBudget,
            );
        }
    }
}

fn record_exact_tail_atomic_namespace_without_promotion(
    context: &CoreToolPlanContext<'_>,
    planned_tools: &PlannedTools,
    reference_runtime_indices: &[(ToolName, usize)],
    tool_search_visible: bool,
    state: &mut ExactTailToolSurfaceContinuityState,
    missing_reason: ExactTailMissingToolReason,
) {
    for (reference, index) in reference_runtime_indices {
        let exposure = planned_tools.runtimes[*index].exposure();
        match exposure {
            ToolExposure::Direct | ToolExposure::DirectModelOnly
                if !is_hidden_by_code_mode_only(
                    context.step_context.turn.as_ref(),
                    reference,
                    exposure,
                ) && exact_tail_runtime_is_model_visible(
                    context.step_context.turn.as_ref(),
                    planned_tools,
                    *index,
                    exposure,
                ) =>
            {
                state.record_already_direct();
            }
            ToolExposure::Deferred => {
                let has_discovery_path =
                    planned_tools.runtimes[*index].search_info().is_some() && tool_search_visible;
                if has_discovery_path {
                    state.record_discoverable();
                }
                state.record_missing(
                    reference,
                    missing_reason,
                    matches!(missing_reason, ExactTailMissingToolReason::SpecBudget)
                        && has_discovery_path,
                );
            }
            ToolExposure::Direct | ToolExposure::DirectModelOnly | ToolExposure::Hidden => {
                state.record_missing(
                    reference,
                    ExactTailMissingToolReason::NotModelVisible,
                    /*has_discovery_path*/ false,
                );
            }
        }
    }
}

fn apply_exact_tail_reference_continuity(
    context: &CoreToolPlanContext<'_>,
    planned_tools: &mut PlannedTools,
    reference: &ToolName,
    tool_search_visible: bool,
    state: &mut ExactTailToolSurfaceContinuityState,
) {
    let Some(index) = planned_tools
        .runtimes
        .iter()
        .position(|runtime| runtime.tool_name() == *reference)
    else {
        state.record_missing(
            reference,
            ExactTailMissingToolReason::NoRuntime,
            /*has_discovery_path*/ false,
        );
        return;
    };

    let exposure = planned_tools.runtimes[index].exposure();
    match exposure {
        ToolExposure::Direct | ToolExposure::DirectModelOnly
            if !is_hidden_by_code_mode_only(
                context.step_context.turn.as_ref(),
                reference,
                exposure,
            ) =>
        {
            if exact_tail_runtime_is_model_visible(
                context.step_context.turn.as_ref(),
                planned_tools,
                index,
                exposure,
            ) {
                state.record_already_direct();
            } else {
                state.record_missing(
                    reference,
                    ExactTailMissingToolReason::NotModelVisible,
                    /*has_discovery_path*/ false,
                );
            }
        }
        ToolExposure::Deferred => {
            let runtime = Arc::clone(&planned_tools.runtimes[index]);
            if runtime.search_info().is_none() || !tool_search_visible {
                state.record_missing(
                    reference,
                    ExactTailMissingToolReason::NotModelVisible,
                    /*has_discovery_path*/ false,
                );
                return;
            }

            state.record_discoverable();
            match exact_tail_promotion_eligibility(
                context.step_context.turn.as_ref(),
                planned_tools,
                index,
                reference,
                state.rehydrated_tool_spec_tokens,
            ) {
                ExactTailPromotionEligibility::Eligible { spec_tokens } => {
                    planned_tools.runtimes[index] =
                        override_tool_exposure(runtime, ToolExposure::Direct);
                    state.record_rehydrated(reference, spec_tokens);
                }
                ExactTailPromotionEligibility::NotModelVisible => {
                    state.record_missing(
                        reference,
                        ExactTailMissingToolReason::NotModelVisible,
                        /*has_discovery_path*/ false,
                    );
                }
                ExactTailPromotionEligibility::ExceedsSpecBudget => {
                    state.record_missing(
                        reference,
                        ExactTailMissingToolReason::SpecBudget,
                        /*has_discovery_path*/ true,
                    );
                }
            }
        }
        ToolExposure::Direct | ToolExposure::DirectModelOnly | ToolExposure::Hidden => {
            state.record_missing(
                reference,
                ExactTailMissingToolReason::NotModelVisible,
                /*has_discovery_path*/ false,
            );
        }
    }
}

fn tool_reference_label(reference: &ToolName) -> String {
    match &reference.namespace {
        Some(namespace) => format!("{namespace}.{}", reference.name),
        None => reference.name.clone(),
    }
}

enum ExactTailPromotionEligibility {
    Eligible { spec_tokens: usize },
    NotModelVisible,
    ExceedsSpecBudget,
}

fn exact_tail_promotion_eligibility(
    turn_context: &TurnContext,
    planned_tools: &PlannedTools,
    runtime_index: usize,
    tool_name: &ToolName,
    rehydrated_tool_spec_tokens: usize,
) -> ExactTailPromotionEligibility {
    if is_hidden_by_code_mode_only(turn_context, tool_name, ToolExposure::Direct) {
        return ExactTailPromotionEligibility::NotModelVisible;
    }

    let specs = exact_tail_model_visible_specs_with_runtime_exposure(
        turn_context,
        planned_tools,
        runtime_index,
        ToolExposure::Direct,
        tool_name,
    );
    if specs.is_empty() {
        return ExactTailPromotionEligibility::NotModelVisible;
    }

    let spec_tokens = specs
        .iter()
        .map(serialized_tool_spec_token_count)
        .try_fold(0usize, usize::checked_add)
        .unwrap_or(usize::MAX);
    if specs
        .iter()
        .map(serialized_tool_spec_token_count)
        .any(|tokens| tokens > EXACT_TAIL_PROMOTED_TOOL_SPEC_TOKEN_LIMIT)
        || rehydrated_tool_spec_tokens.saturating_add(spec_tokens)
            > EXACT_TAIL_PROMOTED_TOOL_SPECS_TOTAL_TOKEN_LIMIT
    {
        return ExactTailPromotionEligibility::ExceedsSpecBudget;
    }

    ExactTailPromotionEligibility::Eligible { spec_tokens }
}

fn exact_tail_namespace_promotion_eligibility(
    turn_context: &TurnContext,
    planned_tools: &PlannedTools,
    namespace_runtime_indices: &[usize],
    tool_names: &[ToolName],
    rehydrated_tool_spec_tokens: usize,
) -> ExactTailPromotionEligibility {
    if tool_names
        .iter()
        .any(|tool_name| is_hidden_by_code_mode_only(turn_context, tool_name, ToolExposure::Direct))
    {
        return ExactTailPromotionEligibility::NotModelVisible;
    }

    let runtime_overrides = namespace_runtime_indices
        .iter()
        .copied()
        .filter(|index| planned_tools.runtimes[*index].exposure() == ToolExposure::Deferred)
        .map(|index| (index, ToolExposure::Direct))
        .collect::<Vec<_>>();
    if runtime_overrides.is_empty() {
        return ExactTailPromotionEligibility::NotModelVisible;
    }

    let specs = exact_tail_model_visible_specs_with_runtime_exposures(
        turn_context,
        planned_tools,
        &runtime_overrides,
        tool_names,
    );
    if specs.is_empty() {
        return ExactTailPromotionEligibility::NotModelVisible;
    }

    let spec_tokens = specs
        .iter()
        .map(serialized_tool_spec_token_count)
        .try_fold(0usize, usize::checked_add)
        .unwrap_or(usize::MAX);
    if specs
        .iter()
        .map(serialized_tool_spec_token_count)
        .any(|tokens| tokens > EXACT_TAIL_PROMOTED_NAMESPACE_SPEC_TOKEN_LIMIT)
        || rehydrated_tool_spec_tokens.saturating_add(spec_tokens)
            > EXACT_TAIL_PROMOTED_TOOL_SPECS_TOTAL_TOKEN_LIMIT
    {
        return ExactTailPromotionEligibility::ExceedsSpecBudget;
    }

    ExactTailPromotionEligibility::Eligible { spec_tokens }
}

fn exact_tail_runtime_is_model_visible(
    turn_context: &TurnContext,
    planned_tools: &PlannedTools,
    runtime_index: usize,
    exposure: ToolExposure,
) -> bool {
    let tool_name = planned_tools.runtimes[runtime_index].tool_name();
    !exact_tail_model_visible_specs_with_runtime_exposure(
        turn_context,
        planned_tools,
        runtime_index,
        exposure,
        &tool_name,
    )
    .is_empty()
}

fn exact_tail_model_visible_specs_with_runtime_exposure(
    turn_context: &TurnContext,
    planned_tools: &PlannedTools,
    runtime_index: usize,
    runtime_exposure: ToolExposure,
    tool_name: &ToolName,
) -> Vec<ToolSpec> {
    exact_tail_model_visible_specs_with_runtime_exposures(
        turn_context,
        planned_tools,
        &[(runtime_index, runtime_exposure)],
        std::slice::from_ref(tool_name),
    )
}

fn exact_tail_model_visible_specs_with_runtime_exposures(
    turn_context: &TurnContext,
    planned_tools: &PlannedTools,
    runtime_overrides: &[(usize, ToolExposure)],
    tool_names: &[ToolName],
) -> Vec<ToolSpec> {
    let runtimes = planned_tools
        .runtimes
        .iter()
        .enumerate()
        .map(|(index, runtime)| {
            runtime_overrides
                .iter()
                .find_map(|(runtime_index, runtime_exposure)| {
                    (index == *runtime_index)
                        .then(|| override_tool_exposure(Arc::clone(runtime), *runtime_exposure))
                })
                .unwrap_or_else(|| Arc::clone(runtime))
        })
        .collect::<Vec<_>>();
    let mut final_runtimes = runtimes.clone();
    let affects_code_mode_execute = tool_names.iter().any(|tool_name| {
        runtime_overrides
            .iter()
            .any(|(runtime_index, runtime_exposure)| {
                planned_tools.runtimes[*runtime_index].tool_name() == *tool_name
                    && exact_tail_tool_affects_code_mode_executor(
                        turn_context,
                        tool_name,
                        *runtime_exposure,
                    )
            })
    });
    final_runtimes.splice(0..0, build_code_mode_executors(turn_context, &runtimes));
    model_visible_specs_for_runtimes(
        turn_context,
        &final_runtimes,
        planned_tools.hosted_specs.iter().cloned(),
    )
    .into_iter()
    .filter(|spec| {
        tool_names
            .iter()
            .any(|tool_name| exact_tail_spec_contains_tool(spec, tool_name))
            || (affects_code_mode_execute && spec.name() == codex_code_mode::PUBLIC_TOOL_NAME)
    })
    .collect()
}

fn exact_tail_tool_affects_code_mode_executor(
    turn_context: &TurnContext,
    tool_name: &ToolName,
    exposure: ToolExposure,
) -> bool {
    matches!(
        effective_tool_mode(turn_context),
        ToolMode::CodeMode | ToolMode::CodeModeOnly
    ) && !matches!(
        exposure,
        ToolExposure::DirectModelOnly | ToolExposure::Hidden
    ) && !is_hidden_by_code_mode_only(turn_context, tool_name, exposure)
        && !is_excluded_from_code_mode(turn_context, tool_name)
}

fn model_visible_specs_for_runtimes(
    turn_context: &TurnContext,
    runtimes: &[Arc<dyn CoreToolRuntime>],
    hosted_specs: impl IntoIterator<Item = ToolSpec>,
) -> Vec<ToolSpec> {
    let mut specs = Vec::new();
    let mut seen_tool_names = HashSet::new();
    for runtime in runtimes {
        let tool_name = runtime.tool_name();
        if !seen_tool_names.insert(tool_name.clone()) {
            continue;
        }

        let exposure = runtime.exposure();
        if exposure.is_direct() && !is_hidden_by_code_mode_only(turn_context, &tool_name, exposure)
        {
            specs.push(spec_for_model_request(
                turn_context,
                exposure,
                &tool_name,
                runtime.spec(),
            ));
        }
    }
    specs.extend(hosted_specs);

    merge_into_namespaces(specs)
        .into_iter()
        .filter(|spec| model_visible_spec_enabled(turn_context, spec))
        .collect()
}

fn exact_tail_spec_contains_tool(spec: &ToolSpec, tool_name: &ToolName) -> bool {
    match (spec, &tool_name.namespace) {
        (ToolSpec::Function(tool), None) => tool.name == tool_name.name,
        (ToolSpec::Freeform(tool), None) => tool.name == tool_name.name,
        (ToolSpec::Namespace(namespace), Some(expected_namespace))
            if namespace.name == *expected_namespace =>
        {
            namespace
                .tools
                .iter()
                .any(|namespace_tool| match namespace_tool {
                    ResponsesApiNamespaceTool::Function(tool) => tool.name == tool_name.name,
                })
        }
        (ToolSpec::ToolSearch { .. } | ToolSpec::WebSearch { .. }, None) => {
            spec.name() == tool_name.name
        }
        (
            ToolSpec::Function(_)
            | ToolSpec::Freeform(_)
            | ToolSpec::Namespace(_)
            | ToolSpec::ToolSearch { .. }
            | ToolSpec::WebSearch { .. },
            _,
        ) => false,
    }
}

fn serialized_tool_spec_token_count(spec: &ToolSpec) -> usize {
    let Ok(serialized) = serde_json::to_string(spec) else {
        return usize::MAX;
    };
    approx_token_count(&serialized)
}

fn apply_direct_model_only_namespace_overrides(
    turn_context: &TurnContext,
    planned_tools: &mut PlannedTools,
) {
    for runtime in &mut planned_tools.runtimes {
        let configured = runtime
            .tool_name()
            .namespace
            .as_ref()
            .is_some_and(|namespace| {
                turn_context
                    .config
                    .code_mode
                    .direct_only_tool_namespaces
                    .contains(namespace)
            });
        match runtime.exposure() {
            ToolExposure::Direct | ToolExposure::Deferred if configured => {
                *runtime =
                    override_tool_exposure(Arc::clone(runtime), ToolExposure::DirectModelOnly);
            }
            ToolExposure::Direct
            | ToolExposure::Deferred
            | ToolExposure::DirectModelOnly
            | ToolExposure::Hidden => {}
        }
    }
}

#[instrument(level = "trace", skip_all)]
fn build_model_visible_specs_and_registry(
    turn_context: &TurnContext,
    planned_tools: PlannedTools,
) -> (Vec<ToolSpec>, ToolRegistry) {
    let PlannedTools {
        runtimes,
        hosted_specs,
    } = planned_tools;
    let mut specs = Vec::new();
    let mut seen_tool_names = HashSet::new();
    for runtime in &runtimes {
        let tool_name = runtime.tool_name();
        if !seen_tool_names.insert(tool_name.clone()) {
            continue;
        }
        let exposure = runtime.exposure();
        if exposure.is_direct() && !is_hidden_by_code_mode_only(turn_context, &tool_name, exposure)
        {
            let spec = runtime.spec();
            specs.push(spec_for_model_request(
                turn_context,
                exposure,
                &tool_name,
                spec,
            ));
        }
    }
    specs.extend(hosted_specs);

    let registry = ToolRegistry::from_tools(runtimes);
    let model_visible_specs = merge_into_namespaces(specs)
        .into_iter()
        .filter(|spec| model_visible_spec_enabled(turn_context, spec))
        .collect();

    (model_visible_specs, registry)
}

fn model_visible_spec_enabled(turn_context: &TurnContext, spec: &ToolSpec) -> bool {
    namespace_tools_enabled(turn_context) || !matches!(spec, ToolSpec::Namespace(_))
}

fn spec_for_model_request(
    turn_context: &TurnContext,
    exposure: ToolExposure,
    tool_name: &ToolName,
    spec: ToolSpec,
) -> ToolSpec {
    let tool_mode = effective_tool_mode(turn_context);
    if matches!(tool_mode, ToolMode::CodeMode | ToolMode::CodeModeOnly)
        && exposure != ToolExposure::DirectModelOnly
        && !is_excluded_from_code_mode(turn_context, tool_name)
        && codex_code_mode::is_code_mode_nested_tool(spec.name())
    {
        codex_tools::augment_tool_spec_for_code_mode(spec)
    } else {
        spec
    }
}

#[instrument(level = "trace", skip_all)]
fn hosted_model_tool_specs(context: &CoreToolPlanContext<'_>) -> Vec<ToolSpec> {
    let turn_context = context.step_context.turn.as_ref();
    // Responses Lite accepts schemas for client-executed tools, not hosted Responses tools.
    if turn_context.model_info.use_responses_lite {
        return Vec::new();
    }

    let mut specs = Vec::new();
    let standalone_web_search_available = standalone_web_search_enabled(turn_context)
        && context
            .extension_tool_executors
            .iter()
            .any(|executor| executor.tool_name() == ToolName::namespaced("web", "run"));
    // `Some(Cached/Live/Disabled)` are the options for mode when standalone search is unavailable
    // and the provider supports hosted search. `None` prevents emitting a hosted search tool.
    let web_search_mode = (!standalone_web_search_available
        && turn_context.provider.capabilities().web_search)
        .then_some(turn_context.config.web_search_mode.value());
    let web_search_config = web_search_mode
        .as_ref()
        .and(turn_context.config.web_search_config.as_ref());
    if let Some(hosted_web_search_tool) = create_web_search_tool(WebSearchToolOptions {
        web_search_mode,
        web_search_config,
        web_search_tool_type: turn_context.model_info.web_search_tool_type,
    }) {
        specs.push(hosted_web_search_tool);
    }
    specs
}

pub(crate) fn search_tool_enabled(turn_context: &TurnContext) -> bool {
    turn_context.model_info.supports_search_tool && namespace_tools_enabled(turn_context)
}

pub(crate) fn tool_suggest_enabled(turn_context: &TurnContext) -> bool {
    let features = turn_context.config.features.get();
    features.enabled(Feature::ToolSuggest)
        && features.enabled(Feature::Apps)
        && features.enabled(Feature::Plugins)
}

fn namespace_tools_enabled(turn_context: &TurnContext) -> bool {
    turn_context.provider.capabilities().namespace_tools
}

fn multi_agent_v2_enabled(turn_context: &TurnContext) -> bool {
    turn_context.multi_agent_version == MultiAgentVersion::V2
}

fn collab_tools_enabled(turn_context: &TurnContext) -> bool {
    match turn_context.multi_agent_version {
        MultiAgentVersion::Disabled => false,
        MultiAgentVersion::V1 => !exceeds_thread_spawn_depth_limit(
            next_thread_spawn_depth(&turn_context.session_source),
            turn_context.config.agent_max_depth,
        ),
        MultiAgentVersion::V2 => true,
    }
}

fn agent_jobs_tools_enabled(turn_context: &TurnContext) -> bool {
    turn_context
        .config
        .features
        .get()
        .enabled(Feature::SpawnCsv)
        && collab_tools_enabled(turn_context)
}

fn agent_jobs_worker_tools_enabled(turn_context: &TurnContext) -> bool {
    agent_jobs_tools_enabled(turn_context)
        && matches!(
            &turn_context.session_source,
            SessionSource::SubAgent(SubAgentSource::Other(label))
                if label.starts_with("agent_job:")
        )
}

fn image_generation_runtime_enabled(turn_context: &TurnContext) -> bool {
    (turn_context
        .provider
        .info()
        .uses_openai_actor_authorization()
        || (turn_context.provider.info().requires_openai_auth
            && turn_context
                .auth_manager
                .as_deref()
                .is_some_and(AuthManager::current_auth_uses_codex_backend)))
        && turn_context.provider.capabilities().image_generation
        && turn_context
            .model_info
            .input_modalities
            .contains(&InputModality::Image)
}

fn standalone_image_generation_model_visible(turn_context: &TurnContext) -> bool {
    if !image_generation_runtime_enabled(turn_context) || !namespace_tools_enabled(turn_context) {
        return false;
    }

    turn_context
        .config
        .features
        .get()
        .enabled(Feature::ImageGeneration)
}

fn wait_agent_timeout_options(turn_context: &TurnContext) -> WaitAgentTimeoutOptions {
    if multi_agent_v2_enabled(turn_context) {
        return WaitAgentTimeoutOptions {
            default_timeout_ms: turn_context.config.multi_agent_v2.default_wait_timeout_ms,
            min_timeout_ms: turn_context.config.multi_agent_v2.min_wait_timeout_ms,
            max_timeout_ms: turn_context.config.multi_agent_v2.max_wait_timeout_ms,
        };
    }

    WaitAgentTimeoutOptions {
        default_timeout_ms: DEFAULT_WAIT_TIMEOUT_MS,
        min_timeout_ms: MIN_WAIT_TIMEOUT_MS,
        max_timeout_ms: MAX_WAIT_TIMEOUT_MS,
    }
}

fn agent_type_description(
    turn_context: &TurnContext,
    default_agent_type_description: &str,
) -> String {
    let agent_type_description =
        crate::agent::role::spawn_tool_spec::build(&turn_context.config.agent_roles);
    if agent_type_description.is_empty() {
        default_agent_type_description.to_string()
    } else {
        agent_type_description
    }
}

fn is_hidden_by_code_mode_only(
    turn_context: &TurnContext,
    tool_name: &ToolName,
    exposure: ToolExposure,
) -> bool {
    let tool_mode = effective_tool_mode(turn_context);
    tool_mode == ToolMode::CodeModeOnly
        && exposure != ToolExposure::DirectModelOnly
        && codex_code_mode::is_code_mode_nested_tool(&codex_tools::code_mode_name_for_tool_name(
            tool_name,
        ))
}

fn is_excluded_from_code_mode(turn_context: &TurnContext, tool_name: &ToolName) -> bool {
    tool_name.namespace.as_ref().is_some_and(|namespace| {
        turn_context
            .config
            .code_mode
            .excluded_tool_namespaces
            .contains(namespace)
    })
}

fn build_code_mode_executors(
    turn_context: &TurnContext,
    executors: &[Arc<dyn CoreToolRuntime>],
) -> Vec<Arc<dyn CoreToolRuntime>> {
    let tool_mode = effective_tool_mode(turn_context);
    if !matches!(tool_mode, ToolMode::CodeMode | ToolMode::CodeModeOnly) {
        return vec![];
    }

    let mut code_mode_nested_tool_specs = Vec::new();
    let mut exec_prompt_tool_specs = Vec::new();
    let mut deferred_tools_available = false;
    let deferred_tools_guidance_enabled = search_tool_enabled(turn_context);
    for executor in executors {
        let exposure = executor.exposure();
        if exposure == ToolExposure::DirectModelOnly {
            continue;
        }

        if exposure == ToolExposure::Hidden {
            continue;
        }

        if is_excluded_from_code_mode(turn_context, &executor.tool_name()) {
            continue;
        }

        let spec = executor.spec();

        if exposure == ToolExposure::Deferred {
            // Only show deferred-tool guidance when supported and an included spec is usable by code mode.
            deferred_tools_available |= deferred_tools_guidance_enabled
                && !collect_code_mode_exec_prompt_tool_definitions(std::iter::once(&spec))
                    .is_empty();
        } else {
            exec_prompt_tool_specs.push(spec.clone());
        }
        code_mode_nested_tool_specs.push(spec);
    }

    let namespace_descriptions = code_mode_namespace_descriptions(&exec_prompt_tool_specs);
    let mut enabled_tools =
        collect_code_mode_exec_prompt_tool_definitions(exec_prompt_tool_specs.iter());
    enabled_tools
        .sort_by(|left, right| compare_code_mode_tools(left, right, &namespace_descriptions));

    vec![
        Arc::new(CodeModeExecuteHandler::new(
            create_code_mode_tool(
                &enabled_tools,
                &namespace_descriptions,
                tool_mode == ToolMode::CodeModeOnly,
                deferred_tools_available,
            ),
            code_mode_nested_tool_specs,
        )),
        Arc::new(CodeModeWaitHandler),
    ]
}

#[instrument(level = "trace", skip_all, fields(tool_spec_count = specs.len()))]
fn merge_into_namespaces(specs: Vec<ToolSpec>) -> Vec<ToolSpec> {
    let mut merged_specs = Vec::with_capacity(specs.len());
    let mut namespace_indices = BTreeMap::<String, usize>::new();
    for spec in specs {
        match spec {
            ToolSpec::Namespace(mut namespace) => {
                if let Some(index) = namespace_indices.get(&namespace.name).copied() {
                    let ToolSpec::Namespace(existing_namespace) = &mut merged_specs[index] else {
                        unreachable!("namespace index must point to a namespace spec");
                    };
                    if existing_namespace.description.trim().is_empty()
                        && !namespace.description.trim().is_empty()
                    {
                        existing_namespace.description = namespace.description;
                    }
                    existing_namespace.tools.append(&mut namespace.tools);
                    continue;
                }

                namespace_indices.insert(namespace.name.clone(), merged_specs.len());
                merged_specs.push(ToolSpec::Namespace(namespace));
            }
            spec => merged_specs.push(spec),
        }
    }

    for spec in &mut merged_specs {
        let ToolSpec::Namespace(namespace) = spec else {
            continue;
        };

        namespace.tools.sort_by(|left, right| match (left, right) {
            (
                ResponsesApiNamespaceTool::Function(left),
                ResponsesApiNamespaceTool::Function(right),
            ) => left.name.cmp(&right.name),
        });

        if namespace.description.trim().is_empty() {
            namespace.description = default_namespace_description(&namespace.name);
        }
    }

    merged_specs
}

fn code_mode_namespace_descriptions(
    specs: &[ToolSpec],
) -> BTreeMap<String, codex_code_mode::ToolNamespaceDescription> {
    let mut namespace_descriptions = BTreeMap::new();
    for spec in specs {
        let ToolSpec::Namespace(namespace) = spec else {
            continue;
        };

        let entry = namespace_descriptions
            .entry(namespace.name.clone())
            .or_insert_with(|| codex_code_mode::ToolNamespaceDescription {
                name: namespace.name.clone(),
                description: namespace.description.clone(),
            });
        if entry.description.trim().is_empty() && !namespace.description.trim().is_empty() {
            entry.description = namespace.description.clone();
        }
    }
    namespace_descriptions
}

#[instrument(level = "trace", skip_all)]
fn add_tool_sources(context: &CoreToolPlanContext<'_>, planned_tools: &mut PlannedTools) {
    if crate::guardian::is_guardian_reviewer_source(&context.step_context.turn.session_source) {
        let turn_context = context.step_context.turn.as_ref();
        let environment_mode = tool_environment_mode(context.step_context);
        if environment_mode.has_environment() {
            let include_environment_id = matches!(environment_mode, ToolEnvironmentMode::Multiple);
            planned_tools.add(ExecCommandHandler::new(ExecCommandHandlerOptions {
                allow_login_shell: turn_context.config.permissions.allow_login_shell,
                exec_permission_approvals_enabled: false,
                include_environment_id,
                include_shell_parameter: unified_exec_should_include_shell_parameter(
                    turn_context,
                    context.step_context,
                ),
            }));
            planned_tools.add(WriteStdinHandler);
            planned_tools.add(ViewImageHandler::new(ViewImageToolOptions {
                can_request_original_image_detail: can_request_original_image_detail(
                    &turn_context.model_info,
                ),
                include_environment_id,
            }));
        }
        return;
    }

    add_shell_tools(context, planned_tools);
    add_mcp_resource_tools(context, planned_tools);
    add_core_utility_tools(context, planned_tools);
    add_collaboration_tools(context, planned_tools);
    add_mcp_runtime_tools(context, planned_tools);
    add_extension_tools(context, planned_tools);
    add_dynamic_tools(context, planned_tools);
    for spec in hosted_model_tool_specs(context) {
        planned_tools.add_hosted_spec(spec);
    }
}

fn standalone_web_search_enabled(turn_context: &TurnContext) -> bool {
    namespace_tools_enabled(turn_context)
        && (turn_context.model_info.use_responses_lite
            || turn_context
                .config
                .features
                .get()
                .enabled(Feature::StandaloneWebSearch))
}

fn tool_environment_mode(step_context: &StepContext) -> ToolEnvironmentMode {
    ToolEnvironmentMode::from_count(step_context.environments.turn_environments.len())
}

#[instrument(level = "trace", skip_all)]
fn add_shell_tools(context: &CoreToolPlanContext<'_>, planned_tools: &mut PlannedTools) {
    let turn_context = context.step_context.turn.as_ref();
    let features = turn_context.config.features.get();
    let environment_mode = tool_environment_mode(context.step_context);
    if !environment_mode.has_environment() {
        return;
    }

    let allow_login_shell = turn_context.config.permissions.allow_login_shell;
    let exec_permission_approvals_enabled = features.enabled(Feature::ExecPermissionApprovals);
    let include_environment_id = matches!(environment_mode, ToolEnvironmentMode::Multiple);
    let shell_command_options = ShellCommandHandlerOptions {
        backend_config: shell_command_backend_for_features(features),
        allow_login_shell,
        exec_permission_approvals_enabled,
    };

    match shell_type_for_model_and_features(&turn_context.model_info, features) {
        ConfigShellToolType::UnifiedExec => {
            planned_tools.add(ExecCommandHandler::new(ExecCommandHandlerOptions {
                allow_login_shell,
                exec_permission_approvals_enabled,
                include_environment_id,
                include_shell_parameter: unified_exec_should_include_shell_parameter(
                    turn_context,
                    context.step_context,
                ),
            }));
            planned_tools.add(WriteStdinHandler);

            // Keep the legacy shell tool registered while unified exec is
            // model-visible.
            planned_tools.add_dispatch_only(ShellCommandHandler::new(shell_command_options));
        }
        ConfigShellToolType::Disabled => {}
        ConfigShellToolType::Default
        | ConfigShellToolType::Local
        | ConfigShellToolType::ShellCommand => {
            planned_tools.add(ShellCommandHandler::new(shell_command_options));
        }
    }
}

fn unified_exec_should_include_shell_parameter(
    turn_context: &TurnContext,
    step_context: &StepContext,
) -> bool {
    !matches!(
        &turn_context.unified_exec_shell_mode,
        UnifiedExecShellMode::ZshFork(_)
    ) || step_context
        .environments
        .turn_environments
        .iter()
        .any(|environment| environment.environment.is_remote())
}

#[instrument(level = "trace", skip_all)]
fn add_mcp_resource_tools(context: &CoreToolPlanContext<'_>, planned_tools: &mut PlannedTools) {
    if context.mcp_tools.is_some() {
        planned_tools.add(ListMcpResourcesHandler);
        planned_tools.add(ListMcpResourceTemplatesHandler);
        planned_tools.add(ReadMcpResourceHandler);
    }
}

#[instrument(level = "trace", skip_all)]
fn add_core_utility_tools(context: &CoreToolPlanContext<'_>, planned_tools: &mut PlannedTools) {
    let turn_context = context.step_context.turn.as_ref();
    let features = turn_context.config.features.get();
    let environment_mode = tool_environment_mode(context.step_context);

    planned_tools.add(PlanHandler);

    if features.enabled(Feature::DeferredExecutor) {
        planned_tools.add(WaitForEnvironmentHandler);
    }

    if turn_context.config.experimental_request_user_input_enabled {
        planned_tools.add_with_exposure(
            RequestUserInputHandler {
                available_modes: request_user_input_available_modes(features),
            },
            ToolExposure::DirectModelOnly,
        );
    }

    if environment_mode.has_environment() && features.enabled(Feature::RequestPermissionsTool) {
        planned_tools.add(RequestPermissionsHandler);
    }

    if features.enabled(Feature::TokenBudget) {
        planned_tools.add_with_exposure(NewContextWindowHandler, ToolExposure::DirectModelOnly);
        planned_tools.add(GetContextRemainingHandler);
    }

    if features.enabled(Feature::CurrentTimeReminder) {
        planned_tools.add(CurrentTimeHandler);
        if turn_context
            .config
            .current_time_reminder
            .as_ref()
            .is_some_and(|config| config.sleep_tool)
        {
            planned_tools.add(SleepHandler);
        }
    }

    if tool_suggest_enabled(turn_context)
        && let Some(candidates) = context
            .tool_suggest_candidates
            .filter(|candidates| !candidates.tools.is_empty())
    {
        if candidates.presentation == crate::tools::router::ToolSuggestPresentation::ListTool {
            planned_tools.add(ListAvailablePluginsToInstallHandler::new(
                collect_request_plugin_install_entries(&candidates.tools),
            ));
        }
        planned_tools.add(RequestPluginInstallHandler::new(
            candidates.tools.clone(),
            candidates.presentation,
        ));
    }

    if environment_mode.has_environment() && turn_context.model_info.apply_patch_tool_type.is_some()
    {
        let include_environment_id = matches!(environment_mode, ToolEnvironmentMode::Multiple);
        planned_tools.add(ApplyPatchHandler::new(include_environment_id));
    }

    if turn_context
        .model_info
        .experimental_supported_tools
        .iter()
        .any(|tool| tool == "test_sync_tool")
    {
        planned_tools.add(TestSyncHandler);
    }

    if environment_mode.has_environment() {
        let include_environment_id = matches!(environment_mode, ToolEnvironmentMode::Multiple);
        planned_tools.add(ViewImageHandler::new(ViewImageToolOptions {
            can_request_original_image_detail: can_request_original_image_detail(
                &turn_context.model_info,
            ),
            include_environment_id,
        }));
    }
}

#[instrument(level = "trace", skip_all)]
fn add_collaboration_tools(context: &CoreToolPlanContext<'_>, planned_tools: &mut PlannedTools) {
    let turn_context = context.step_context.turn.as_ref();
    if collab_tools_enabled(turn_context) {
        if multi_agent_v2_enabled(turn_context) {
            let exposure = if turn_context.config.multi_agent_v2.non_code_mode_only {
                ToolExposure::DirectModelOnly
            } else {
                ToolExposure::Direct
            };
            let tool_namespace = namespace_tools_enabled(turn_context)
                .then_some(turn_context.config.multi_agent_v2.tool_namespace.as_deref())
                .flatten();
            let agent_type_description =
                agent_type_description(turn_context, context.default_agent_type_description);
            planned_tools.add_arc(override_tool_exposure(
                multi_agent_v2_handler(
                    SpawnAgentHandlerV2::new(SpawnAgentToolOptions {
                        available_models: turn_context.available_models.clone(),
                        agent_type_description,
                        hide_agent_type_model_reasoning: turn_context
                            .config
                            .multi_agent_v2
                            .hide_spawn_agent_metadata,
                        usage_hint_text: turn_context.config.multi_agent_v2.usage_hint_text.clone(),
                    }),
                    tool_namespace,
                ),
                exposure,
            ));
            planned_tools.add_arc(override_tool_exposure(
                multi_agent_v2_handler(SendMessageHandlerV2, tool_namespace),
                exposure,
            ));
            planned_tools.add_arc(override_tool_exposure(
                multi_agent_v2_handler(FollowupTaskHandlerV2, tool_namespace),
                exposure,
            ));
            planned_tools.add_arc(override_tool_exposure(
                multi_agent_v2_handler(
                    WaitAgentHandlerV2::new(context.wait_agent_timeouts),
                    tool_namespace,
                ),
                exposure,
            ));
            planned_tools.add_arc(override_tool_exposure(
                multi_agent_v2_handler(InterruptAgentHandler, tool_namespace),
                exposure,
            ));
            planned_tools.add_arc(override_tool_exposure(
                multi_agent_v2_handler(ListAgentsHandlerV2, tool_namespace),
                exposure,
            ));
        } else {
            let agent_type_description =
                agent_type_description(turn_context, context.default_agent_type_description);
            let exposure = if search_tool_enabled(turn_context) {
                ToolExposure::Deferred
            } else {
                ToolExposure::Direct
            };
            planned_tools.add_with_exposure(
                SpawnAgentHandler::new(SpawnAgentToolOptions {
                    available_models: turn_context.available_models.clone(),
                    agent_type_description,
                    hide_agent_type_model_reasoning: false,
                    usage_hint_text: turn_context.config.multi_agent_v2.usage_hint_text.clone(),
                }),
                exposure,
            );
            planned_tools.add_with_exposure(SendInputHandler, exposure);
            planned_tools.add_with_exposure(ResumeAgentHandler, exposure);
            planned_tools
                .add_with_exposure(WaitAgentHandler::new(context.wait_agent_timeouts), exposure);
            planned_tools.add_with_exposure(CloseAgentHandler, exposure);
        }
    }

    if agent_jobs_tools_enabled(turn_context) {
        planned_tools.add(SpawnAgentsOnCsvHandler);
        if agent_jobs_worker_tools_enabled(turn_context) {
            planned_tools.add(ReportAgentJobResultHandler);
        }
    }
}

#[instrument(
    level = "trace",
    skip_all,
    fields(
        direct_mcp_tool_count = context.mcp_tools.map_or(0, <[ToolInfo]>::len),
        deferred_mcp_tool_count = context.deferred_mcp_tools.map_or(0, <[ToolInfo]>::len)
    )
)]
fn add_mcp_runtime_tools(context: &CoreToolPlanContext<'_>, planned_tools: &mut PlannedTools) {
    if let Some(mcp_tools) = context.mcp_tools {
        for tool in mcp_tools {
            match McpHandler::new(tool.clone()) {
                Ok(handler) => planned_tools.add(handler),
                Err(err) => warn!(
                    "Skipping MCP tool `{}`: failed to build tool spec: {err}",
                    tool.canonical_tool_name()
                ),
            }
        }
    }

    if let Some(deferred_mcp_tools) = context.deferred_mcp_tools {
        for tool in deferred_mcp_tools {
            match McpHandler::new(tool.clone()) {
                Ok(handler) => planned_tools.add_with_exposure(handler, ToolExposure::Deferred),
                Err(err) => warn!(
                    "Skipping deferred MCP tool `{}`: failed to build tool spec: {err}",
                    tool.canonical_tool_name()
                ),
            }
        }
    }
}

#[instrument(
    level = "trace",
    skip_all,
    fields(dynamic_tool_count = context.dynamic_tools.len())
)]
fn add_dynamic_tools(context: &CoreToolPlanContext<'_>, planned_tools: &mut PlannedTools) {
    for spec in context.dynamic_tools {
        match spec {
            DynamicToolSpec::Function(tool) => {
                let Some(handler) = DynamicToolHandler::new(tool) else {
                    tracing::error!(
                        "Failed to convert dynamic tool {:?} to OpenAI tool",
                        tool.name
                    );
                    continue;
                };
                planned_tools.add(handler);
            }
            DynamicToolSpec::Namespace(namespace) => {
                for tool in &namespace.tools {
                    let DynamicToolNamespaceTool::Function(tool) = tool;
                    let Some(handler) = DynamicToolHandler::new_in_namespace(namespace, tool)
                    else {
                        tracing::error!(
                            "Failed to convert dynamic tool {:?}.{:?} to OpenAI tool",
                            namespace.name,
                            tool.name
                        );
                        continue;
                    };
                    planned_tools.add(handler);
                }
            }
        }
    }
}

#[instrument(
    level = "trace",
    skip_all,
    fields(extension_tool_executor_count = context.extension_tool_executors.len())
)]
fn add_extension_tools(context: &CoreToolPlanContext<'_>, planned_tools: &mut PlannedTools) {
    // Extension ToolContributor implementations are resolved into executors
    // before planning. Core only adapts those executors into its runtime set.
    append_extension_tool_executors(
        context.step_context.turn.as_ref(),
        context.extension_tool_executors,
        planned_tools,
    );
}

#[instrument(level = "trace", skip_all)]
fn append_tool_search_executor(
    context: &CoreToolPlanContext<'_>,
    planned_tools: &mut PlannedTools,
) {
    let turn_context = context.step_context.turn.as_ref();
    if !search_tool_enabled(turn_context) {
        return;
    }

    let search_infos = planned_tools
        .runtimes()
        .iter()
        .filter(|executor| executor.exposure() == ToolExposure::Deferred)
        .filter_map(|executor| executor.search_info())
        .collect::<Vec<_>>();
    if search_infos.is_empty() {
        return;
    }

    let handler: PlannedRuntime = context.tool_search_handler_cache.get_or_build(search_infos);
    planned_tools.add_arc(handler);
}

fn prepend_code_mode_executors(
    context: &CoreToolPlanContext<'_>,
    planned_tools: &mut PlannedTools,
) {
    let turn_context = context.step_context.turn.as_ref();
    let code_mode_executors = build_code_mode_executors(turn_context, planned_tools.runtimes());
    planned_tools.runtimes.splice(0..0, code_mode_executors);
}

fn append_extension_tool_executors(
    turn_context: &TurnContext,
    executors: &[Arc<dyn ToolExecutor<ExtensionToolCall>>],
    planned_tools: &mut PlannedTools,
) {
    if executors.is_empty() {
        return;
    }

    let mut reserved_tool_names = planned_tools
        .runtimes()
        .iter()
        .map(|executor| executor.tool_name())
        .collect::<HashSet<_>>();
    let tool_mode = effective_tool_mode(turn_context);
    if matches!(tool_mode, ToolMode::CodeMode | ToolMode::CodeModeOnly) {
        reserved_tool_names.insert(ToolName::plain(codex_code_mode::PUBLIC_TOOL_NAME));
        reserved_tool_names.insert(ToolName::plain(codex_code_mode::WAIT_TOOL_NAME));
    }
    if search_tool_enabled(turn_context)
        && planned_tools
            .runtimes()
            .iter()
            .any(|executor| executor.exposure() == ToolExposure::Deferred)
    {
        reserved_tool_names.insert(ToolName::plain(TOOL_SEARCH_TOOL_NAME));
    }

    let standalone_web_search_enabled = standalone_web_search_enabled(turn_context);
    let web_search_mode_on = turn_context.config.web_search_mode.value() != WebSearchMode::Disabled;

    for executor in executors.iter().cloned() {
        let tool_name = executor.tool_name();
        if tool_name == ToolName::namespaced("web", "run")
            && (!standalone_web_search_enabled || !web_search_mode_on)
        {
            continue;
        }
        if tool_name == ToolName::namespaced(IMAGE_GEN_NAMESPACE, IMAGEGEN_TOOL_NAME)
            && !standalone_image_generation_model_visible(turn_context)
        {
            continue;
        }
        if !reserved_tool_names.insert(tool_name.clone()) {
            warn!("Skipping extension tool `{tool_name}`: tool already registered");
            continue;
        }
        planned_tools.add(ExtensionToolAdapter::new(executor));
    }
}

fn multi_agent_v2_handler(
    handler: impl CoreToolRuntime + 'static,
    namespace: Option<&str>,
) -> Arc<dyn CoreToolRuntime> {
    match namespace {
        Some(namespace) => Arc::new(MultiAgentV2NamespaceOverride {
            handler: Arc::new(handler),
            namespace: namespace.to_string(),
        }),
        None => Arc::new(handler),
    }
}

struct MultiAgentV2NamespaceOverride {
    handler: Arc<dyn CoreToolRuntime>,
    namespace: String,
}

impl ToolExecutor<ToolInvocation> for MultiAgentV2NamespaceOverride {
    fn tool_name(&self) -> ToolName {
        ToolName::namespaced(self.namespace.clone(), self.handler.tool_name().name)
    }

    fn spec(&self) -> ToolSpec {
        match self.handler.spec() {
            ToolSpec::Function(tool) => ToolSpec::Namespace(ResponsesApiNamespace {
                name: self.namespace.clone(),
                description: MULTI_AGENT_V2_NAMESPACE_DESCRIPTION.to_string(),
                tools: vec![ResponsesApiNamespaceTool::Function(tool)],
            }),
            spec => spec,
        }
    }

    fn exposure(&self) -> ToolExposure {
        self.handler.exposure()
    }

    fn supports_parallel_tool_calls(&self) -> bool {
        self.handler.supports_parallel_tool_calls()
    }

    fn search_info(&self) -> Option<ToolSearchInfo> {
        self.handler.search_info()
    }

    fn handle(&self, invocation: ToolInvocation) -> codex_tools::ToolExecutorFuture<'_> {
        self.handler.handle(invocation)
    }
}

impl CoreToolRuntime for MultiAgentV2NamespaceOverride {
    fn matches_kind(&self, payload: &crate::tools::context::ToolPayload) -> bool {
        self.handler.matches_kind(payload)
    }

    fn create_diff_consumer(
        &self,
    ) -> Option<Box<dyn crate::tools::registry::ToolArgumentDiffConsumer>> {
        self.handler.create_diff_consumer()
    }
}

fn compare_code_mode_tools(
    left: &codex_code_mode::ToolDefinition,
    right: &codex_code_mode::ToolDefinition,
    namespace_descriptions: &BTreeMap<String, codex_code_mode::ToolNamespaceDescription>,
) -> std::cmp::Ordering {
    let left_namespace = code_mode_namespace_name(left, namespace_descriptions);
    let right_namespace = code_mode_namespace_name(right, namespace_descriptions);

    left_namespace
        .cmp(&right_namespace)
        .then_with(|| left.tool_name.name.cmp(&right.tool_name.name))
        .then_with(|| left.name.cmp(&right.name))
}

fn code_mode_namespace_name<'a>(
    tool: &codex_code_mode::ToolDefinition,
    namespace_descriptions: &'a BTreeMap<String, codex_code_mode::ToolNamespaceDescription>,
) -> Option<&'a str> {
    tool.tool_name
        .namespace
        .as_ref()
        .and_then(|namespace| namespace_descriptions.get(namespace))
        .map(|namespace_description| namespace_description.name.as_str())
}

#[cfg(test)]
#[path = "spec_plan_tests.rs"]
mod tests;
