use crate::compact::should_use_remote_compact_task;
use crate::compact::should_use_remote_compact_task_v2_for_config;
use crate::compact_exact_tail::CompactionHistoryPolicy;
use crate::config::Config;
use crate::session::turn_context::TurnContext;
use codex_model_provider_info::ModelProviderInfo;
use codex_protocol::config_types::CompactExactTailStrategy;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompactRoute {
    Local,
    RemoteLegacy,
    RemoteV2,
    SemanticTranscript,
}

impl CompactRoute {
    pub(crate) fn metric_name(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::RemoteLegacy => "remote",
            Self::RemoteV2 => "remote_v2",
            Self::SemanticTranscript => "semantic_transcript",
        }
    }
}

pub(crate) fn compact_route_for_config(
    config: &Config,
    provider: &ModelProviderInfo,
) -> CodexResult<CompactRoute> {
    let remote_available = should_use_remote_compact_task(provider);
    let exact_tail_active = matches!(
        CompactionHistoryPolicy::from_config(config),
        CompactionHistoryPolicy::PreserveRecentExact { .. }
    );

    if !exact_tail_active {
        return Ok(default_compact_route(config, remote_available));
    }

    match config.compact_exact_tail_strategy {
        CompactExactTailStrategy::Auto => Ok(default_compact_route(config, remote_available)),
        CompactExactTailStrategy::RemoteLegacy => {
            explicit_remote_exact_tail_route(remote_available, CompactRoute::RemoteLegacy)
        }
        CompactExactTailStrategy::RemoteV2 => {
            explicit_remote_exact_tail_route(remote_available, CompactRoute::RemoteV2)
        }
        CompactExactTailStrategy::SemanticTranscript => {
            explicit_remote_exact_tail_route(remote_available, CompactRoute::SemanticTranscript)
        }
    }
}

pub(crate) fn compact_route(turn_context: &TurnContext) -> CodexResult<CompactRoute> {
    compact_route_for_config(&turn_context.config, turn_context.provider.info())
}

fn default_compact_route(config: &Config, remote_available: bool) -> CompactRoute {
    if !remote_available {
        return CompactRoute::Local;
    }
    if should_use_remote_compact_task_v2_for_config(config) {
        CompactRoute::RemoteV2
    } else {
        CompactRoute::RemoteLegacy
    }
}

fn explicit_remote_exact_tail_route(
    remote_available: bool,
    route: CompactRoute,
) -> CodexResult<CompactRoute> {
    if remote_available {
        Ok(route)
    } else {
        Err(CodexErr::Stream(
            format!(
                "Exact-tail compaction strategy `{}` requires a provider with remote compaction support.",
                route.metric_name()
            ),
            None,
        ))
    }
}
