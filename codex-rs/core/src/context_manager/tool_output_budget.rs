use codex_utils_output_truncation::TruncationPolicy;

const TOOL_OUTPUT_ITEM_OVERHEAD_MIN_TOKENS: i64 = 128;
const TOOL_OUTPUT_ITEM_OVERHEAD_MAX_TOKENS: i64 = 4_096;
const TOOL_OUTPUT_ITEM_OVERHEAD_RATIO_DIVISOR: i64 = 10;

pub(crate) fn model_visible_tool_output_item_token_limit(body_token_limit: i64) -> i64 {
    if body_token_limit <= 0 {
        return body_token_limit;
    }

    let proportional_overhead = body_token_limit
        .checked_div(TOOL_OUTPUT_ITEM_OVERHEAD_RATIO_DIVISOR)
        .unwrap_or(0);
    let overhead = proportional_overhead.clamp(
        TOOL_OUTPUT_ITEM_OVERHEAD_MIN_TOKENS,
        TOOL_OUTPUT_ITEM_OVERHEAD_MAX_TOKENS,
    );
    body_token_limit.saturating_add(overhead)
}

pub(crate) fn model_visible_tool_output_item_token_limit_for_policy(
    policy: TruncationPolicy,
) -> i64 {
    let body_token_limit = i64::try_from(policy.token_budget()).unwrap_or(i64::MAX);
    model_visible_tool_output_item_token_limit(body_token_limit)
}
