"""Tool result persistence -- preserves large outputs to disk instead of truncating.

Defense against context-window overflow operates at three levels:

1. **Per-tool output cap** (inside each tool): Tools like search_files
   pre-truncate their own output before returning. This is the first line
   of defense and the only one the tool author controls.

2. **Per-result persistence** (maybe_persist_tool_result): After a tool
   returns, if its output exceeds the tool's registered threshold
   (registry.get_max_result_size), the full output is written to disk and
   the in-context content is replaced with a preview + file path reference.
   The model can read_file to access the full output.

3. **Per-turn aggregate budget** (enforce_turn_budget): After all tool
   results in a single assistant turn are collected, if the total exceeds
   MAX_TURN_BUDGET_CHARS (200K), the largest non-persisted results are
   spilled to disk until the aggregate is under budget. This catches cases
   where many medium-sized results combine to overflow context.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULT_SIZE_CHARS: int = 50_000
MAX_TURN_BUDGET_CHARS: int = 200_000
PREVIEW_SIZE_CHARS: int = 2000
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"


@dataclass
class PersistedResult:
    tool_use_id: str
    original_size: int
    file_path: str
    preview: str
    has_more: bool


def get_storage_dir(session_id: str) -> Path:
    """Return ~/.hermes/sessions/{session_id}/tool-results/, creating if needed."""
    from hermes_constants import get_hermes_home
    d = get_hermes_home() / "sessions" / session_id / "tool-results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def generate_preview(content: str, max_chars: int = PREVIEW_SIZE_CHARS) -> tuple[str, bool]:
    """Truncate at last newline within max_chars. Returns (preview, has_more)."""
    if len(content) <= max_chars:
        return content, False
    # Find last newline within budget
    truncated = content[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars // 2:  # Only use newline boundary if it's past halfway
        truncated = truncated[:last_nl + 1]
    return truncated, True


def persist_large_result(
    content: str,
    tool_use_id: str,
    storage_dir: Path,
    threshold: int = DEFAULT_MAX_RESULT_SIZE_CHARS,
) -> PersistedResult | None:
    """Write full content to disk if it exceeds *threshold*.

    Uses open(path, 'x') for atomic exclusive create — dedup on retry.
    Returns None if content is small enough to keep inline.
    """
    if len(content) <= threshold:
        return None

    file_path = storage_dir / f"{tool_use_id}.txt"
    try:
        with open(file_path, "x", encoding="utf-8") as f:
            f.write(content)
    except FileExistsError:
        pass  # Already persisted (e.g. retry) — fall through to preview

    preview, has_more = generate_preview(content)
    return PersistedResult(
        tool_use_id=tool_use_id,
        original_size=len(content),
        file_path=str(file_path),
        preview=preview,
        has_more=has_more,
    )


def build_persisted_output_message(result: PersistedResult) -> str:
    """Build the <persisted-output> replacement block."""
    size_kb = result.original_size / 1024
    if size_kb >= 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"

    msg = f"{PERSISTED_OUTPUT_TAG}\n"
    msg += f"This tool result was too large ({result.original_size:,} characters, {size_str}).\n"
    msg += f"Full output saved to: {result.file_path}\n"
    msg += "Use read_file to access specific sections of this output if needed.\n\n"
    msg += f"Preview (first {len(result.preview)} chars):\n"
    msg += result.preview
    if result.has_more:
        msg += "\n..."
    msg += f"\n{PERSISTED_OUTPUT_CLOSING_TAG}"
    return msg


def maybe_persist_tool_result(
    content: str,
    tool_name: str,
    tool_use_id: str,
    storage_dir: Path,
) -> str:
    """Per-result persistence entry point (level 2). Check per-tool threshold, persist if needed.

    Returns original content (if small) or the <persisted-output> replacement.
    """
    from tools.registry import registry
    threshold = registry.get_max_result_size(tool_name)

    # Infinity means never persist (e.g. read_file)
    if not isinstance(threshold, (int, float)) or threshold == float('inf'):
        return content

    if len(content) <= threshold:
        return content

    result = persist_large_result(content, tool_use_id, storage_dir,
                                  threshold=threshold)
    if result is None:
        return content

    logger.info(
        "Persisted large tool result: %s (%s, %d chars -> %s)",
        tool_name, tool_use_id, result.original_size, result.file_path,
    )
    return build_persisted_output_message(result)


def enforce_turn_budget(
    tool_messages: list[dict],
    storage_dir: Path,
    budget: int = MAX_TURN_BUDGET_CHARS,
) -> list[dict]:
    """Per-turn aggregate budget entry point (level 3). After all tool results in a turn, enforce aggregate budget.

    If total chars exceed budget, persist the largest results first until under budget.
    Already-persisted results (containing <persisted-output>) are skipped.
    Mutates the list in-place and returns it.
    """
    # Calculate total and identify candidates
    candidates = []  # (index, size) for non-persisted results
    total_size = 0
    for i, msg in enumerate(tool_messages):
        content = msg.get("content", "")
        size = len(content)
        total_size += size
        if PERSISTED_OUTPUT_TAG not in content:
            candidates.append((i, size))

    if total_size <= budget:
        return tool_messages

    # Sort candidates by size descending — persist largest first
    candidates.sort(key=lambda x: x[1], reverse=True)

    for idx, size in candidates:
        if total_size <= budget:
            break
        msg = tool_messages[idx]
        content = msg["content"]
        tool_use_id = msg.get("tool_call_id", f"budget_{idx}")

        result = persist_large_result(content, tool_use_id, storage_dir,
                                          threshold=0)
        if result:
            replacement = build_persisted_output_message(result)
            total_size -= size
            total_size += len(replacement)
            tool_messages[idx]["content"] = replacement
            logger.info(
                "Budget enforcement: persisted tool result %s (%d chars)",
                tool_use_id, size,
            )

    return tool_messages
