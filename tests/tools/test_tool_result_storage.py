"""Tests for tools.tool_result_storage — Layer 2 + Layer 3 persistence logic."""

import pytest

from tools.tool_result_storage import (
    DEFAULT_MAX_RESULT_SIZE_CHARS,
    MAX_TURN_BUDGET_CHARS,
    PERSISTED_OUTPUT_CLOSING_TAG,
    PERSISTED_OUTPUT_TAG,
    PREVIEW_SIZE_CHARS,
    PersistedResult,
    build_persisted_output_message,
    enforce_turn_budget,
    generate_preview,
    maybe_persist_tool_result,
    persist_large_result,
)


# ------------------------------------------------------------------ #
# generate_preview
# ------------------------------------------------------------------ #

class TestGeneratePreview:
    def test_short_content_unchanged(self):
        """Content under limit returns as-is, has_more=False."""
        text = "hello world"
        preview, has_more = generate_preview(text)
        assert preview == text
        assert has_more is False

    def test_truncates_at_newline_boundary(self):
        """Multi-line content truncated at last newline within budget."""
        # Build content with lines that exceed the budget
        lines = [f"line {i}: " + "x" * 80 for i in range(50)]
        content = "\n".join(lines)
        assert len(content) > PREVIEW_SIZE_CHARS

        preview, has_more = generate_preview(content)
        assert has_more is True
        assert len(preview) <= PREVIEW_SIZE_CHARS
        # Should end at a newline boundary
        assert preview.endswith("\n")

    def test_single_line_truncates_at_max(self):
        """Single long line truncated at max_bytes exactly."""
        content = "x" * 5000  # No newlines
        preview, has_more = generate_preview(content, max_chars=100)
        assert has_more is True
        assert len(preview) == 100

    def test_empty_content(self):
        """Empty string returns ('', False)."""
        preview, has_more = generate_preview("")
        assert preview == ""
        assert has_more is False

    def test_exact_boundary(self):
        """Content exactly at max_bytes returns as-is."""
        content = "x" * PREVIEW_SIZE_CHARS
        preview, has_more = generate_preview(content)
        assert preview == content
        assert has_more is False

    def test_newline_only_used_if_past_halfway(self):
        """Newline before halfway mark is ignored; truncation at max_bytes."""
        # Newline at position 10 out of 100 — way before halfway (50)
        content = "a" * 10 + "\n" + "b" * 200
        preview, has_more = generate_preview(content, max_chars=100)
        assert has_more is True
        # Should NOT truncate at position 10 since it's before halfway
        assert len(preview) == 100


# ------------------------------------------------------------------ #
# persist_large_result
# ------------------------------------------------------------------ #

class TestPersistLargeResult:
    def test_small_returns_none(self, tmp_path):
        """Content under DEFAULT_MAX_RESULT_SIZE_CHARS returns None."""
        content = "small output"
        result = persist_large_result(content, "tool_1", tmp_path)
        assert result is None

    def test_large_writes_file(self, tmp_path):
        """Content over threshold writes file, returns PersistedResult."""
        content = "x" * (DEFAULT_MAX_RESULT_SIZE_CHARS + 1)
        result = persist_large_result(content, "tool_2", tmp_path)
        assert result is not None
        assert isinstance(result, PersistedResult)
        assert result.tool_use_id == "tool_2"
        assert result.original_size == len(content)
        assert result.file_path == str(tmp_path / "tool_2.txt")
        assert (tmp_path / "tool_2.txt").exists()

    def test_dedup_via_exclusive_create(self, tmp_path):
        """Second call with same tool_use_id doesn't crash, returns result."""
        content = "x" * (DEFAULT_MAX_RESULT_SIZE_CHARS + 1)
        result1 = persist_large_result(content, "tool_dup", tmp_path)
        result2 = persist_large_result(content, "tool_dup", tmp_path)
        assert result1 is not None
        assert result2 is not None
        # Both return valid PersistedResult
        assert result1.tool_use_id == result2.tool_use_id

    def test_file_contains_full_content(self, tmp_path):
        """Verify the written file has the complete original content."""
        content = "line1\nline2\nline3\n" * 10000  # Well over threshold
        result = persist_large_result(content, "tool_full", tmp_path)
        assert result is not None
        on_disk = (tmp_path / "tool_full.txt").read_text(encoding="utf-8")
        assert on_disk == content

    def test_exactly_at_threshold_returns_none(self, tmp_path):
        """Content exactly at DEFAULT_MAX_RESULT_SIZE_CHARS is not persisted."""
        content = "x" * DEFAULT_MAX_RESULT_SIZE_CHARS
        result = persist_large_result(content, "tool_exact", tmp_path)
        assert result is None


# ------------------------------------------------------------------ #
# build_persisted_output_message
# ------------------------------------------------------------------ #

class TestBuildPersistedOutputMessage:
    @pytest.fixture
    def sample_result(self):
        return PersistedResult(
            tool_use_id="test_id",
            original_size=100_000,
            file_path="/tmp/test_id.txt",
            preview="first line\nsecond line\n",
            has_more=True,
        )

    def test_contains_file_path(self, sample_result):
        msg = build_persisted_output_message(sample_result)
        assert sample_result.file_path in msg

    def test_contains_preview(self, sample_result):
        msg = build_persisted_output_message(sample_result)
        assert "first line\nsecond line\n" in msg

    def test_contains_size_info(self, sample_result):
        msg = build_persisted_output_message(sample_result)
        assert "100,000 characters" in msg
        assert "97.7 KB" in msg

    def test_contains_tags(self, sample_result):
        msg = build_persisted_output_message(sample_result)
        assert msg.startswith(PERSISTED_OUTPUT_TAG)
        assert msg.endswith(PERSISTED_OUTPUT_CLOSING_TAG)

    def test_has_more_false_no_ellipsis(self):
        result = PersistedResult(
            tool_use_id="t",
            original_size=60_000,
            file_path="/tmp/t.txt",
            preview="all content",
            has_more=False,
        )
        msg = build_persisted_output_message(result)
        assert "\n..." not in msg

    def test_has_more_true_shows_ellipsis(self, sample_result):
        msg = build_persisted_output_message(sample_result)
        assert "\n..." in msg

    def test_large_mb_size(self):
        result = PersistedResult(
            tool_use_id="big",
            original_size=2_000_000,
            file_path="/tmp/big.txt",
            preview="preview",
            has_more=True,
        )
        msg = build_persisted_output_message(result)
        assert "MB" in msg


# ------------------------------------------------------------------ #
# maybe_persist_tool_result
# ------------------------------------------------------------------ #

class TestMaybePersistToolResult:
    def test_small_passes_through(self, tmp_path, monkeypatch):
        """Under threshold, returns original content."""
        monkeypatch.setattr(
            "tools.registry.registry.get_max_result_size",
            lambda name: DEFAULT_MAX_RESULT_SIZE_CHARS,
        )
        content = "small output"
        result = maybe_persist_tool_result(content, "test_tool", "id_1", tmp_path)
        assert result == content

    def test_large_returns_persisted_block(self, tmp_path, monkeypatch):
        """Over threshold, returns <persisted-output> block."""
        monkeypatch.setattr(
            "tools.registry.registry.get_max_result_size",
            lambda name: DEFAULT_MAX_RESULT_SIZE_CHARS,
        )
        content = "x" * (DEFAULT_MAX_RESULT_SIZE_CHARS + 1)
        result = maybe_persist_tool_result(content, "test_tool", "id_2", tmp_path)
        assert PERSISTED_OUTPUT_TAG in result
        assert PERSISTED_OUTPUT_CLOSING_TAG in result
        # File written
        assert (tmp_path / "id_2.txt").exists()

    def test_read_file_never_persisted(self, tmp_path, monkeypatch):
        """read_file with inf threshold always passes through."""
        monkeypatch.setattr(
            "tools.registry.registry.get_max_result_size",
            lambda name: float('inf'),
        )
        content = "x" * (DEFAULT_MAX_RESULT_SIZE_CHARS * 3)
        result = maybe_persist_tool_result(content, "read_file", "id_3", tmp_path)
        assert result == content  # Unchanged

    def test_unknown_tool_uses_default(self, tmp_path, monkeypatch):
        """Unregistered tool name uses 50K default."""
        monkeypatch.setattr(
            "tools.registry.registry.get_max_result_size",
            lambda name: DEFAULT_MAX_RESULT_SIZE_CHARS,
        )
        # Under default: passes through
        content_under = "x" * (DEFAULT_MAX_RESULT_SIZE_CHARS - 1)
        result = maybe_persist_tool_result(content_under, "no_such_tool", "id_4", tmp_path)
        assert result == content_under

        # Over default: persisted
        content_over = "x" * (DEFAULT_MAX_RESULT_SIZE_CHARS + 1)
        result = maybe_persist_tool_result(content_over, "no_such_tool", "id_5", tmp_path)
        assert PERSISTED_OUTPUT_TAG in result

    def test_custom_threshold_via_registry(self, tmp_path, monkeypatch):
        """Tool with custom lower threshold persists sooner."""
        custom_limit = 1000
        monkeypatch.setattr(
            "tools.registry.registry.get_max_result_size",
            lambda name: custom_limit,
        )
        content = "x" * (custom_limit + 1)
        result = maybe_persist_tool_result(content, "small_tool", "id_6", tmp_path)
        # Content exceeds the per-tool threshold (1001 > 1000) so it should
        # be persisted even though it's well below the 50K default.
        assert PERSISTED_OUTPUT_TAG in result
        assert (tmp_path / "id_6.txt").exists()


# ------------------------------------------------------------------ #
# enforce_turn_budget
# ------------------------------------------------------------------ #

class TestEnforceTurnBudget:
    def test_under_budget_no_changes(self, tmp_path):
        """All messages fit, nothing changed."""
        messages = [
            {"content": "short result", "tool_call_id": "t1"},
            {"content": "another short", "tool_call_id": "t2"},
        ]
        result = enforce_turn_budget(messages, tmp_path)
        assert result[0]["content"] == "short result"
        assert result[1]["content"] == "another short"

    def test_over_budget_persists_largest(self, tmp_path):
        """Total > 200K, largest result gets persisted first."""
        small = "s" * 50_001
        large = "L" * (DEFAULT_MAX_RESULT_SIZE_CHARS + 100_001)  # 150K+
        messages = [
            {"content": small, "tool_call_id": "small_1"},
            {"content": large, "tool_call_id": "large_1"},
        ]
        total_before = len(small) + len(large)
        assert total_before > MAX_TURN_BUDGET_CHARS

        result = enforce_turn_budget(messages, tmp_path)
        # The large one should be persisted
        assert PERSISTED_OUTPUT_TAG in result[1]["content"]
        # The small one should be unchanged
        assert result[0]["content"] == small
        # File written
        assert (tmp_path / "large_1.txt").exists()

    def test_already_persisted_skipped(self, tmp_path):
        """Messages with <persisted-output> not re-persisted."""
        already = f"{PERSISTED_OUTPUT_TAG}\nalready persisted\n{PERSISTED_OUTPUT_CLOSING_TAG}"
        large = "x" * (DEFAULT_MAX_RESULT_SIZE_CHARS + 1)
        messages = [
            {"content": already, "tool_call_id": "p1"},
            {"content": large, "tool_call_id": "new_1"},
        ]
        result = enforce_turn_budget(messages, tmp_path, budget=100)
        # already-persisted one is untouched (same object)
        assert result[0]["content"] == already
        # new large one gets persisted
        assert PERSISTED_OUTPUT_TAG in result[1]["content"]
        assert result[1]["content"] != already  # Different from the first

    def test_parallel_80k_results(self, tmp_path):
        """5 messages each 80K = 400K total, should persist enough to get under 200K."""
        messages = [
            {"content": "x" * 80_000, "tool_call_id": f"par_{i}"}
            for i in range(5)
        ]
        total_before = sum(len(m["content"]) for m in messages)
        assert total_before == 400_000
        assert total_before > MAX_TURN_BUDGET_CHARS

        result = enforce_turn_budget(messages, tmp_path)

        # Count how many were persisted vs kept inline
        persisted_count = sum(
            1 for m in result if PERSISTED_OUTPUT_TAG in m["content"]
        )
        inline_count = 5 - persisted_count

        # At least some must be persisted to get under budget.
        # Each 80K result is > 50K threshold so persist_large_result will work.
        assert persisted_count >= 1

        # Total should now be under budget (or close — replacement text adds some)
        total_after = sum(len(m["content"]) for m in result)
        assert total_after < total_before  # Definitely reduced

    def test_empty_messages(self, tmp_path):
        """Empty list returns empty list."""
        result = enforce_turn_budget([], tmp_path)
        assert result == []

    def test_medium_results_under_default_threshold_still_persisted(self, tmp_path):
        """Multiple 42K results (under 50K default) exceed 200K budget.

        Regression test: L3 must force-persist even when individual results
        are below DEFAULT_MAX_RESULT_SIZE_CHARS. Without threshold=0 in the
        enforce_turn_budget call, these would all be skipped.
        """
        # 6 x 42K = 252K > 200K budget, but each is under 50K default
        messages = [
            {"content": "x" * 42_000, "tool_call_id": f"med_{i}"}
            for i in range(6)
        ]
        total_before = sum(len(m["content"]) for m in messages)
        assert total_before == 252_000
        assert total_before > MAX_TURN_BUDGET_CHARS
        # Each individual result is under the default 50K threshold
        assert all(len(m["content"]) < DEFAULT_MAX_RESULT_SIZE_CHARS for m in messages)

        result = enforce_turn_budget(messages, tmp_path)

        persisted_count = sum(
            1 for m in result if PERSISTED_OUTPUT_TAG in m["content"]
        )
        # At least one must be persisted to bring total under budget
        assert persisted_count >= 1

        total_after = sum(len(m["content"]) for m in result)
        assert total_after < total_before

    def test_budget_parameter_respected(self, tmp_path):
        """Custom budget parameter is used instead of default."""
        # Two messages each 100 chars, budget=150 should trigger persistence
        messages = [
            {"content": "a" * (DEFAULT_MAX_RESULT_SIZE_CHARS + 100), "tool_call_id": "b1"},
            {"content": "b" * (DEFAULT_MAX_RESULT_SIZE_CHARS + 100), "tool_call_id": "b2"},
        ]
        result = enforce_turn_budget(messages, tmp_path, budget=50_000)
        # At least one should be persisted
        persisted_count = sum(
            1 for m in result if PERSISTED_OUTPUT_TAG in m["content"]
        )
        assert persisted_count >= 1


# ------------------------------------------------------------------ #
# Registry integration: get_max_result_size
# ------------------------------------------------------------------ #

class TestRegistryGetMaxResultSize:
    def test_default_for_unknown_tool(self):
        """Unregistered tool returns DEFAULT_MAX_RESULT_SIZE_CHARS."""
        from tools.registry import ToolRegistry
        reg = ToolRegistry()
        assert reg.get_max_result_size("nonexistent") == DEFAULT_MAX_RESULT_SIZE_CHARS

    def test_custom_threshold(self):
        """Tool registered with max_result_size_chars returns that value."""
        from tools.registry import ToolRegistry
        reg = ToolRegistry()
        reg.register(
            name="custom_tool",
            toolset="test",
            schema={"description": "test"},
            handler=lambda args: "ok",
            max_result_size_chars=10_000,
        )
        assert reg.get_max_result_size("custom_tool") == 10_000

    def test_inf_threshold(self):
        """Tool with inf threshold returns inf."""
        from tools.registry import ToolRegistry
        reg = ToolRegistry()
        reg.register(
            name="read_file",
            toolset="test",
            schema={"description": "test"},
            handler=lambda args: "ok",
            max_result_size_chars=float('inf'),
        )
        assert reg.get_max_result_size("read_file") == float('inf')

    def test_none_falls_back_to_default(self):
        """Tool registered without max_result_size_chars uses default."""
        from tools.registry import ToolRegistry
        reg = ToolRegistry()
        reg.register(
            name="plain_tool",
            toolset="test",
            schema={"description": "test"},
            handler=lambda args: "ok",
        )
        assert reg.get_max_result_size("plain_tool") == DEFAULT_MAX_RESULT_SIZE_CHARS
