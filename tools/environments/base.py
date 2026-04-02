"""Base class for all Hermes execution environment backends."""

from abc import ABC, abstractmethod
import logging
import os
import shlex
import threading
import time
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

from hermes_constants import get_hermes_home
from tools.interrupt import is_interrupted

logger = logging.getLogger(__name__)

# Marker echoed to stdout by the wrapping template so the local Hermes
# process can extract the remote shell's cwd without a separate round-trip.
_CWD_MARKER = "__HERMES_CWD__"

# Min seconds between file-sync checks in _before_execute hooks.
# Remote backends (SSH, Modal, Daytona) skip re-walking the skills
# directory and re-statting credential files within this window.
_SYNC_INTERVAL_SECONDS: float = 5.0


def get_sandbox_dir() -> Path:
    """Return the host-side root for all sandbox storage (Docker workspaces,
    Singularity overlays/SIF cache, etc.).

    Configurable via TERMINAL_SANDBOX_DIR. Defaults to {HERMES_HOME}/sandboxes/.
    """
    custom = os.getenv("TERMINAL_SANDBOX_DIR")
    if custom:
        p = Path(custom)
    else:
        p = get_hermes_home() / "sandboxes"
    p.mkdir(parents=True, exist_ok=True)
    return p


@runtime_checkable
class ProcessHandle(Protocol):
    """Duck type for anything _run_bash returns.

    subprocess.Popen satisfies this natively.  SDK backends (Modal, Daytona)
    return small adapters that wrap async/blocking calls in a thread + OS pipe.
    """

    def poll(self) -> int | None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...

    @property
    def stdout(self): ...  # readable, iterable-of-str (for drain thread)

    @property
    def returncode(self) -> int | None: ...


class _ThreadedProcessHandle:
    """ProcessHandle adapter for SDK backends that run in a background thread."""

    def __init__(self, exec_fn):
        self._done = threading.Event()
        self._returncode = None
        self._read_fd, self._write_fd = os.pipe()
        self.stdout = os.fdopen(self._read_fd, "r")
        self.stdin = None

        def _run():
            # Open the write end exactly once to avoid double-close races.
            writer = os.fdopen(self._write_fd, "w")
            try:
                output, exit_code = exec_fn()
                writer.write(output)
                self._returncode = exit_code
            except Exception as e:
                try:
                    writer.write(str(e))
                except Exception:
                    pass
                self._returncode = 1
            finally:
                try:
                    writer.close()
                except Exception:
                    pass
                self._done.set()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def poll(self):
        return self._returncode if self._done.is_set() else None

    def kill(self):
        pass

    def wait(self, timeout=None):
        self._done.wait(timeout=timeout)
        return self._returncode

    @property
    def returncode(self):
        return self._returncode


class BaseEnvironment(ABC):
    """Common interface for all Hermes execution backends.

    **Unified execution model (spawn-per-call):**

    Backends implement ``_run_bash()`` — the ONLY thing that differs per
    backend.  Everything else (command wrapping, CWD tracking, snapshot
    management, timeout/interrupt handling, output collection) lives here.

    Backends that cannot return a ProcessHandle (e.g. HTTP-based
    ManagedModal) may override ``execute()`` directly and use
    ``_wrap_command()`` for command shaping only.
    """

    def __init__(self, cwd: str, timeout: int, env: dict = None):
        self.cwd = cwd
        self.timeout = timeout
        self.env = env or {}
        self._snapshot_path: str | None = None
        self._snapshot_ready: bool = False
        self._session_id: str = ""

    # ------------------------------------------------------------------
    # Abstract — the ONLY thing backends implement
    # ------------------------------------------------------------------

    def _run_bash(self, cmd_string: str, *,
                  timeout: int | None = None,
                  stdin_data: str | None = None) -> ProcessHandle:
        """Spawn ``bash -c <cmd_string>`` in the backend.

        Returns a ProcessHandle (subprocess.Popen or equivalent adapter).
        The caller owns polling, timeout, output collection, and cleanup.

        *timeout* is the effective per-command timeout.  Backends that use
        SDK-level or shell-level timeouts (Modal, Daytona) should forward
        this value.  Backends where timeout is enforced by
        ``_wait_for_process`` (local, docker, ssh, singularity) may ignore it.

        If *stdin_data* is provided, write it to the process's stdin and
        close.  Backends that cannot pipe stdin (Modal, Daytona) must embed
        it via heredoc in *cmd_string* before calling their SDK.

        Subclasses MUST override this.  The base implementation raises
        NotImplementedError (not declared abstract so legacy backends that
        still override execute() directly can be instantiated during migration).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _run_bash()"
        )

    def cleanup(self):
        """Release backend resources (container, instance, connection).

        Subclasses should override.  Base implementation cleans up snapshot
        and cwdfile if they exist.
        """
        pass

    # ------------------------------------------------------------------
    # Snapshot — login-shell env capture (called once at session init)
    # ------------------------------------------------------------------

    def _run_bash_login(self, cmd_string: str, *,
                        timeout: int | None = None) -> ProcessHandle:
        """Spawn ``bash -l -c <cmd_string>`` for snapshot creation.

        Defaults to ``_run_bash`` — backends override this when the login
        flag needs different handling (e.g. local adds ``-l`` to Popen args).
        """
        return self._run_bash(cmd_string, timeout=timeout)

    _snapshot_timeout: int = 15

    def init_session(self):
        """Capture the login-shell environment into a snapshot file.

        Called once after ``__init__`` completes.  If it fails, commands
        still work — they just don't get env restoration.
        """
        self._session_id = uuid.uuid4().hex[:12]
        self._snapshot_path = f"/tmp/hermes-snap-{self._session_id}.sh"

        bootstrap = (
            f"set +e\n"
            f"export -p > {self._snapshot_path}\n"
            f"if type declare >/dev/null 2>&1; then "
            f"declare -f >> {self._snapshot_path} 2>/dev/null; fi\n"
            f"alias -p >> {self._snapshot_path} 2>/dev/null || true\n"
            f"echo 'shopt -s expand_aliases' >> {self._snapshot_path}\n"
            f"echo 'set +e' >> {self._snapshot_path}\n"
            f"echo 'set +u' >> {self._snapshot_path}\n"
            f"printf '{_CWD_MARKER}%s{_CWD_MARKER}' \"$(pwd -P)\"\n"
        )

        result = {}
        try:
            proc = self._run_bash_login(bootstrap, timeout=self._snapshot_timeout)
            result = self._wait_for_process(proc, timeout=self._snapshot_timeout)
            if result["returncode"] == 0:
                self._snapshot_ready = True
                logger.info(
                    "Snapshot created (session=%s)", self._session_id,
                )
            else:
                logger.warning(
                    "Snapshot creation failed (rc=%d), commands will "
                    "run without env restoration", result["returncode"],
                )
        except Exception as e:
            logger.warning("Snapshot creation failed: %s", e)

        self._extract_cwd_from_output(result)

    # ------------------------------------------------------------------
    # Command wrapping
    # ------------------------------------------------------------------

    @staticmethod
    def _embed_stdin_heredoc(cmd_string: str, stdin_data: str) -> str:
        """Wrap *stdin_data* as a shell heredoc appended to *cmd_string*.

        Used by backends that cannot pipe stdin (Modal, Daytona, ManagedModal).
        """
        marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
        while marker in stdin_data:
            marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
        return f"{cmd_string} << '{marker}'\n{stdin_data}\n{marker}"

    def _resolve_tilde(self, path: str) -> str:
        """Expand ``~`` to the actual home directory path.

        Remote backends (SSH, Daytona) set ``_remote_home`` during init;
        local uses ``os.path.expanduser``.  Tilde must be resolved before
        ``shlex.quote`` since single-quoting prevents shell tilde expansion.
        """
        if not path or not path.startswith("~"):
            return path
        home = getattr(self, "_remote_home", None) or os.path.expanduser("~")
        if path == "~":
            return home
        if path.startswith("~/"):
            return home + path[1:]
        return path  # ~otheruser — leave for shell to handle

    def _wrap_command(self, command: str, cwd: str) -> str:
        """Wrap a user command with snapshot sourcing and CWD tracking.

        Returns a bash script string.
        """
        parts: list[str] = []

        # 1. Source snapshot (if available)
        if self._snapshot_ready and self._snapshot_path:
            parts.append(
                f"source {self._snapshot_path} 2>/dev/null || true"
            )

        # 2. cd to working directory (resolve ~ before quoting)
        work_dir = cwd or self.cwd
        if work_dir:
            work_dir = self._resolve_tilde(work_dir)
            parts.append(f"cd {shlex.quote(work_dir)} || exit 1")

        # 3. The actual command (eval to handle complex shell syntax)
        escaped = command.replace("'", "'\\''")
        parts.append(f"eval '{escaped}'")

        # 4. Capture exit code, record CWD
        parts.append("__hermes_ec=$?")
        parts.append(
            f"printf '\\n{_CWD_MARKER}%s{_CWD_MARKER}\\n' \"$(pwd -P)\""
        )
        parts.append("exit $__hermes_ec")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Unified execute()
    # ------------------------------------------------------------------

    def execute(self, command: str, cwd: str = "", *,
                timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        """Execute a command, return ``{"output": str, "returncode": int}``."""
        self._before_execute()

        exec_command, sudo_stdin = self._prepare_command(command)

        # Merge sudo stdin with caller stdin
        effective_stdin: str | None = None
        if sudo_stdin is not None and stdin_data is not None:
            effective_stdin = sudo_stdin + stdin_data
        elif sudo_stdin is not None:
            effective_stdin = sudo_stdin
        else:
            effective_stdin = stdin_data

        wrapped = self._wrap_command(exec_command, cwd)
        effective_timeout = timeout or self.timeout

        proc = self._run_bash(wrapped, timeout=effective_timeout,
                              stdin_data=effective_stdin)
        result = self._wait_for_process(proc, timeout=effective_timeout)

        self._extract_cwd_from_output(result)

        return result

    # ------------------------------------------------------------------
    # Process lifecycle (shared — not overridden except _kill_process)
    # ------------------------------------------------------------------

    def _wait_for_process(self, proc: ProcessHandle,
                          timeout: int) -> dict:
        """Poll process with interrupt checking, drain stdout, enforce timeout."""
        output_chunks: list[str] = []

        def _drain():
            try:
                for line in proc.stdout:
                    output_chunks.append(line)
            except (ValueError, OSError):
                pass

        reader = threading.Thread(target=_drain, daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout

        try:
            while proc.poll() is None:
                if is_interrupted():
                    self._kill_process(proc)
                    reader.join(timeout=2)
                    partial = "".join(output_chunks)
                    return {
                        "output": partial + "\n[Command interrupted]",
                        "returncode": 130,
                    }
                if time.monotonic() > deadline:
                    self._kill_process(proc)
                    reader.join(timeout=2)
                    partial = "".join(output_chunks)
                    msg = f"\n[Command timed out after {timeout}s]"
                    return {
                        "output": (partial + msg) if partial else msg.lstrip(),
                        "returncode": 124,
                    }
                time.sleep(0.2)

            reader.join(timeout=5)
            return {"output": "".join(output_chunks), "returncode": proc.returncode}
        finally:
            # Close the stdout pipe to prevent FD leaks, especially for
            # SDK-backed handles (Modal, Daytona) that use os.pipe().
            try:
                proc.stdout.close()
            except Exception:
                pass

    def _kill_process(self, proc: ProcessHandle):
        """Kill a process.  Backends may override for process-group kill."""
        try:
            if hasattr(proc, "terminate"):
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                    return
                except Exception:
                    pass
            proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass

    # ------------------------------------------------------------------
    # CWD tracking
    # ------------------------------------------------------------------

    def _extract_cwd_from_output(self, result: dict) -> dict:
        """Parse CWD marker from command output, update self.cwd, strip marker.

        The wrapping template echoes ``__HERMES_CWD__/path__HERMES_CWD__``
        to stdout.  This method extracts the path, updates ``self.cwd``,
        and removes the marker from the output so the caller sees clean output.
        """
        output = result.get("output", "")
        ml = len(_CWD_MARKER)
        # Find the last pair: look for the second-to-last marker (open),
        # then the last marker (close).
        close = output.rfind(_CWD_MARKER)
        if close == -1:
            return result
        open_ = output.rfind(_CWD_MARKER, 0, close)
        if open_ == -1:
            return result
        cwd = output[open_ + ml:close].strip()
        if cwd:
            self.cwd = cwd
        # Strip the marker and surrounding whitespace from output
        before = output[:open_].rstrip("\n")
        after = output[close + ml:].lstrip("\n")
        result["output"] = (before + after) if after else before
        return result

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _before_execute(self):
        """Hook for pre-execution sync (SSH rsync, Modal file push, etc.)."""
        pass

    # ------------------------------------------------------------------
    # Compat
    # ------------------------------------------------------------------

    def stop(self):
        """Alias for cleanup (compat with older callers)."""
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _prepare_command(self, command: str) -> tuple[str, str | None]:
        """Transform sudo commands if SUDO_PASSWORD is available."""
        from tools.terminal_tool import _transform_sudo_command
        return _transform_sudo_command(command)
