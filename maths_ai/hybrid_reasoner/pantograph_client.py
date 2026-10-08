"""Owned serialized Pantograph transport and cancellation-safe process lifetime.

Stage 2 implements contract startup, stat, and private transactions. Typed goal
operations and lexical allocation owners are the subsequent Stage 3 deliverable.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from enum import Enum
import os
import signal
from typing import TYPE_CHECKING, Any, Callable, Mapping
from uuid import uuid4

from .pantograph_launch import PantographLaunchSpec, build_launch_spec
from .pantograph_protocol import (
    CommandAcknowledgement,
    CommandError,
    GoalStartResult,
    PantographConfigurationError,
    PantographGoalRejected,
    PantographProtocolError,
    PantographRequestTimeout,
    PantographStartupError,
    PantographStateError,
    PantographStats,
    PantographTransportError,
    Response,
    SessionOptions,
    TacticSuccess,
    decode_json_response,
    decode_response,
    encode_request,
)

if TYPE_CHECKING:
    from .pantograph_env import PantographEnv


class SessionPhase(str, Enum):
    STARTING = "STARTING"
    READY = "READY"
    FAILED = "FAILED"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


def _observe(task: asyncio.Future) -> None:
    # Retained background failures are still raised when their owner awaits them.
    if not task.cancelled():
        task.exception()


async def _join_cleanup(task: asyncio.Task) -> Any:
    """Delay caller cancellation until the retained cleanup task has finished."""
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is None:
                raise
            cancellation.add_note(f"Cleanup also failed: {exc}")
            raise cancellation
    if cancellation is not None:
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                cancellation.add_note(f"Cleanup also failed: {error}")
        raise cancellation
    return task.result()


async def _discard(stream: asyncio.StreamReader) -> None:
    """Keep a rejected pipe flowing until termination closes its writer."""
    while await stream.read(4096):
        pass


async def _drain_tail(
    stream: asyncio.StreamReader, buffer: bytearray, limit: int
) -> bytes:
    while chunk := await stream.read(4096):
        buffer.extend(chunk)
        del buffer[:-limit]
    return bytes(buffer)


class _ProcessOwner:
    """One spawn, one retained cleanup task, and all pipe-reader tasks."""

    def __init__(self, shutdown_timeout_s: float, stderr_tail_limit: int):
        self.loop = asyncio.get_running_loop()
        self.shutdown_timeout_s = shutdown_timeout_s
        self.stderr_tail_limit = stderr_tail_limit
        self.stderr = bytearray()
        self.process: asyncio.subprocess.Process | None = None
        self.pid: int | None = None
        self.spawn_task: asyncio.Task | None = None
        self.close_task: asyncio.Task | None = None
        self.readers: list[asyncio.Task] = []

    async def spawn(self, *command: str, **kwargs) -> asyncio.subprocess.Process:
        if self.spawn_task is not None or self.close_task is not None:
            raise PantographStateError("An owned process lifetime cannot be restarted.")
        self.spawn_task = self.loop.create_task(
            asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                **kwargs,
            ),
            name="pantograph-spawn",
        )
        self.spawn_task.add_done_callback(_observe)
        self.process = await asyncio.shield(self.spawn_task)
        self.pid = self.process.pid
        return self.process

    def reader(self, coroutine, name: str) -> asyncio.Task:
        task = self.loop.create_task(coroutine, name=name)
        task.add_done_callback(_observe)
        self.readers.append(task)
        return task

    async def aclose(self) -> None:
        if self.close_task is None:
            self.close_task = self.loop.create_task(
                self._shutdown(), name="pantograph-process-close"
            )
            self.close_task.add_done_callback(_observe)
        await _join_cleanup(self.close_task)

    async def _shutdown(self) -> None:
        deadline = self.loop.time() + self.shutdown_timeout_s
        wait_task = None
        try:
            async with asyncio.timeout_at(deadline):
                if self.spawn_task is not None:
                    if not self.spawn_task.done():
                        self.spawn_task.cancel()
                    await asyncio.gather(self.spawn_task, return_exceptions=True)
                    if (
                        not self.spawn_task.cancelled()
                        and self.spawn_task.exception() is None
                    ):
                        self.process = self.spawn_task.result()
                        self.pid = self.process.pid
                process = self.process
                # Join old readers before handing each pipe to cleanup's drainer.
                # This also covers cancellation before normal readers are installed.
                for reader in self.readers:
                    reader.cancel()
                await asyncio.gather(*self.readers, return_exceptions=True)
                self.readers.clear()
                if process is not None:
                    self.reader(_discard(process.stdout), "pantograph-close-stdout")
                    self.reader(
                        _drain_tail(
                            process.stderr, self.stderr, self.stderr_tail_limit
                        ),
                        "pantograph-close-stderr",
                    )
                    if process.stdin is not None:
                        process.stdin.close()
                    if process.returncode is None:
                        self._signal_group(signal.SIGTERM)
                    wait_task = self.loop.create_task(
                        process.wait(), name="pantograph-reap"
                    )
                    wait_task.add_done_callback(_observe)
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(wait_task), self.shutdown_timeout_s / 2
                        )
                    except TimeoutError:
                        self._signal_group(signal.SIGKILL)
                        await wait_task
                await asyncio.gather(*self.readers, return_exceptions=True)
        except TimeoutError as exc:
            self._signal_group(signal.SIGKILL)
            for reader in self.readers:
                reader.cancel()
            await asyncio.gather(*self.readers, return_exceptions=True)
            if wait_task is not None and not wait_task.done():
                wait_task.cancel()
                await asyncio.gather(wait_task, return_exceptions=True)
            raise PantographTransportError(
                f"Shutdown deadline expired; reaping could not be confirmed for pid={self.pid}."
            ) from exc
        self.process = None
        self.readers.clear()

    def _signal_group(self, sig: signal.Signals) -> None:
        if self.process is not None:
            try:
                # Each owned child is launched in its own session/process group.
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass


async def _capture_tool(
    command: tuple[str, ...],
    spec: PantographLaunchSpec,
    environment: dict[str, str],
    deadline: float,
) -> bytes:
    owner = _ProcessOwner(spec.shutdown_timeout_s, spec.stderr_tail_limit)
    cancellation = None

    async def collect(stream: asyncio.StreamReader) -> bytes:
        output = bytearray()
        while chunk := await stream.read(4096):
            output.extend(chunk)
            if len(output) > spec.response_line_limit:
                raise PantographConfigurationError(
                    "Environment helper output exceeds the response limit."
                )
        return bytes(output)

    try:
        async with asyncio.timeout_at(deadline):
            process = await owner.spawn(*command, cwd=spec.source_root, env=environment)
            stdout = owner.reader(collect(process.stdout), "pantograph-helper-stdout")
            stderr = owner.reader(
                _drain_tail(process.stderr, owner.stderr, spec.stderr_tail_limit),
                "pantograph-helper-stderr",
            )
            out, err = await asyncio.gather(stdout, stderr)
            code = await process.wait()
            if code != 0:
                raise PantographConfigurationError(
                    f"Environment helper {command!r} exited {code}: {err[-spec.stderr_tail_limit:].decode('utf-8', errors='replace')}"
                )
            return out
    except TimeoutError as exc:
        raise PantographStartupError(
            f"Startup deadline expired resolving environment: {command!r}"
        ) from exc
    except OSError as exc:
        raise PantographConfigurationError(
            f"Cannot execute environment helper {command!r}: {exc}"
        ) from exc
    except asyncio.CancelledError as exc:
        cancellation = exc
        raise
    finally:
        try:
            await owner.aclose()
        except BaseException as exc:
            if cancellation is None:
                raise
            if not isinstance(exc, asyncio.CancelledError):
                cancellation.add_note(f"Environment helper cleanup also failed: {exc}")
            raise cancellation


class PantographClient:
    def __init__(self, spec: PantographLaunchSpec):
        self.launch_spec = spec
        self.session_id = uuid4().hex
        self._loop = asyncio.get_running_loop()
        self._owner = _ProcessOwner(spec.shutdown_timeout_s, spec.stderr_tail_limit)
        self._phase = SessionPhase.STARTING
        self._lock = asyncio.Lock()
        self._ready = self._loop.create_future()
        self._ready.add_done_callback(_observe)
        self._reply: asyncio.Future | None = None
        self._close_task: asyncio.Task | None = None
        self._failure: Exception | None = None
        self._stderr = self._owner.stderr
        self._allocated_ids: set[int] = set()
        self.effective_options: SessionOptions | None = None

    @classmethod
    async def create(cls, env: "PantographEnv") -> "PantographClient":
        started = asyncio.get_running_loop().time()
        client = cls(build_launch_spec(env))
        deadline = started + client.launch_spec.startup_timeout_s
        spec = client.launch_spec
        environment = dict(os.environ)
        try:
            async with asyncio.timeout_at(deadline):
                version = await _capture_tool(
                    (str(spec.stdbuf), "--version"), spec, environment, deadline
                )
                if b"GNU coreutils" not in version:
                    raise PantographConfigurationError(
                        "The configured stdbuf is not GNU coreutils stdbuf."
                    )
                lean_path = await _capture_tool(
                    (str(spec.lake), "env", "printenv", "LEAN_PATH"),
                    spec,
                    environment,
                    deadline,
                )
                try:
                    resolved = lean_path.decode("utf-8").strip()
                except UnicodeDecodeError as exc:
                    raise PantographConfigurationError(
                        "Lake returned a non-UTF-8 LEAN_PATH."
                    ) from exc
                if not resolved or "\n" in resolved or "\x00" in resolved:
                    raise PantographConfigurationError(
                        "Lake returned empty or invalid LEAN_PATH."
                    )
                environment["LEAN_PATH"] = resolved
                process = await client._owner.spawn(
                    *spec.command,
                    cwd=spec.source_root,
                    env=environment,
                    limit=spec.response_line_limit + 1,
                )
                client._owner.reader(
                    client._read_stdout(process.stdout), "pantograph-stdout"
                )
                client._owner.reader(
                    client._read_stderr(process.stderr), "pantograph-stderr"
                )
                await asyncio.shield(client._ready)
                await client._exchange(
                    "protocol.describe", {}, deadline=deadline, starting=True
                )
                await client._exchange(
                    "options.set", dict(spec.options), deadline=deadline, starting=True
                )
                result = await client._exchange(
                    "options.print", {}, deadline=deadline, starting=True
                )
                client.effective_options = result
                for key, value in spec.options:
                    if result.to_dict()[key] is not value:
                        raise PantographProtocolError(
                            f"Requested option {key} was not applied."
                        )
                if client._phase is not SessionPhase.STARTING:
                    raise client._failure or PantographStartupError(
                        "Session closed during startup."
                    )
                client._phase = SessionPhase.READY
            return client
        except BaseException as exc:
            client._fail(
                exc
                if isinstance(exc, Exception)
                else PantographTransportError("Startup cancelled.")
            )
            await client._close_after_error(exc)
            if isinstance(exc, (TimeoutError, PantographRequestTimeout)):
                raise PantographStartupError(
                    f"Startup deadline expired; {client._context()}"
                ) from exc
            if isinstance(exc, OSError):
                raise PantographStartupError(
                    f"Could not launch REPL: {exc}; {client._context()}"
                ) from exc
            if isinstance(exc, PantographProtocolError):
                raise PantographProtocolError(
                    f"{exc}; {client._context()}",
                    command=exc.command,
                    payload=exc.payload,
                ) from exc
            raise

    @property
    def phase(self) -> SessionPhase:
        return self._phase

    @property
    def is_usable(self) -> bool:
        return self._phase is SessionPhase.READY

    @property
    def pid(self) -> int | None:
        return self._owner.pid

    @property
    def stderr_tail(self) -> str:
        return self._stderr.decode("utf-8", errors="replace")

    def _check_loop(self) -> None:
        if asyncio.get_running_loop() is not self._loop:
            raise PantographStateError(
                "A Pantograph client must stay on its owning event loop."
            )

    def _context(self) -> str:
        return f"command={self.launch_spec.command!r} source_root={self.launch_spec.source_root} phase={self.phase.value} stderr={self.stderr_tail!r}"

    async def stat(self) -> PantographStats:
        return await self._exchange("stat", {})

    async def __aenter__(self) -> "PantographClient":
        self._check_loop()
        if not self.is_usable:
            raise PantographStateError("Cannot enter a closed or failed client.")
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.aclose()

    async def _read_stderr(self, stream: asyncio.StreamReader) -> None:
        try:
            await _drain_tail(stream, self._stderr, self.launch_spec.stderr_tail_limit)
        except Exception as exc:
            self._fail(PantographTransportError(f"Failed to drain stderr: {exc}"))

    async def _read_stdout(self, stream: asyncio.StreamReader) -> None:
        first = True
        try:
            while True:
                line = await stream.readline()
                if not line:
                    if self._phase not in (SessionPhase.CLOSING, SessionPhase.CLOSED):
                        self._fail(
                            PantographTransportError(
                                f"REPL stdout reached EOF; {self._context()}"
                            )
                        )
                    return
                if self._phase in (SessionPhase.CLOSING, SessionPhase.CLOSED):
                    continue
                if len(
                    line
                ) > self.launch_spec.response_line_limit or not line.endswith(b"\n"):
                    raise PantographTransportError(
                        "REPL response exceeds the line limit or lacks its newline."
                    )
                if first:
                    first = False
                    if line != b"ready.\n":
                        raise PantographStartupError(
                            f"Invalid readiness line; {self._context()}"
                        )
                    self._ready.set_result(None)
                elif self._reply is None or self._reply.done():
                    raise PantographProtocolError(
                        f"Unsolicited REPL stdout; {self._context()}"
                    )
                else:
                    self._reply.set_result(line)
        except Exception as exc:
            error = (
                exc
                if isinstance(
                    exc,
                    (
                        PantographProtocolError,
                        PantographTransportError,
                        PantographStartupError,
                    ),
                )
                else PantographTransportError(
                    f"Cannot read REPL stdout: {exc}; {self._context()}"
                )
            )
            self._fail(error)
            await _discard(stream)

    def _fail(self, error: Exception) -> None:
        if self._phase in (SessionPhase.CLOSING, SessionPhase.CLOSED):
            return
        if self._failure is None:
            self._failure = error
        self._phase = SessionPhase.FAILED
        self._allocated_ids.clear()
        for future in (self._ready, self._reply):
            if future is not None and not future.done():
                future.set_exception(self._failure)
        self._begin_close()

    def _begin_close(self) -> asyncio.Task:
        if self._close_task is None:
            self._phase = SessionPhase.CLOSING
            self._close_task = self._loop.create_task(
                self._shutdown(), name="pantograph-client-close"
            )
            self._close_task.add_done_callback(_observe)
        return self._close_task

    async def _shutdown(self) -> None:
        for future in (self._ready, self._reply):
            if future is not None and not future.done():
                future.set_exception(
                    self._failure
                    or PantographTransportError("Client closed during a transaction.")
                )
        await self._owner.aclose()
        self._allocated_ids.clear()
        self._phase = SessionPhase.CLOSED

    async def aclose(self) -> None:
        self._check_loop()
        await _join_cleanup(self._begin_close())

    async def _close_after_error(self, original: BaseException) -> None:
        try:
            await self.aclose()
        except asyncio.CancelledError:
            if not isinstance(original, asyncio.CancelledError):
                raise
        except Exception as exc:
            if not isinstance(original, asyncio.CancelledError):
                raise
            original.add_note(f"Cleanup also failed: {exc}")

    async def _exchange(
        self,
        command: str,
        payload: Mapping[str, Any],
        *,
        deadline: float | None = None,
        starting: bool = False,
        record: Callable[[Response], None] | None = None,
    ) -> Response:
        """Private transaction; synchronous bookkeeping precedes lock release."""
        self._check_loop()
        permitted = SessionPhase.STARTING if starting else SessionPhase.READY
        if self._phase is not permitted:
            if self._close_task is not None:
                await self.aclose()
            raise self._failure or PantographStateError(
                "Client is closed or unavailable."
            )
        deadline = (
            self._loop.time() + self.launch_spec.request_timeout_s
            if deadline is None
            else deadline
        )
        body = deepcopy(dict(payload))
        request = encode_request(command, body)
        acquired = False
        wrote = False
        try:
            async with asyncio.timeout_at(deadline):
                if self._loop.time() >= deadline:
                    raise TimeoutError
                await self._lock.acquire()
                acquired = True
                if self._phase is not permitted:
                    raise self._failure or PantographStateError(
                        "Client closed while waiting for a transaction."
                    )
                if self._loop.time() >= deadline:
                    raise TimeoutError
                self._reply = self._loop.create_future()
                self._reply.add_done_callback(_observe)
                process = self._owner.process
                wrote = True
                process.stdin.write(request)
                await process.stdin.drain()
                line = await asyncio.shield(self._reply)
                if self._phase is not permitted:
                    raise self._failure or PantographTransportError(
                        "Client failed before response decoding."
                    )
                result = decode_response(command, decode_json_response(line))
                if self._loop.time() >= deadline:
                    raise TimeoutError
                if isinstance(result, CommandError):
                    raise result.as_exception()
                if isinstance(result, (GoalStartResult, TacticSuccess)):
                    state_id = (
                        result.state_id
                        if isinstance(result, GoalStartResult)
                        else result.next_state_id
                    )
                    if state_id in self._allocated_ids:
                        raise PantographProtocolError(
                            "REPL reused a live allocation ID."
                        )
                    self._allocated_ids.add(state_id)
                elif command == "goal.delete" and isinstance(
                    result, CommandAcknowledgement
                ):
                    self._allocated_ids.difference_update(body["stateIds"])
                if record is not None:
                    record(result)
                if self._loop.time() >= deadline:
                    raise TimeoutError
                return result
        except TimeoutError as exc:
            error = PantographRequestTimeout(
                f"Deadline exceeded for {command}; {self._context()}", command=command
            )
            if wrote:
                self._fail(error)
                await self.aclose()
            raise error from exc
        except asyncio.CancelledError as exc:
            if wrote:
                self._fail(
                    PantographTransportError(
                        f"Transaction cancelled after possible write: {command}"
                    )
                )
                await self._close_after_error(exc)
            raise
        except PantographGoalRejected:
            raise
        except (
            PantographProtocolError,
            PantographStateError,
            PantographTransportError,
            OSError,
        ) as exc:
            if wrote:
                error = (
                    exc
                    if not isinstance(exc, OSError)
                    else PantographTransportError(
                        f"REPL pipe failed: {exc}; {self._context()}"
                    )
                )
                self._fail(error)
                await self.aclose()
                raise error
            raise
        except Exception as exc:
            if wrote:
                error = PantographProtocolError(
                    f"Transaction bookkeeping failed: {exc}", command=command
                )
                self._fail(error)
                await self.aclose()
                raise error from exc
            raise
        finally:
            if acquired:
                self._reply = None
                self._lock.release()
