import asyncio
import os
import random
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from threading import Lock
from typing import Any, Protocol

import aiohttp
import requests
import torch.distributed as dist
import uvloop
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    WeightUpdateMeta,
    WeightUpdateRequests,
)
from areal.api.workflow_api import RolloutWorkflow
from areal.platforms import current_platform
from areal.utils import logging, name_resolve, names
from areal.utils.http import arequest_with_retry, get_default_connector
from areal.utils.launcher import wait_llm_server_addrs
from areal.utils.network import find_free_ports, gethostip
from areal.utils.perf_tracer import trace_perf

from .workflow_executor import WorkflowExecutor
from dataclasses import dataclass as _dc, field as _field

RID_CACHE_SIZE = 128

# Thread-local storage for aiohttp sessions
# Each thread gets its own session to ensure thread safety and event loop compatibility
_session_storage = threading.local()

@_dc
class _ServerLoadState:
    """Tracks estimated pending work per server for least-loaded routing.

    Thread-safety: all mutations happen in a single asyncio event loop
    with no await between read and write in choose_server(). No lock needed.
    """
    pending_tokens: dict[str, int] = _field(default_factory=dict)
    pending_requests: dict[str, int] = _field(default_factory=dict)

    def add(self, addr: str, estimated_tokens: int) -> None:
        self.pending_tokens[addr] = self.pending_tokens.get(addr, 0) + estimated_tokens
        self.pending_requests[addr] = self.pending_requests.get(addr, 0) + 1

    def remove(self, addr: str, estimated_tokens: int) -> None:
        self.pending_tokens[addr] = max(0, self.pending_tokens.get(addr, 0) - estimated_tokens)
        self.pending_requests[addr] = max(0, self.pending_requests.get(addr, 0) - 1)

    def least_loaded(self, addresses: list[str]) -> str:
        return min(addresses, key=lambda a: (
            self.pending_tokens.get(a, 0),
            self.pending_requests.get(a, 0),
        ))

    def reset(self) -> None:
        self.pending_tokens.clear()
        self.pending_requests.clear()


class RemoteInfBackendProtocol(Protocol):
    """Protocol defining backend-specific operations for remote inference engines.

    This protocol abstracts the differences between various remote inference servers
    (SGLang, vLLM, etc.) by defining a common interface for:
    - Building HTTP requests with backend-specific formats
    - Parsing backend-specific responses
    - Handling weight updates
    - Managing control flow (pause/resume)
    - Supporting optional features (LoRA)

    Implementations can raise NotImplementedError for unsupported features.
    """

    def build_generation_request(
        self, req: ModelRequest, with_lora: bool
    ) -> HttpRequest:
        """Build HTTP request for text generation.

        Parameters
        ----------
        req : ModelRequest
            The generation request containing input and parameters
        with_lora : bool
            Whether to specify a LoRA to use

        Returns
        -------
        HttpRequest
            The HTTP request with endpoint and payload
        """
        ...

    def parse_generation_response(
        self, response: dict[str, Any]
    ) -> HttpGenerationResult:
        """Parse generation response into standard format.

        Parameters
        ----------
        response : Dict[str, Any]
            The raw JSON response from the server

        Returns
        -------
        HttpGenerationResult
            Parsed result with tokens, logprobs, and stop reason
        """
        ...

    def build_disk_weight_update_requests(
        self, meta: WeightUpdateMeta, lora_initialized: bool
    ) -> WeightUpdateRequests:
        """Build requests for loading weights from disk.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing path and configuration
        lora_initialized : bool
            Whether LoRA has been initialized in the server.
            If so, we need to unload the previous LoRA before
            uploading a new one.

        Returns
        -------
        WeightUpdateRequests
            Collection of HTTP requests (may be multiple for LoRA workflows)
        """
        ...

    def build_distributed_weight_update_requests(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> WeightUpdateRequests:
        """Build requests for distributed weight update via NCCL/XCCL.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing communication group info
        param_specs : List[ParamSpec]
            Specifications for parameters to be updated

        Returns
        -------
        WeightUpdateRequests
            Collection of HTTP requests for distributed update
        """
        ...

    def build_init_weights_group_request(
        self, addr: str, server_idx: int, meta: WeightUpdateMeta
    ) -> HttpRequest:
        """Build request to initialize weight update xccl group.

        Parameters
        ----------
        addr : str
            Server address
        server_idx : int
            Index of this server in the server list
        meta : WeightUpdateMeta
            Metadata containing communication backend configuration

        Returns
        -------
        HttpRequest
            The HTTP request to initialize the group
        """
        ...

    def get_pause_request(self) -> HttpRequest:
        """Get request to pause generation.

        Returns
        -------
        HttpRequest
            The HTTP request to pause generation

        Raises
        ------
        NotImplementedError
            If pause is not supported by this backend
        """
        ...

    def get_resume_request(self) -> HttpRequest:
        """Get request to resume generation.

        Returns
        -------
        HttpRequest
            The HTTP request to resume generation

        Raises
        ------
        NotImplementedError
            If resume is not supported by this backend
        """
        ...

    def get_health_check_request(self) -> HttpRequest:
        """Get the health check request.

        Returns
        -------
        HttpRequest
            The HTTP request for health checks
        """
        ...

    def launch_server(self, server_args: dict[str, Any]) -> subprocess.Popen:
        """Launch inference server subprocess.

        Parameters
        ----------
        server_args : dict[str, Any]
            Server configuration arguments for build_cmd_from_args

        Returns
        -------
        subprocess.Popen
            The launched server process
        """
        ...


class RemoteInfEngine:
    """
    Base implementation for HTTP-based remote inference engines.

    This class provides common functionality for communicating with remote
    inference servers via HTTP REST APIs. Backend-specific behaviors are
    delegated to an injected RemoteInfBackendProtocol implementation.

    Uses composition pattern - instantiate directly with a backend rather
    than inheriting from this class.

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine
    backend : RemoteInfBackendProtocol
        Backend implementation providing server-specific behavior
    """

    def __init__(
        self, config: InferenceEngineConfig, backend: RemoteInfBackendProtocol
    ):
        self.config = config
        self.backend = backend

        self.rid_to_address = {}
        # Maintain the addresses for the recent 128 requests
        self.rid_queue = []
        self.addresses = []
        self.server_idx = 0

        # Work-aware routing state (used when schedule_policy == "least_loaded")
        self._server_load = _ServerLoadState()
        # rid -> charged token estimate (for accurate subtraction on completion/pause)
        self._rid_charged_tokens: dict[str, int] = {}


        # Debug: per-server routing counter for load imbalance analysis
        self._route_counter: dict[str, int] = {}
        self._route_total = 0
        self._pause_cycle = 0

        self.distributed_weight_update_initialized = False
        self._version = 0

        self.lock = Lock()

        self.lora_initialized = False

        self.workflow_executor: WorkflowExecutor
        self.local_server_processes: list[LocalInfServerInfo] = []

    def _get_or_create_session(self) -> aiohttp.ClientSession:
        """Get or create a ClientSession for the current thread/event loop.

        This method provides thread-local session storage to avoid creating
        a new session for every request while maintaining thread safety.
        Each thread gets its own isolated session that is bound to that
        thread's event loop.

        Returns
        -------
        aiohttp.ClientSession
            A session object for the current thread
        """
        if (
            not hasattr(_session_storage, "session")
            or _session_storage.session is None
            or _session_storage.session.closed
        ):
            _session_storage.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.config.request_timeout,
                    sock_connect=self.config.request_timeout,
                    connect=self.config.request_timeout,
                ),
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            )
        return _session_storage.session

    def _wait_for_server(self, address):
        """Wait for a server to become healthy."""
        base_url = f"http://{address}"
        tik = time.time()
        while time.time() - tik < self.config.setup_timeout:
            if self.check_health(base_url):
                return
            time.sleep(1)
        raise TimeoutError("server launch failed")

    def check_health(self, base_url):
        """Check if server is healthy."""
        try:
            health_req = self.backend.get_health_check_request()
            url = f"{base_url}{health_req.endpoint}"
            response = requests.request(
                health_req.method, url, json=health_req.payload, timeout=30
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def initialize(
        self,
        engine_id: str | None = None,
        addr: str | list[str] | None = None,
        train_data_parallel_size: int | None = None,
    ):
        """Initialize the engine by discovering and connecting to servers.

        Parameters
        ----------
        engine_id : Optional[str]
            Unique identifier for this engine instance
        addr : str | List[str] | None
            Server address(es) to connect to. If None, will auto-discover.
        train_data_parallel_size : int | None
            Data parallel size of the training engine
        """
        if engine_id is None:
            if dist.is_initialized():
                engine_id = str(dist.get_rank())
            else:
                engine_id = uuid.uuid4().hex
        self.engine_id = engine_id
        self.logger = logging.getLogger(f"[Remote Inference Engine Rank {engine_id}]")

        if addr:
            self.addresses = addr if isinstance(addr, list) else [addr]
            self.logger.info("Get server addresses from the `addr` argument.")
        else:
            if (
                self.config.experiment_name is not None
                and self.config.trial_name is not None
            ):
                try:
                    self.addresses = wait_llm_server_addrs(
                        experiment_name=self.config.experiment_name,
                        trial_name=self.config.trial_name,
                        timeout=1,
                    )
                    self.logger.info("Get server addresses from name_resolve.")
                except (TimeoutError, RuntimeError):
                    # RuntimeError happens when name_resolve is not properly configured.
                    pass
        if not self.addresses and os.getenv("AREAL_LLM_SERVER_ADDRS"):
            # When addr is not provided, fallback to reading addrs from env var
            self.addresses = os.environ["AREAL_LLM_SERVER_ADDRS"].split(",")
            self.logger.info("Get server addresses from environment variable.")
        if not self.addresses:
            raise RuntimeError(
                "No configured inference servers. "
                "Please pass in server addresses by arguments "
                "for `initialize` or environment "
                "variable `AREAL_LLM_SERVER_ADDRS`."
            )

        self.logger.info("Waiting for server ready...")
        for addr_ in self.addresses:
            self._wait_for_server(addr_)
        self.server_idx = random.randint(0, len(self.addresses) - 1)

        for addr in self.addresses:
            self._server_load.pending_tokens[addr] = 0
            self._server_load.pending_requests[addr] = 0

        self.logger.info("Servers are all ready!")
        self.executor = ProcessPoolExecutor(max_workers=1)

        self.workflow_executor = WorkflowExecutor(
            config=self.config,
            inference_engine=self,
        )
        self.workflow_executor.initialize(
            logger=self.logger, train_data_parallel_size=train_data_parallel_size
        )

        # Register session cleanup hook for AsyncTaskRunner thread
        # This ensures sessions created in the background thread are properly closed
        async def cleanup_session():
            """Close thread-local aiohttp session in AsyncTaskRunner thread."""
            if hasattr(_session_storage, "session") and _session_storage.session:
                if not _session_storage.session.closed:
                    await _session_storage.session.close()
                _session_storage.session = None

        self.workflow_executor.runner.register_shutdown_hook(cleanup_session)

    def destroy(self):
        """Destroy the engine and clean up resources."""
        # Clean up thread-local session if it exists in the current thread
        if hasattr(_session_storage, "session") and _session_storage.session:
            try:
                if not _session_storage.session.closed:
                    # Try to close the session synchronously
                    # Note: This only cleans up the session in the current thread.
                    # Sessions in AsyncTaskRunner thread are cleaned up via shutdown hooks
                    # registered during initialize().
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're in an async context, schedule the close
                        asyncio.create_task(_session_storage.session.close())
                    else:
                        # If no loop is running, run the close synchronously
                        loop.run_until_complete(_session_storage.session.close())
            except RuntimeError:
                # Ignore errors during cleanup (e.g., no event loop)
                pass
            finally:
                _session_storage.session = None

        if hasattr(self, "workflow_executor"):
            self.workflow_executor.destroy()
        if hasattr(self, "executor"):
            self.executor.shutdown()

    def set_version(self, version):
        """Set the current weight version."""
        with self.lock:
            self._version = version

    def get_version(self):
        """Get the current weight version."""
        with self.lock:
            return self._version

    def choose_server(self, estimated_tokens: int = 0) -> str:
        """Choose a server based on the scheduling policy.

        Returns
        -------
        str
            Selected server address

        Raises
        ------
        NotImplementedError
            If schedule policy other than round-robin is used
        """
        if self.config.schedule_policy == "round_robin":
            server = self.addresses[self.server_idx]
            self.server_idx = (self.server_idx + 1) % len(self.addresses)
            return server
        elif self.config.schedule_policy == "least_loaded":
            server = self._server_load.least_loaded(self.addresses)
            # Speculative charge BEFORE dispatch (Miles pattern: _use_url increments before HTTP)
            # Prevents N concurrent coroutines from all picking the same "empty" server
            self._server_load.add(server, estimated_tokens)
            return server
        raise ValueError(f"Unknown schedule_policy: {self.config.schedule_policy!r}")
    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request.

        Parameters
        ----------
        req : ModelRequest
            The model request containing input data and generation parameters

        Returns
        -------
        ModelResponse
            The generated response from the model
        """
        # Create a shallow copy of the input request
        # we are going to modify it in-place
        req = req.copy()

        # Validate n_samples
        gconfig = req.gconfig
        if gconfig.n_samples != 1:
            raise ValueError(
                "Inference engines do not support n_samples > 1. "
                "Please call generate multiple times with n_samples = 1."
            )

        # Validate max_new_tokens
        max_new_tokens = min(
            gconfig.max_tokens - len(req.input_ids), gconfig.max_new_tokens
        )
        if max_new_tokens <= 0:
            raise RuntimeError(
                f"max_new_tokens ({max_new_tokens}) is non-positive! "
                f"max_tokens={gconfig.max_tokens}, prompt_len={len(req.input_ids)}, "
                f"max_new_tokens={gconfig.max_new_tokens}."
            )

        # Update max_new_tokens in request
        req.gconfig.max_new_tokens = max_new_tokens

        # Make request
        start_time = time.perf_counter()
        accumulated_output_tokens = []
        accumulated_output_logprobs = []
        accumulated_versions = []
        estimated_tokens = req.gconfig.max_new_tokens

        # A single "rid" shares the same server to allow KV cache reuse
        if req.rid in self.rid_to_address:
            server_addr = self.rid_to_address[req.rid]
            if self.config.schedule_policy == "least_loaded":
                self._server_load.add(server_addr, estimated_tokens)
                self._rid_charged_tokens[req.rid] = estimated_tokens
            self.logger.info(f"[ROUTE] rid={req.rid} -> {server_addr} (CACHED)")
        else:
            server_addr = self.choose_server(estimated_tokens=estimated_tokens)
            if self.config.schedule_policy == "least_loaded":
                # choose_server already called add() for least_loaded
                self._rid_charged_tokens[req.rid] = estimated_tokens
            if len(self.rid_queue) >= RID_CACHE_SIZE:
                oldest_rid = self.rid_queue.pop(0)
                self.rid_to_address.pop(oldest_rid, None)
                self._rid_charged_tokens.pop(oldest_rid, None)
            self.rid_to_address[req.rid] = server_addr
            self.rid_queue.append(req.rid)
            self.logger.info(
                f"[ROUTE] rid={req.rid} -> {server_addr} (NEW) est_tokens={estimated_tokens}"
            )

        # Debug stats with load snapshot
        self._route_counter[server_addr] = self._route_counter.get(server_addr, 0) + 1
        self._route_total += 1
        if self._route_total % 50 == 0:
            load_snap = {a: self._server_load.pending_tokens.get(a, 0) for a in self.addresses}
            self.logger.info(
                f"[ROUTE_STATS] total={self._route_total} "
                f"per_server={dict(self._route_counter)} load={load_snap}"
            )

        # Get or create thread-local session
        # Thread-local storage ensures each thread has its own session,
        # maintaining thread safety and event loop compatibility
        session = self._get_or_create_session()

        # Deal with rollout interruption
        stop_reason = None
        _decode_iterations = 0
        while (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens) < gconfig.max_new_tokens
        ):
            # Request is interrupted, wait for some time to avoid interfering
            # with update weights requests
            _was_paused = False
            while self.workflow_executor.is_paused():
                _was_paused = True
                await asyncio.sleep(0.5)
            if _was_paused:
                # Release old load charge (server aborted this request during pause)
                if self.config.schedule_policy == "least_loaded":
                    old_charged = self._rid_charged_tokens.pop(req.rid, 0)
                    self._server_load.remove(server_addr, old_charged)

                # Re-route: KV cache is wiped, no benefit from stickiness
                new_estimated = req.gconfig.max_new_tokens  # shrunk by prior iterations
                server_addr = self.choose_server(estimated_tokens=new_estimated)
                if self.config.schedule_policy == "least_loaded":
                    self._rid_charged_tokens[req.rid] = new_estimated
                self.rid_to_address[req.rid] = server_addr

                self.logger.info(
                    f"[RESEND] rid={req.rid} server={server_addr} "
                    f"accumulated={len(accumulated_output_tokens)} tokens, "
                    f"prompt_len={len(req.input_ids)} "
                    f"est_remaining={new_estimated} "
                    f"iteration={_decode_iterations} "
                    f"pause_cycle={self._pause_cycle}"
                )

            _decode_iterations += 1

            # Build request using backend
            http_req = self.backend.build_generation_request(req, self.lora_initialized)

            _iter_start = time.perf_counter()

            # Loop until the generation is complete
            result = await arequest_with_retry(
                session=session,
                addr=server_addr,
                endpoint=http_req.endpoint,
                payload=http_req.payload,
                method=http_req.method,
                max_retries=self.config.request_retries,
                timeout=self.config.request_timeout,
            )

            # Parse response using backend
            gen_result = self.backend.parse_generation_response(result)
            stop_reason = gen_result.stop_reason

            # Log tokens per decode patch
            tokens_in_patch = len(gen_result.output_tokens)
            logger = getattr(self, "logger", logging.getLogger("RemoteInfEngine"))
            logger.info(f"Decode patch size: {tokens_in_patch} tokens")

            if _was_paused:
                self.logger.info(
                    f"[RESEND_DURATION] rid={req.rid} server={server_addr} "
                    f"prompt_len={len(req.input_ids)} "
                    f"new_tokens={tokens_in_patch} "
                    f"duration={time.perf_counter() - _iter_start:.2f}s "
                    f"pause_cycle={self._pause_cycle}"
                )

            # Update accumulated outputs
            accumulated_output_tokens.extend(gen_result.output_tokens)
            accumulated_output_logprobs.extend(gen_result.output_logprobs)
            accumulated_versions.extend(
                [self.get_version()] * len(gen_result.output_tokens)
            )

            # Update request for next iteration
            req.input_ids += gen_result.output_tokens
            req.gconfig.max_new_tokens -= len(gen_result.output_tokens)
            assert req.gconfig.max_new_tokens >= 0, (
                req.gconfig.max_new_tokens,
                len(gen_result.output_tokens),
                len(req.input_ids),
            )
        
        if self.config.schedule_policy == "least_loaded":
            charged = self._rid_charged_tokens.pop(req.rid, 0)
            self._server_load.remove(server_addr, charged)
        
        # Debug: log generation completion stats
        gen_duration = time.perf_counter() - start_time
        self.logger.info(
            f"[GEN_DONE] rid={req.rid} server={server_addr} "
            f"tokens={len(accumulated_output_tokens)} duration={gen_duration:.2f}s "
            f"iterations={_decode_iterations} stop={stop_reason}"
        )

        # Final abort handling
        if stop_reason == "abort":
            # If stop_reason is "abort", the only reason we exit the loop is
            # len(accumulated_output_tokens) >= gconfig.max_new_tokens
            # so the actual reason is length
            stop_reason = "length"

        latency = time.perf_counter() - start_time

        response = ModelResponse(
            input_tokens=req.input_ids[
                : len(req.input_ids) - len(accumulated_output_tokens)
            ],
            input_images=req.image_data,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
            tokenizer=req.tokenizer,
            processor=req.processor,
        )
        return response

    def init_weights_update_group(self, meta: WeightUpdateMeta) -> Future[None]:
        """Initialize the weight update process group for distributed weight updates.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update

        Returns
        -------
        Future[None]
            A future object representing the asynchronous initialization operation
        """
        assert meta.type == current_platform.communication_backend
        assert not self.distributed_weight_update_initialized

        fut = self.executor.submit(
            _init_weights_update_group_remote,
            self.backend,
            meta,
            self.addresses,
            self.config.request_timeout,
        )

        def callback(fut):
            self.logger.info(
                f"Initialized {current_platform.communication_backend.upper()} group "
                f"for distributed weight update for {meta.nccl_group_name}."
            )
            self.distributed_weight_update_initialized = True

        fut.add_done_callback(callback)

        return fut

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        """Update weights in the inference engine from distributed memory.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        param_specs : List[ParamSpec]
            A list of parameter specifications for the weights to be updated

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """
        assert meta.type == current_platform.communication_backend

        fut = self.executor.submit(
            _update_weights_from_distributed,
            self.backend,
            meta,
            param_specs,
            self.addresses,
            self.config.request_timeout,
        )

        return fut

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        """Update weights in the inference engine from disk.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """
        assert meta.type == "disk"

        tik = time.perf_counter()

        # Use ProcessPool to bypass python GIL for running async coroutines
        if self.config.experiment_name is None or self.config.trial_name is None:
            raise RuntimeError(
                "Experiment and trial names must be set for disk-based weight updates."
            )

        fut = self.executor.submit(
            _update_weights_from_disk,
            self.backend,
            self.lora_initialized,
            self.config.experiment_name,
            self.config.trial_name,
            self.get_version(),
            self.addresses,
            meta,
            self.config.request_retries,
            self.config.request_timeout,
        )

        def callback(fut):
            respond_time = fut.result()
            self.logger.info(
                f"Loading weights from disk done "
                f"in {(time.perf_counter() - tik):.2f}s. "
                f"Respond time: {respond_time:.2f}s."
            )
            # Update LoRA state if this was a LoRA update
            if meta.use_lora:
                self.lora_initialized = True
            if meta.clear_checkpoint_after_load:
                shutil.rmtree(meta.path, ignore_errors=True)

        fut.add_done_callback(callback)

        return fut

    def submit(
        self,
        data: dict[str, Any],
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
    ) -> None:
        """Submit a request to the inference engine and return immediately.

        Parameters
        ----------
        data : Dict[str, Any]
            The input data for rollout
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            The workflow to use for rollout generation
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str, optional
            A function or module path for trajectory filtering
        """
        assert workflow is not None, "Workflow must be specified for submit."
        return self.workflow_executor.submit(
            data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
        )

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> dict[str, Any]:
        """Wait for a specified number of requests to complete.

        Parameters
        ----------
        count : int
            The number of accepted trajectories to wait for
        timeout : float, optional
            Timeout in seconds
        raise_timeout : bool, optional
            Whether to raise a TimeoutError when the timeout is exceeded, by default True

        Returns
        -------
        Dict[str, Any]
            A concatenated batch of trajectories, or an empty dict if timeout exceeded and raise_timeout is False
        """
        return self.workflow_executor.wait(
            count, timeout=timeout, raise_timeout=raise_timeout
        )

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a batch of requests and wait for results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            A list of input data dictionaries for rollout
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            The workflow to use for rollout generation
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor

        Returns
        -------
        Dict[str, Any]
            A concatenated batch of trajectory results
        """
        assert workflow is not None, "Workflow must be specified for rollout_batch."
        return self.workflow_executor.rollout_batch(
            data=data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
    ):
        """Asynchronously submit and wait until a full batch is ready.

        Parameters
        ----------
        dataloader : StatefulDataLoader
            The data loader to pull data from
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            The workflow to use for rollout generation
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str, optional
            A function or module path for trajectory filtering

        Returns
        -------
        Dict[str, Any]
            A full batch of trajectory results
        """
        assert workflow is not None, "Workflow must be specified for prepare_batch."
        return self.workflow_executor.prepare_batch(
            dataloader=dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
        )

    def prepare_batch_cooperative(
        self,
        dataloader: StatefulDataLoader,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        dp_group,
        sync_interval: float = 2.0,
        max_local_factor: int = 2,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
    ):
        """Cooperatively prepare a batch across DP ranks.

        Fast DPs collect more results to compensate for slow DPs.
        See WorkflowExecutor.prepare_batch_cooperative for details.
        """
        assert workflow is not None, "Workflow must be specified."
        return self.workflow_executor.prepare_batch_cooperative(
            dataloader=dataloader,
            workflow=workflow,
            dp_group=dp_group,
            sync_interval=sync_interval,
            max_local_factor=max_local_factor,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
        )

    @trace_perf("remote_inf_engine.pause_generation", category="misc")
    def pause_generation(self):
        """Pause request submission for async rollout."""
        self._pause_cycle += 1
        try:
            pause_req = self.backend.get_pause_request()
            t_pause_start = time.perf_counter()
            for i, addr in enumerate(self.addresses):
                t0 = time.perf_counter()
                res = requests.post(
                    f"http://{addr}{pause_req.endpoint}",
                    json=pause_req.payload,
                )
                res.raise_for_status()
                self.logger.info(
                    f"[PAUSE] server[{i}]={addr} took {time.perf_counter()-t0:.4f}s"
                )
            self.logger.info(
                f"[PAUSE] cycle={self._pause_cycle} "
                f"total={time.perf_counter()-t_pause_start:.4f}s "
                f"grace_period={self.config.pause_grace_period}s"
            )
        except NotImplementedError:
            self.logger.warning("Backend does not support pause operation")

        # The above http request may require some time to be scheduled and executed.
        # The following line waits until all requests are indeed dropped.
        time.sleep(self.config.pause_grace_period)

    @trace_perf("remote_inf_engine.continue_generation", category="misc")
    def continue_generation(self):
        """Resume request submission for async rollout."""
        # After pause, KV caches wiped. Clear sticky routing so coroutines
        # get fresh routing when they wake from asyncio.sleep(0.5).
        old_count = len(self.rid_to_address)
        self.rid_to_address.clear()
        self.rid_queue.clear()
        if self.config.schedule_policy == "least_loaded":
            self._server_load.reset()
            self._rid_charged_tokens.clear()
        self.logger.info(f"[RESUME] Cleared {old_count} rid routes, reset load state")

        try:            
            resume_req = self.backend.get_resume_request()
            t_resume_start = time.perf_counter()
            for i, addr in enumerate(self.addresses):
                t0 = time.perf_counter()
                res = requests.post(
                    f"http://{addr}{resume_req.endpoint}",
                    json=resume_req.payload,
                )
                res.raise_for_status()
                self.logger.info(
                    f"[RESUME] server[{i}]={addr} took {time.perf_counter()-t0:.4f}s"
                )
            self.logger.info(
                f"[RESUME] cycle={self._pause_cycle} "
                f"total={time.perf_counter()-t_resume_start:.4f}s"
            )
        except NotImplementedError:
            self.logger.warning("Backend does not support resume operation")

        # Poll queue stats after resume to capture thundering-herd buildup
        _poll_logger = self.logger
        _poll_cycle = self._pause_cycle
        _poll_addrs = list(self.addresses)
        def _poll_queue_background():
            try:
                for poll_idx in range(15):  # 15 polls x 2s = 30s window
                    for i, addr in enumerate(_poll_addrs):
                        try:
                            resp = requests.get(
                                f"http://{addr}/areal_queue_stats",
                                timeout=2.0,
                            )
                            if resp.status_code == 200:
                                stats = resp.json()
                                _poll_logger.info(
                                    f"[QUEUE_SNAP] server[{i}]={addr} "
                                    f"running={stats.get('running', -1)} "
                                    f"waiting={stats.get('waiting', -1)} "
                                    f"cycle={_poll_cycle} "
                                    f"poll={poll_idx}"
                                )
                        except Exception:
                            pass
                    time.sleep(2.0)
            except Exception:
                pass

        t = threading.Thread(target=_poll_queue_background, daemon=True)
        t.start()

    def pause(self):
        """Pause request submission for async rollout.
        Used during evaluation to prevent data over generation.
        """
        return self.workflow_executor.pause()

    def resume(self):
        """Resume request submission for async rollout."""
        return self.workflow_executor.resume()

    # def recompute_kv_cache(self) -> Future[None]:
    #     """Recompute KV cache in background."""
    #     fut = self.executor.submit(
    #         _recompute_kv_cache_remote,
    #         self.backend,
    #         self.addresses,
    #         self.config.pause_grace_period,
    #         self.config.request_timeout,
    #     )

    #     def callback(fut):
    #         try:
    #             fut.result()
    #             self.logger.info("Background KV cache recomputation finished")
    #         except Exception as e:
    #             self.logger.error(f"Background KV cache recomputation failed: {e}")

    #     fut.add_done_callback(callback)
    #     return fut

    def launch_server(self, server_args: dict[str, Any]) -> LocalInfServerInfo:
        """Launch a local inference server."""
        server_args["host"] = gethostip()
        server_args["port"] = find_free_ports(1)[0]
        process = self.backend.launch_server(server_args)
        address = f"{server_args['host']}:{server_args['port']}"
        server_info = LocalInfServerInfo(
            host=server_args["host"],
            port=server_args["port"],
            process=process,
        )
        try:
            self._wait_for_server(address)
            self.local_server_processes.append(server_info)
            return server_info
        except TimeoutError:
            self._shutdown_one_server(server_info)
            raise

    def _shutdown_one_server(self, server_info: LocalInfServerInfo):
        if server_info.process.poll() is not None:
            return
        server_info.process.terminate()
        try:
            server_info.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.logger.warning(
                f"Server process {server_info.process.pid} did not terminate gracefully. Killing it."
            )
            server_info.process.kill()
            server_info.process.wait()

    def teardown_server(self):
        """Teardown all locally launched servers."""
        for server_info in self.local_server_processes:
            self._shutdown_one_server(server_info)
        self.local_server_processes.clear()


# Helper functions that run in ProcessPoolExecutor


def _update_weights_from_disk(
    backend: RemoteInfBackendProtocol,
    lora_initialized: bool,
    experiment_name: str,
    trial_name: str,
    model_version: int,
    addresses: list[str],
    meta: WeightUpdateMeta,
    request_retries: int,
    request_timeout: float,
):
    """Helper to update weights from disk in a separate process."""

    async def _fn():
        update_name = names.update_weights_from_disk(
            experiment_name, trial_name, model_version
        )
        save_timestamp = float(name_resolve.wait(update_name, timeout=120))
        load_timestamp = datetime.now().timestamp()

        # Get requests from backend
        weight_reqs = backend.build_disk_weight_update_requests(meta, lora_initialized)

        # Execute all requests
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        ) as session:
            for http_req in weight_reqs.requests:
                jobs = [
                    arequest_with_retry(
                        session=session,
                        addr=addr,
                        endpoint=http_req.endpoint,
                        payload=http_req.payload,
                        method=http_req.method,
                        max_retries=request_retries,
                        timeout=request_timeout,
                    )
                    for addr in addresses
                ]
                await asyncio.gather(*jobs)

        return load_timestamp - save_timestamp

    return uvloop.run(_fn())


def _init_weights_update_group_remote(
    backend: RemoteInfBackendProtocol,
    meta: WeightUpdateMeta,
    addresses: list[str],
    request_timeout: float,
):
    """Helper to initialize weight update group in a separate process."""

    async def _fn():
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        ) as session:
            jobs = []
            for i, addr in enumerate(addresses):
                http_req = backend.build_init_weights_group_request(addr, i, meta)
                jobs.append(
                    arequest_with_retry(
                        session=session,
                        addr=addr,
                        endpoint=http_req.endpoint,
                        payload=http_req.payload,
                        method=http_req.method,
                        max_retries=1,
                        timeout=request_timeout,
                    )
                )
            await asyncio.gather(*jobs)

    return uvloop.run(_fn())


def _update_weights_from_distributed(
    backend: RemoteInfBackendProtocol,
    meta: WeightUpdateMeta,
    param_specs: list[ParamSpec],
    addresses: list[str],
    request_timeout: float,
):
    """Helper to update weights from distributed memory in a separate process."""

    async def _fn():
        # Get requests from backend
        weight_reqs = backend.build_distributed_weight_update_requests(
            meta, param_specs
        )

        # Execute all requests sequentially (they may have dependencies)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        ) as session:
            for http_req in weight_reqs.requests:
                jobs = [
                    arequest_with_retry(
                        session=session,
                        addr=addr,
                        endpoint=http_req.endpoint,
                        payload=http_req.payload,
                        method=http_req.method,
                        max_retries=1,
                        timeout=request_timeout,
                    )
                    for addr in addresses
                ]
                await asyncio.gather(*jobs)

    return uvloop.run(_fn())


# def _recompute_kv_cache_remote(
#     backend: RemoteInfBackendProtocol,
#     addresses: list[str],
#     pause_grace_period: float,
#     request_timeout: float,
# ):
#     """Helper to recompute KV cache in a separate process."""
#     try:
#         pause_req = backend.get_pause_request()
#         for addr in addresses:
#             requests.post(
#                 f"http://{addr}{pause_req.endpoint}",
#                 json=pause_req.payload,
#                 timeout=request_timeout,
#             ).raise_for_status()
#     except NotImplementedError:
#         pass
#     except Exception:
#         pass

#     time.sleep(pause_grace_period)

#     try:
#         resume_req = backend.get_resume_request()
#         for addr in addresses:
#             requests.post(
#                 f"http://{addr}{resume_req.endpoint}",
#                 json=resume_req.payload,
#                 timeout=request_timeout,
#             ).raise_for_status()
#     except NotImplementedError:
#         pass
#     except Exception:
#         pass
