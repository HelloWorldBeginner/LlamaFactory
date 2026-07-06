# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored from the cp-precision-debug skill (MIT) for CP1/CP2 precision debugging.

"""CP Debug hook core implementation."""

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from .utils import all_gather_seq, format_stats, tensor_stats


@dataclass
class CPDebugConfig:
    """CP Debug configuration.

    `record` is the user-facing master switch ("forward" / "backward" / "both"),
    expanded in `__post_init__` into record_forward / record_backward /
    record_gradients booleans. Environment variables override the corresponding
    fields (see README env var table).
    """

    enabled: bool = False
    max_steps: int = 3
    step_range: Optional[Union[tuple[int, int], list[int]]] = None
    dump_dir: str = "./cp_debug_dumps"
    mode: str = "dump"  # "print" | "dump" | "both"
    record: str = "both"  # "forward" | "backward" | "both"

    # CP related
    cp_group: Any = None
    cp_group_name: str = "cp"

    # Sequence dimension
    expected_seq_len: int = None
    default_seq_gather_dim: int = 1

    # Features
    print_weights: bool = True
    print_full_tensor: bool = False

    # Derived from `record` in __post_init__; do not set directly
    record_forward: bool = field(default=False, init=False)
    record_backward: bool = field(default=False, init=False)
    record_gradients: bool = field(default=False, init=False)

    # Forward / backward independent mode override (None means follow `mode`)
    forward_mode: Optional[str] = None
    backward_mode: Optional[str] = None

    # Filtering
    module_filter: Optional[str] = None

    # Step control
    auto_step: bool = True

    def __post_init__(self):
        env = os.environ

        # Master switch: CP_DEBUG=1 means enabled
        if env.get("CP_DEBUG", "0") == "1":
            self.enabled = True

        # Environment variable overrides (only when set)
        if "CP_DEBUG_MODE" in env:
            self.mode = env["CP_DEBUG_MODE"]
        if "CP_DEBUG_DUMP_DIR" in env:
            self.dump_dir = env["CP_DEBUG_DUMP_DIR"]
        if "CP_DEBUG_MAX_STEPS" in env:
            self.max_steps = int(env["CP_DEBUG_MAX_STEPS"])
        if "CP_DEBUG_SEQ_LEN" in env:
            self.expected_seq_len = int(env["CP_DEBUG_SEQ_LEN"])
        if "CP_DEBUG_MODULE_FILTER" in env:
            self.module_filter = env["CP_DEBUG_MODULE_FILTER"]
        if "CP_DEBUG_RECORD" in env:
            self.record = env["CP_DEBUG_RECORD"]

        # Validate enum fields
        if self.mode not in ("print", "dump", "both"):
            raise ValueError(f"mode must be print/dump/both, got {self.mode!r}")
        if self.record not in ("forward", "backward", "both"):
            raise ValueError(f"record must be forward/backward/both, got {self.record!r}")

        # Derive boolean switches from record
        self.record_forward = self.record in ("forward", "both")
        self.record_backward = self.record in ("backward", "both")
        self.record_gradients = self.record in ("backward", "both")


class NoOpCPDebugManager:
    """No-op implementation returned when disabled, so callers need no null checks."""

    def step(self):
        pass

    def get_step(self) -> int:
        return 0

    def set_in_backward(self, flag: bool):
        del flag
        pass

    def collect_param_gradients(self):
        pass

    def cleanup(self):
        pass

    def flush(self, step: Optional[int] = None):
        del step
        pass


class CPDebugManager:
    """CP Debug manager."""

    def __init__(self, config: CPDebugConfig):
        self.config = config
        self._auto_step = 0
        self._manual_step = None
        self._handles = []
        self._printed_weights = set()
        # Per-step independent execution order cache: {step: [records]}
        self._execution_order: dict = {}
        self._step_record_count: dict = {}  # records written per step, for per-step idx
        self._weights_recorded = False
        self._in_backward = False
        self._first_forward = True  # in auto_step mode, first forward does not increment
        self._model_ref = None

    def step(self):
        """Manually advance step."""
        self._manual_step = (self._manual_step or 0) + 1

    def get_step(self) -> int:
        """Get current step (manual takes priority)."""
        if self._manual_step is not None:
            return self._manual_step
        return self._auto_step

    def _root_pre_hook(self, module: nn.Module, inputs):
        """Root model pre-hook: auto-increment + lazy weight recording.

        In auto_step mode, the first forward keeps step=0 (aligned with the
        gradient collection in the backward phase on the same step); from the
        second forward onward it increments before entering.
        """
        del inputs
        if self._in_backward:
            return
        if self.config.auto_step:
            if not self._first_forward:
                self._auto_step += 1
            self._first_forward = False

        if self.config.print_weights and not self._weights_recorded:
            self._model_ref = module
            self._record_all_weights(module)
            self._weights_recorded = True

    def should_record(self) -> bool:
        """Whether the current step should be recorded."""
        step = self.get_step()

        if self.config.step_range is not None:
            if isinstance(self.config.step_range, tuple):
                start, end = self.config.step_range
                return start <= step < end
            elif isinstance(self.config.step_range, list):
                return step in self.config.step_range

        return step < self.config.max_steps

    def get_cp_group(self) -> Optional[Any]:
        """Get CP group."""
        if self.config.cp_group is not None:
            return self.config.cp_group

        try:
            from mindspeed_llm.fsdp2.distributed.parallel_state import ParallelState

            ps = ParallelState()
            return ps.get_group(self.config.cp_group_name)
        except Exception as e:
            # Common during debugging: mindspeed_llm not installed or CP group not initialized
            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"[CP_DEBUG] get_cp_group fallback failed: {e!r}")
            return None

    def record(self, name: str, tensor: torch.Tensor, step: int, hook_type: str = "forward"):
        """Record a tensor and append to execution order.

        weight / param_grad do NOT participate in sequence-dim all-gather (params
        are not CP-sharded; forcing gather is meaningless and may trigger false
        positives or deadlock when hidden_size coincidentally equals seq_len).
        """
        with torch.no_grad():
            tensor = tensor.detach()

            if hook_type in ("weight", "param_grad"):
                gathered = tensor
            else:
                cp_group = self.get_cp_group()
                gathered = all_gather_seq(
                    tensor,
                    cp_group,
                    self.config.expected_seq_len,
                    name,
                    self.config.default_seq_gather_dim,
                )

            # all-gather is collective, all ranks must participate; only rank0 writes
            if dist.is_initialized() and dist.get_rank() != 0:
                return

            idx = self._step_record_count.get(step, 0)
            self._execution_order.setdefault(step, []).append(
                {
                    "step": step,
                    "name": name,
                    "hook_type": hook_type,
                    "shape": list(gathered.shape),
                }
            )
            self._step_record_count[step] = idx + 1

            stats = tensor_stats(gathered)

            if hook_type in ("fwd_in", "fwd_out"):
                effective_mode = self.config.forward_mode or self.config.mode
            elif hook_type in ("bwd_in", "bwd_out", "param_grad"):
                effective_mode = self.config.backward_mode or self.config.mode
            else:
                effective_mode = self.config.mode

            if effective_mode in ("print", "both"):
                print(format_stats(name, stats, step))
                if self.config.print_full_tensor:
                    print(f"[STEP {step}] {name} tensor:\n{gathered.cpu()}")

            if effective_mode in ("dump", "both"):
                dump_path = Path(self.config.dump_dir) / f"step{step}"
                dump_path.mkdir(parents=True, exist_ok=True)
                torch.save(gathered.cpu(), dump_path / f"{name}.pt")
                # Incremental append to avoid rewriting the whole file per record (O(n^2) -> O(n))
                self._append_execution_order(step, idx, hook_type, name, gathered.shape)

    def _append_execution_order(self, step: int, idx: int, hook_type: str, name: str, shape):
        """Incrementally append one execution order record to the step directory."""
        dump_path = Path(self.config.dump_dir) / f"step{step}"
        dump_path.mkdir(parents=True, exist_ok=True)
        order_file = dump_path / "execution_order.txt"

        # Write header once when file does not exist
        if not order_file.exists():
            with open(order_file, "w") as f:
                f.write(f"# Execution order for step {step}\n")
                f.write("# Format: [index] hook_type | module_name | shape\n\n")

        with open(order_file, "a") as f:
            f.write(f"[{idx:4d}] {hook_type:8s} | {name:50s} | {list(shape)}\n")

    def flush(self, step: Optional[int] = None):
        """Explicitly flush a step's execution order summary (with Total line).

        When step is None, flush all cached steps. The incremental file already
        contains every record; here we only append the Total summary line for readability.
        """
        steps = list(self._execution_order.keys()) if step is None else [step]
        for s in steps:
            records = self._execution_order.get(s, [])
            if not records:
                continue
            dump_path = Path(self.config.dump_dir) / f"step{s}"
            order_file = dump_path / "execution_order.txt"
            if not order_file.exists():
                continue
            with open(order_file) as f:
                content = f.read()
            marker = "# Format: [index] hook_type | module_name | shape\n"
            if marker in content and "# Total records:" not in content:
                content = content.replace(marker, marker + f"# Total records: {len(records)}\n")
                with open(order_file, "w") as f:
                    f.write(content)

    def cleanup(self):
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def set_in_backward(self, flag: bool):
        """Mark whether currently in the backward phase (controls forward hook behavior)."""
        self._in_backward = flag

    def collect_param_gradients(self):
        """Collect parameter gradients after loss.backward() (read from param.grad, no backward hook).

        On NPU any tensor op inside register_full_backward_hook conflicts with the
        async backward kernel, so we read param.grad directly after backward.
        Gradients do NOT go through all-gather.
        """
        if not self.config.record_gradients or not self.config.record_backward:
            return
        if not self.should_record():
            return

        model = self._model_ref
        if model is None:
            return

        step = self.get_step()
        module_filter_re = None
        if self.config.module_filter:
            module_filter_re = re.compile(self.config.module_filter)

        with torch.no_grad():
            for name, module in model.named_modules():
                if module_filter_re and not module_filter_re.search(name):
                    continue

                for param_name, param in module.named_parameters(recurse=False):
                    if param is None or param.grad is None:
                        continue

                    full_name = f"{name}.{param_name}.grad" if name else f"{param_name}.grad"
                    grad = param.grad.detach()

                    if hasattr(grad, "full_tensor"):
                        grad = grad.full_tensor()

                    self.record(full_name, grad, step, hook_type="param_grad")

        # Gradient collection finalizes a step; flush its execution order summary
        self.flush(step)

    def _record_all_weights(self, model: nn.Module):
        """Lazily record all weights (called on first forward to ensure weights are materialized)."""
        module_filter_re = None
        if self.config.module_filter:
            module_filter_re = re.compile(self.config.module_filter)

        for name, module in model.named_modules():
            if module_filter_re and not module_filter_re.search(name):
                continue
            _record_weights(self, name, module)


def register_cp_debug_hooks(model: nn.Module, config: CPDebugConfig) -> Union[CPDebugManager, NoOpCPDebugManager]:
    """Register CP Debug hooks.

    Args:
        model: PyTorch model
        config: configuration

    Returns:
        A CPDebugManager; when disabled returns NoOpCPDebugManager (all methods
        no-op, so callers need no null checks).
    """
    if not config.enabled:
        return NoOpCPDebugManager()

    manager = CPDebugManager(config)
    manager._model_ref = model

    handle = model.register_forward_pre_hook(manager._root_pre_hook)
    manager._handles.append(handle)

    module_filter_re = None
    if config.module_filter:
        module_filter_re = re.compile(config.module_filter)

    for name, module in model.named_modules():
        if module_filter_re and not module_filter_re.search(name):
            continue

        if config.record_forward:
            fwd_hook = _make_forward_hook(manager, name)
            handle = module.register_forward_hook(fwd_hook)
            manager._handles.append(handle)

    if dist.is_initialized() and dist.get_rank() == 0:
        print(
            f"[CP_DEBUG] Registered {len(manager._handles)} hooks "
            f"(no backward hooks, use collect_param_gradients after backward)"
        )
        if config.module_filter:
            print(f"[CP_DEBUG] Module filter: {config.module_filter}")

    return manager


def _make_forward_hook(manager: CPDebugManager, module_name: str) -> Callable:
    """Create a forward hook."""

    def hook_fn(module: nn.Module, inputs, output):
        del module
        if not manager.should_record():
            return

        if manager._in_backward:
            return

        step = manager.get_step()

        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                manager.record(f"{module_name}.in{i}", inp, step, hook_type="fwd_in")

        if isinstance(output, torch.Tensor):
            manager.record(f"{module_name}.out0", output, step, hook_type="fwd_out")
        elif isinstance(output, (tuple, list)):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    manager.record(f"{module_name}.out{i}", out, step, hook_type="fwd_out")

    return hook_fn


def _record_weights(manager: CPDebugManager, module_name: str, module: nn.Module):
    """Record module weights."""
    for param_name, param in module.named_parameters(recurse=False):
        if param is None:
            continue

        full_name = f"{module_name}.{param_name}" if module_name else param_name

        if full_name in manager._printed_weights:
            continue
        manager._printed_weights.add(full_name)

        if hasattr(param, "full_tensor"):
            param = param.full_tensor()

        manager.record(full_name, param, 0, hook_type="weight")
