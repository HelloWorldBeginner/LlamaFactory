"""
CP Debug Hook 核心实现
"""
import os
import re
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path

from .utils import all_gather_seq, tensor_stats, format_stats


@dataclass
class CPDebugConfig:
    """CP Debug 配置

    `record` 是面向用户的主开关（"forward" / "backward" / "both"），在
    `__post_init__` 中展开为 record_forward / record_backward / record_gradients
    三个布尔。环境变量会覆盖对应字段（见 README 环境变量表）。
    """
    enabled: bool = False
    max_steps: int = 3
    step_range: Optional[Union[Tuple[int, int], List[int]]] = None
    dump_dir: str = "./cp_debug_dumps"
    mode: str = "dump"  # "print" | "dump" | "both"
    record: str = "both"  # "forward" | "backward" | "both"

    # CP 相关
    cp_group: Any = None
    cp_group_name: str = "cp"

    # 序列维度
    expected_seq_len: int = None
    default_seq_gather_dim: int = 1

    # 功能
    print_weights: bool = True
    print_full_tensor: bool = False
    # print 模式下，每条输出同时追加到此文件（None 时 print/both 模式默认 {dump_dir}/print.log）
    print_file: Optional[str] = None

    # 以下三项由 `record` 在 __post_init__ 中派生，不要直接设置
    record_forward: bool = field(default=False, init=False)
    record_backward: bool = field(default=False, init=False)
    record_gradients: bool = field(default=False, init=False)

    # 前向 / 反向独立覆盖 mode（None 表示沿用 mode）
    forward_mode: Optional[str] = None
    backward_mode: Optional[str] = None

    # 过滤
    module_filter: Optional[str] = None

    # Step 控制
    auto_step: bool = True

    def __post_init__(self):
        env = os.environ

        # 总开关：CP_DEBUG=1 视为启用
        if env.get("CP_DEBUG", "0") == "1":
            self.enabled = True

        # 环境变量覆盖（仅在设置时生效）
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
        if "CP_DEBUG_PRINT_FILE" in env:
            self.print_file = env["CP_DEBUG_PRINT_FILE"]
        # print/both 模式下，若未显式指定 print_file，默认落到 {dump_dir}/print.log
        if self.print_file is None and self.mode in ("print", "both"):
            self.print_file = str(Path(self.dump_dir) / "print.log")

        # 校验枚举字段
        if self.mode not in ("print", "dump", "both"):
            raise ValueError(f"mode must be print/dump/both, got {self.mode!r}")
        if self.record not in ("forward", "backward", "both"):
            raise ValueError(f"record must be forward/backward/both, got {self.record!r}")

        # 由 record 派生布尔开关
        self.record_forward = self.record in ("forward", "both")
        self.record_backward = self.record in ("backward", "both")
        self.record_gradients = self.record in ("backward", "both")


class NoOpCPDebugManager:
    """未启用时返回的空实现，使调用方无需判空即可照常使用 manager 接口。"""

    def step(self):
        pass

    def get_step(self) -> int:
        return 0

    def set_in_backward(self, flag: bool):
        pass

    def collect_param_gradients(self):
        pass

    def cleanup(self):
        pass

    def flush(self, step: Optional[int] = None):
        pass


class CPDebugManager:
    """CP Debug 管理器"""

    def __init__(self, config: CPDebugConfig):
        self.config = config
        self._auto_step = 0
        self._manual_step = None
        self._handles = []
        self._printed_weights = set()
        # 每步独立的执行顺序缓存：{step: [records]}
        self._execution_order: dict = {}
        self._step_record_count: dict = {}  # 每步已写入的记录数，用于生成 per-step idx
        self._weights_recorded = False
        self._in_backward = False
        self._first_forward = True  # 自动 step 模式下，首次 forward 不递增
        self._model_ref = None
        self._print_file_warned = False

    def _print(self, msg: str) -> None:
        """打印到 stdout，同时追加到 config.print_file（若配置）。

        仅 rank0 调用（record() 已做 rank0 门控）。落盘失败不影响训练，告警一次后放弃。
        """
        print(msg, flush=True)
        pf = self.config.print_file
        if not pf:
            return
        try:
            Path(pf).parent.mkdir(parents=True, exist_ok=True)
            with open(pf, "a") as f:
                f.write(msg + "\n")
        except Exception as e:
            if not self._print_file_warned:
                self._print_file_warned = True
                print(f"[CP_DEBUG] write print_file {pf!r} failed: {e!r}", flush=True)
            self.config.print_file = None  # 放弃，后续只走 stdout

    def step(self):
        """手动推进 step"""
        self._manual_step = (self._manual_step or 0) + 1

    def get_step(self) -> int:
        """获取当前 step（手动优先）"""
        if self._manual_step is not None:
            return self._manual_step
        return self._auto_step

    def _root_pre_hook(self, module: nn.Module, inputs):
        """Root model pre-hook：自动递增 + 延迟记录权重

        自动 step 模式下，首次 forward 保持 step=0（与 backward 阶段的梯度收集
        对齐到同一 step），从第二次 forward 起在进入前递增。
        """
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
        """是否应该记录当前 step"""
        step = self.get_step()

        if self.config.step_range is not None:
            if isinstance(self.config.step_range, tuple):
                start, end = self.config.step_range
                return start <= step < end
            elif isinstance(self.config.step_range, list):
                return step in self.config.step_range

        return step < self.config.max_steps

    def get_cp_group(self) -> Optional[Any]:
        """获取 CP group"""
        if self.config.cp_group is not None:
            return self.config.cp_group

        try:
            from mindspeed_llm.fsdp2.distributed.parallel_state import ParallelState
            ps = ParallelState()
            return ps.get_group(self.config.cp_group_name)
        except Exception as e:
            # 调试期常见：未安装 mindspeed_llm 或尚未初始化 CP group
            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"[CP_DEBUG] get_cp_group fallback failed: {e!r}")
            return None

    def record(self, name: str, tensor: torch.Tensor, step: int, hook_type: str = "forward"):
        """记录 tensor 并追加执行顺序

        weight / param_grad 不参与序列维 all-gather（参数本身不被 CP 切分，
        强行 gather 既无意义又可能因 hidden_size 与 seq_len 巧合触发误判/死锁）。
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
                    self.config.default_seq_gather_dim
                )

            # all-gather 是集合通信，所有 rank 都必须参与；写盘只在 rank0
            if dist.is_initialized() and dist.get_rank() != 0:
                return

            idx = self._step_record_count.get(step, 0)
            self._execution_order.setdefault(step, []).append({
                "step": step,
                "name": name,
                "hook_type": hook_type,
                "shape": list(gathered.shape),
            })
            self._step_record_count[step] = idx + 1

            stats = tensor_stats(gathered)

            if hook_type in ("fwd_in", "fwd_out"):
                effective_mode = self.config.forward_mode or self.config.mode
            elif hook_type in ("bwd_in", "bwd_out", "param_grad"):
                effective_mode = self.config.backward_mode or self.config.mode
            else:
                effective_mode = self.config.mode

            if effective_mode in ("print", "both"):
                self._print(format_stats(name, stats, step))
                if self.config.print_full_tensor:
                    self._print(f"[STEP {step}] {name} tensor:\n{gathered.cpu()}")

            if effective_mode in ("dump", "both"):
                dump_path = Path(self.config.dump_dir) / f"step{step}"
                dump_path.mkdir(parents=True, exist_ok=True)
                torch.save(gathered.cpu(), dump_path / f"{name}.pt")
                # 增量追加执行顺序，避免每条记录重写整文件（O(n²) → O(n)）
                self._append_execution_order(step, idx, hook_type, name, gathered.shape)

    def _append_execution_order(self, step: int, idx: int, hook_type: str, name: str, shape):
        """增量追加一条执行顺序记录到 step 目录"""
        dump_path = Path(self.config.dump_dir) / f"step{step}"
        dump_path.mkdir(parents=True, exist_ok=True)
        order_file = dump_path / "execution_order.txt"

        # 文件不存在时先写表头（仅一次）
        if not order_file.exists():
            with open(order_file, "w") as f:
                f.write(f"# Execution order for step {step}\n")
                f.write("# Format: [index] hook_type | module_name | shape\n\n")

        with open(order_file, "a") as f:
            f.write(f"[{idx:4d}] {hook_type:8s} | {name:50s} | {list(shape)}\n")

    def flush(self, step: Optional[int] = None):
        """显式刷新某 step 的执行顺序汇总（含 Total 行）。

        step 为 None 时刷新所有已缓存 step。增量文件已含每条记录，
        此处仅补写 Total 汇总行，便于人读。
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
            with open(order_file, "r") as f:
                content = f.read()
            marker = "# Format: [index] hook_type | module_name | shape\n"
            if marker in content and "# Total records:" not in content:
                content = content.replace(
                    marker, marker + f"# Total records: {len(records)}\n"
                )
                with open(order_file, "w") as f:
                    f.write(content)

    def cleanup(self):
        """清理所有 hooks"""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def set_in_backward(self, flag: bool):
        """标记当前是否在 backward 阶段（控制 forward hook 行为）"""
        self._in_backward = flag

    def collect_param_gradients(self):
        """在 loss.backward() 之后收集参数梯度（从 param.grad 读取，不使用 backward hook）

        NPU 上 register_full_backward_hook 内任何 tensor 操作都会与异步反向算子冲突，
        因此改为 backward 结束后直接从 param.grad 读取。梯度不走 all-gather。
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

                    if hasattr(grad, 'full_tensor'):
                        grad = grad.full_tensor()

                    self.record(full_name, grad, step, hook_type="param_grad")

        # 梯度收集是一个 step 的收尾，刷新该 step 的执行顺序汇总
        self.flush(step)

    def _record_all_weights(self, model: nn.Module):
        """延迟记录所有权重（在第一次 forward 时调用，确保权重已物化）"""
        module_filter_re = None
        if self.config.module_filter:
            module_filter_re = re.compile(self.config.module_filter)

        for name, module in model.named_modules():
            if module_filter_re and not module_filter_re.search(name):
                continue
            _record_weights(self, name, module)


def register_cp_debug_hooks(
    model: nn.Module,
    config: CPDebugConfig
) -> Union[CPDebugManager, NoOpCPDebugManager]:
    """
    注册 CP Debug hooks

    Args:
        model: PyTorch 模型
        config: 配置

    Returns:
        CPDebugManager 实例；未启用时返回 NoOpCPDebugManager（所有方法空实现，
        调用方无需判空）。
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
        print(f"[CP_DEBUG] Registered {len(manager._handles)} hooks "
              f"(no backward hooks, use collect_param_gradients after backward)")
        if config.module_filter:
            print(f"[CP_DEBUG] Module filter: {config.module_filter}")

    return manager


def _make_forward_hook(manager: CPDebugManager, module_name: str) -> Callable:
    """创建 forward hook"""
    def hook_fn(module: nn.Module, inputs, output):
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
    """记录模块权重"""
    for param_name, param in module.named_parameters(recurse=False):
        if param is None:
            continue

        full_name = f"{module_name}.{param_name}" if module_name else param_name

        if full_name in manager._printed_weights:
            continue
        manager._printed_weights.add(full_name)

        if hasattr(param, 'full_tensor'):
            param = param.full_tensor()

        manager.record(full_name, param, 0, hook_type="weight")
