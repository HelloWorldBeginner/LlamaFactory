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
    # DP 相关（用于多 rank dump 按 dp_rank 分目录；不传则按 global_rank//cp_size 推断）
    dp_group: Any = None

    # 序列维度
    expected_seq_len: int = None
    default_seq_gather_dim: int = 1

    # 功能
    print_weights: bool = True
    # 每个被记录的 step 都 dump 一份当前权重（看训练后漂移）。量大可关：CP_DEBUG_WEIGHTS_PER_STEP=0
    dump_weights_per_step: bool = True
    print_full_tensor: bool = False
    # print 模式下，每条输出同时追加到此文件（None 时 print/both 模式默认 {dump_dir}/print.log）
    print_file: Optional[str] = None
    # 原始本地打印：不 pad、不 all-gather、不算统计，直接把每个 rank 的本地 tensor 原样打印；
    # 每个 cp_rank 各写一个日志文件 {dump_dir}/dp_rank{D}_cp_rank{C}_{ts}.log。
    # 选 dp_rank 用 CP_DEBUG_DP_RANK；CP2 下该 dp_rank 的 rank0/rank1 各一个文件，CP1 只 cp_rank0。
    raw_print: bool = False
    # 动态 seq 长度检测（LlamaFactory 专属）：每步从 root forward 的 input_ids([bs,seqlen])
    # 读 seqlen，动态更新 expected_seq_len = local_seq * cp_size，使变长下 all-gather 仍能拼回全长。
    # 需配合 CP_DEBUG_PAD_TO_CUTOFF=0（不 pad 到 cutoff）。env CP_DEBUG_AUTO_SEQ_LEN=1。
    auto_seq_len: bool = False

    # 以下三项由 `record` 在 __post_init__ 中派生，不要直接设置
    record_forward: bool = field(default=False, init=False)
    record_backward: bool = field(default=False, init=False)
    record_gradients: bool = field(default=False, init=False)

    # 前向 / 反向独立覆盖 mode（None 表示沿用 mode）
    forward_mode: Optional[str] = None
    backward_mode: Optional[str] = None

    # 过滤
    module_filter: Optional[str] = None

    # 写盘 rank 过滤：None=所有 dp_rank 都写（每个 CP 组由 cp_rank 0 落盘一份）；
    # 设为列表则只写指定 dp_rank，如 [0,1]。env CP_DEBUG_DP_RANK 逗号分隔。
    debug_dp_ranks: Optional[List[int]] = None

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
        if "CP_DEBUG_DP_RANK" in env:
            raw = env["CP_DEBUG_DP_RANK"].strip()
            self.debug_dp_ranks = [int(x) for x in raw.split(",") if x.strip() != ""] if raw else None
        if "CP_DEBUG_STEPS" in env:
            raw = env["CP_DEBUG_STEPS"].strip()
            if raw and "-" in raw and "," not in raw:
                # "10-15" → 半开区间 [10,16)，记录 step 10..15
                a, b = raw.split("-", 1)
                self.step_range = (int(a), int(b) + 1)
            elif raw:
                self.step_range = [int(x) for x in raw.split(",") if x.strip() != ""]
        if "CP_DEBUG_RECORD" in env:
            self.record = env["CP_DEBUG_RECORD"]
        if "CP_DEBUG_PRINT_FILE" in env:
            self.print_file = env["CP_DEBUG_PRINT_FILE"]
        if env.get("CP_DEBUG_RAW_PRINT", "0") == "1":
            self.raw_print = True
            # raw_print：不 pad、不 gather。强制关掉 pad_to_cutoff（pad_and_truncate 读此 env）。
            env["CP_DEBUG_PAD_TO_CUTOFF"] = "0"
        if env.get("CP_DEBUG_AUTO_SEQ_LEN", "0") == "1":
            self.auto_seq_len = True
        if env.get("CP_DEBUG_WEIGHTS_PER_STEP", "1") == "0":
            self.dump_weights_per_step = False

        # raw_print 模式：每个 cp_rank 各写一个日志文件，文件名带 dp_rank/cp_rank。
        # 非 raw_print：print/both 模式默认单文件 {dump_dir}/cp{cp_size}_{ts}.log。
        if self.print_file is None:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                if self.cp_group is not None and dist.is_initialized():
                    cp_size = dist.get_world_size(self.cp_group)
                    cp_rank = dist.get_rank(self.cp_group)
                else:
                    cp_size, cp_rank = 1, 0
                if self.dp_group is not None and dist.is_initialized():
                    dp_rank = dist.get_rank(self.dp_group)
                else:
                    dp_rank = 0
            except Exception:
                cp_size, cp_rank, dp_rank = 1, 0, 0
            if self.raw_print:
                self.print_file = str(Path(self.dump_dir) / f"dp_rank{dp_rank}_cp_rank{cp_rank}_{ts}.log")
            elif self.mode in ("print", "both"):
                self.print_file = str(Path(self.dump_dir) / f"cp{cp_size}_{ts}.log")

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

    def _root_pre_hook(self, module: nn.Module, inputs, kwargs: Optional[dict] = None):
        """Root model pre-hook：自动递增 + 延迟记录权重 + 捕获 forward kwargs

        自动 step 模式下，首次 forward 保持 step=0（与 backward 阶段的梯度收集
        对齐到同一 step），从第二次 forward 起在进入前递增。

        额外捕获 forward kwargs（input_ids / attention_mask / position_ids）——
        HF 把这些当 kwargs 传，位置参数 hook 抓不到，不捕获就无法对比这俩 CP
        关键输入。需配合注册处 `with_kwargs=True`。
        """
        if self._in_backward:
            return
        if self.config.auto_step:
            # 1-indexed：首次 forward = step 1，与训练日志 global_step 对齐。
            self._auto_step += 1

        # 动态 seq 长度检测（LlamaFactory 专属）：每步从 input_ids([bs,seqlen]) 读 seqlen，
        # 更新 expected_seq_len = local_seq * cp_size，使变长下 all-gather 仍能拼回全长。
        # 必须在子模块 forward hook 调 record() 之前完成。
        if self.config.auto_seq_len and kwargs:
            ii = kwargs.get("input_ids")
            if isinstance(ii, torch.Tensor) and ii.ndim >= 2:
                self.config.expected_seq_len = int(ii.shape[-1]) * self._cp_world()

        # 原始/init 权重（step 0）：首次 forward 时记录一次（此时还未 optimizer.step，即初始权重）。
        # 仅当 CP_DEBUG_STEPS 显式含 0（如 CP_DEBUG_STEPS=0 或 0,1）才记；不含 0 则不记，避免多余的 step0。
        if self.config.print_weights and not self._weights_recorded and not self.config.raw_print:
            if self.config.step_range is not None and self.should_record(0):
                self._model_ref = module
                self._record_all_weights(module, step=0)
            self._weights_recorded = True
        # 当前 step 权重：每个被记录的 step 都记一份，看训练后权重漂移
        if self.config.dump_weights_per_step and not self.config.raw_print and self.should_record():
            self._record_all_weights(module, step=self.get_step())

        # 捕获 forward kwargs（HF 当 kwargs 传，位置 hook 抓不到；None 安全）
        if kwargs and self.should_record():
            step = self.get_step()
            for key in ("input_ids", "attention_mask", "position_ids"):
                self._record_root_kwarg(key, kwargs.get(key), step)

    def _comm_device(self):
        """all-gather 用的设备：优先模型参数所在设备，回退 CPU。"""
        if self._model_ref is not None:
            try:
                return next(self._model_ref.parameters()).device
            except (StopIteration, RuntimeError):
                pass
        return torch.device("cpu")

    def _cp_rank(self) -> int:
        if self.config.cp_group is not None and dist.is_initialized():
            return dist.get_rank(self.config.cp_group)
        return 0

    def _cp_world(self) -> int:
        if self.config.cp_group is not None and dist.is_initialized():
            return dist.get_world_size(self.config.cp_group)
        return 1

    def _dp_rank(self) -> int:
        if self.config.dp_group is not None and dist.is_initialized():
            return dist.get_rank(self.config.dp_group)
        if dist.is_initialized():
            # 回退：假设 dp-outer mesh，global = dp*cp + cp
            return dist.get_rank() // max(self._cp_world(), 1)
        return 0

    def _should_write(self) -> bool:
        """每个 CP 组由 cp_rank 0 落盘一份（按 dp_rank 分目录）；debug_dp_ranks 过滤 dp_rank。"""
        if self._cp_rank() != 0:
            return False
        if self.config.debug_dp_ranks is None:
            return True
        return self._dp_rank() in self.config.debug_dp_ranks

    def _should_write_raw(self) -> bool:
        """raw_print：选定 dp_rank 的所有 cp_rank 都写（CP2 下 rank0/rank1 各一份）。"""
        if self.config.debug_dp_ranks is None:
            return True
        return self._dp_rank() in self.config.debug_dp_ranks

    def _dump_dir(self, step: int) -> Path:
        """按 dp_rank 分目录：{dump_dir}/dp_rank{D}/step{S}/"""
        return Path(self.config.dump_dir) / f"dp_rank{self._dp_rank()}" / f"step{step}"

    def _record_root_kwarg(self, key: str, val, step: int) -> None:
        """记录 root forward kwarg，None 安全。

        attention_mask / position_ids 可能为 None（全 1 被 HF 折叠、无 padding 路径）。
        若 CP 组内某些 rank 是 None、某些是 tensor，朴素 all-gather 会死锁（有人等没人）。
        先跨 CP 同步 None 性：
        - 全 None → 写哨兵 tensor([0])，rank0 落盘，compare 看到两端一致（shape [1], 0）。
        - 全 tensor → 正常 record()（内含 all-gather，CP2 分片拼回全长）。
        - 混合 → 跳过 + 告警，避免死锁。
        """
        is_tensor = isinstance(val, torch.Tensor)
        # raw_print：直接原样打印本地值（None 也打印出来），不 gather、不分哨兵。
        if self.config.raw_print:
            if not self._should_write_raw():
                return
            if is_tensor:
                self._print(f"[STEP {step}] model.{key} shape={list(val.shape)}:\n{val.cpu()}")
            else:
                self._print(f"[STEP {step}] model.{key} = {val!r}")
            return

        cp_group = self.get_cp_group()
        if cp_group is not None and dist.is_initialized():
            sp = dist.get_world_size(cp_group)
            device = val.device if is_tensor else self._comm_device()
            flag = torch.tensor([1 if is_tensor else 0], dtype=torch.int64, device=device)
            gathered = [torch.empty_like(flag) for _ in range(sp)]
            dist.all_gather(gathered, flag, group=cp_group)
            n_tensor = sum(int(x.item()) for x in gathered)
        else:
            sp = 1
            n_tensor = 1 if is_tensor else 0

        all_none = (n_tensor == 0)
        all_tensor = (n_tensor == sp)

        if all_none:
            # 两端都 None：写哨兵，让 compare 能确认一致（不走 all-gather）
            if not self._should_write():
                return
            dump_path = self._dump_dir(step)
            dump_path.mkdir(parents=True, exist_ok=True)
            torch.save(torch.tensor([0]), dump_path / f"model.{key}.pt")
            return

        if not all_tensor:
            # 混合 None/tensor → 跳过避免死锁
            if self._should_write():
                print(
                    f"[CP_DEBUG] root kwarg '{key}' mixed None/tensor across CP ranks; "
                    f"skipping to avoid all-gather deadlock",
                    flush=True,
                )
            return

        # 全 tensor → 正常记录（record 内 all-gather）
        self.record(f"model.{key}", val, step, hook_type="fwd_in")

    def should_record(self) -> bool:
        """是否应该记录当前 step"""
        step = self.get_step()

        if self.config.step_range is not None:
            if isinstance(self.config.step_range, tuple):
                start, end = self.config.step_range
                return start <= step < end
            elif isinstance(self.config.step_range, list):
                return step in self.config.step_range

        # max_steps 是"记录前 N 步"的计数；step 1-indexed → 记录 step 1..max_steps
        return step <= self.config.max_steps

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

            # raw_print：不 gather、不算统计、不落 .pt，直接把本地 tensor 原样打印；
            # 选定 dp_rank 的所有 cp_rank 各写各的文件。权重/梯度跳过（太大）。
            if self.config.raw_print:
                if hook_type in ("weight", "param_grad"):
                    return
                if not self._should_write_raw():
                    return
                self._print(f"[STEP {step}] {name} ({hook_type}) shape={list(tensor.shape)}:\n{tensor.cpu()}")
                return

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

            # all-gather 是集合通信，所有 rank 都必须参与；写盘只在 cp_rank 0（每 CP 组一份）
            if not self._should_write():
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
                dump_path = self._dump_dir(step)
                dump_path.mkdir(parents=True, exist_ok=True)
                torch.save(gathered.cpu(), dump_path / f"{name}.pt")
                # 增量追加执行顺序，避免每条记录重写整文件（O(n²) → O(n)）
                self._append_execution_order(step, idx, hook_type, name, gathered.shape)

    def _append_execution_order(self, step: int, idx: int, hook_type: str, name: str, shape):
        """增量追加一条执行顺序记录到 step 目录"""
        dump_path = self._dump_dir(step)
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

    def _record_all_weights(self, model: nn.Module, step: int):
        """记录所有权重到 step{step}/（每次调用都全量记录，不跨调用去重）。"""
        module_filter_re = None
        if self.config.module_filter:
            module_filter_re = re.compile(self.config.module_filter)

        for name, module in model.named_modules():
            if module_filter_re and not module_filter_re.search(name):
                continue
            _record_weights(self, name, module, step)


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

    handle = model.register_forward_pre_hook(manager._root_pre_hook, with_kwargs=True)
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
        # 校验 CP_DEBUG_DP_RANK 取值范围（dp_rank，非全局 rank）
        if config.debug_dp_ranks is not None:
            cp_world = dist.get_world_size(config.cp_group) if config.cp_group else 1
            dp_world = dist.get_world_size() // cp_world
            invalid = [r for r in config.debug_dp_ranks if r < 0 or r >= dp_world]
            if invalid:
                print(
                    f"[CP_DEBUG] WARNING: CP_DEBUG_DP_RANK {invalid} out of range "
                    f"(valid dp_rank: 0..{dp_world - 1}; dp_size={dp_world}). "
                    f"These dp_ranks will be silently skipped. "
                    f"CP_DEBUG_DP_RANK takes DP_RANK values, not global ranks.",
                    flush=True,
                )

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


def _record_weights(manager: CPDebugManager, module_name: str, module: nn.Module, step: int):
    """记录模块权重到 step{step}/"""
    for param_name, param in module.named_parameters(recurse=False):
        if param is None:
            continue

        full_name = f"{module_name}.{param_name}" if module_name else param_name

        if hasattr(param, 'full_tensor'):
            param = param.full_tensor()

        manager.record(full_name, param, step, hook_type="weight")
