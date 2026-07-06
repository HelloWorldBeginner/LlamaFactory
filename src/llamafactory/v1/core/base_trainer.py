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

"""The definition of trainer.

Init Phase:

1. Init batch generator.
2. Init optimizer (deepspeed).
3. Shard model.
4. Init optimizer (fsdp).
5. Init lr scheduler.

Train Phase:
1. Train Loop

"""

from abc import abstractmethod

import torch
import torch.nn.functional as F

from ..accelerator.helper import ReduceOp
from ..accelerator.interface import Dim, DistributedInterface
from ..config import BatchingStrategy, TrainingArguments
from ..utils import logging
from ..utils.callbacks import (
    CallbackHandler,
    LoggingCallback,
    TrainerCallback,
    TrainerState,
)
from ..utils.helper import compute_valid_tokens
from ..utils.types import BatchInput, HFModel, ModelOutput, Tensor, TorchDataset
from .utils.batching import BatchGenerator
from .utils.checkpoint import TrainingCheckpointCoordinator
from .utils.rendering import Renderer


logger = logging.get_logger(__name__)


class BaseTrainer:
    def __init__(
        self,
        args: TrainingArguments,
        model: HFModel,
        renderer: Renderer,
        train_dataset: TorchDataset,
        callbacks: list[TrainerCallback] | None = None,
    ) -> None:
        self.args = args
        self.model = model
        self.renderer = renderer
        self.train_dataset = train_dataset

        # info
        self.global_step = 0

        # cached variables
        self.device = DistributedInterface().current_device
        self.dp_size = DistributedInterface().get_world_size(Dim.DP)
        self.cp_size = DistributedInterface().get_world_size(Dim.CP)
        self.model_input_names = self.renderer.processor.model_input_names

        self._create_batch_generator()
        # Calculate num_training_steps: max_steps takes priority if set
        if self.args.max_steps is not None and self.args.max_steps > 0:
            self.num_training_steps = self.args.max_steps
        else:
            self.num_training_steps = self.args.num_train_epochs * len(self.train_batch_generator)

        if self.args.save_epochs is not None:
            steps_per_epoch = len(self.train_batch_generator)
            self.args.save_steps = max(1, int(steps_per_epoch * self.args.save_epochs))

        if self.args.enable_activation_checkpointing:
            self.model.gradient_checkpointing_enable({"use_reentrant": False})

        self._deepspeed_engine = None
        dist_name = self.args.dist_config.name if self.args.dist_config is not None else None

        if dist_name == "deepspeed":
            from ..plugins.trainer_plugins.distributed.hub import DistributedPlugin

            self._deepspeed_engine = DistributedPlugin("deepspeed")(
                self.model,
                self.args.dist_config,
                num_micro_batch=self.train_batch_generator.num_micro_batch,
                micro_batch_size=self.args.micro_batch_size,
            )
            self._init_optimizer()
            self._init_lr_scheduler()
            self.model, self.optimizer, self.lr_scheduler = self._deepspeed_engine.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )
        else:
            # fsdp2 / DDP / no dist
            self._shard_model()
            self._init_optimizer()
            self._init_lr_scheduler()

        self._resume_epoch = 0
        self._checkpoint = TrainingCheckpointCoordinator(self)
        if self.args.resume_from_checkpoint:
            self._checkpoint.resume(self.args.resume_from_checkpoint)

        if self.args.save_ckpt_as_hf:
            logger.warning_rank0(
                "save_ckpt_as_hf is enabled. Intermediate checkpoints will be saved in Hugging Face format. "
                "Note that this will significantly increase memory consumption during saving."
            )

        # Callbacks
        self.callback_handler = CallbackHandler([LoggingCallback()], trainer=self)
        for cb in callbacks or []:
            self.callback_handler.add_callback(cb)

        # Callbacks: TrainerState tracks progress across the full run.
        self.state = TrainerState(
            num_training_steps=self.num_training_steps,
            global_step=self.global_step,
            epoch=self._resume_epoch,
        )
        # Keep callback state aligned with checkpoint-resumed trainer counters.
        self.state.global_step = self.global_step
        self.state.epoch = self._resume_epoch

        if self.args.dist_config is not None and self.args.dist_config.get("cp_size", 1) > 1:
            # qwen3.5 is not supported because of the different attention implementation, which will be supported in the future.
            if model.config.model_type == "qwen3_5":
                raise RuntimeError(
                    "Sequence parallel is not supported for qwen3.5 model due to its different attention implementation, which will be supported in the future."
                )
            from ..plugins.model_plugins.parallelization.sequence_parallel import SequenceParallelModelPlugin

            if model.config._attn_implementation != "flash_attention_2":
                raise ValueError(
                    "Sequence parallelism requires flash attention. Please set `flash_attn: flash_attention_2`."
                )

            SequenceParallelModelPlugin(self.args.dist_config.get("cp_mode", "ulysses"))(model, self.args.dist_config)

        # CP precision debug (cp-precision-debug skill). Env-gated: no-op unless CP_DEBUG=1.
        # Registered after model sharding and the CP plugin so the CP group is available
        # and hooks sit on the wrapped model. Weights are lazily recorded on first forward.
        self.cp_debug_manager = self._register_cp_debug_hooks(model)

    def _create_batch_generator(self) -> None:
        if (
            self.args.batching_strategy == BatchingStrategy.PADDING_FREE
            and getattr(self.model.config, "_attn_implementation", None) != "flash_attention_2"
        ):
            raise ValueError("`padding_free` requires `flash_attn: flash_attention_2`.")

        self.train_batch_generator = BatchGenerator(
            dataset=self.train_dataset,
            renderer=self.renderer,
            micro_batch_size=self.args.micro_batch_size,
            global_batch_size=self.args.global_batch_size,
            cutoff_len=self.args.cutoff_len,
            batching_workers=self.args.batching_workers,
            batching_strategy=self.args.batching_strategy,
            seed=self.args.seed,
        )

    def _shard_model(self) -> None:
        if self.args.dist_config is None:
            if DistributedInterface().get_world_size(Dim.DP) > 1:
                from torch.nn.parallel import DistributedDataParallel as DDP

                logger.warning_rank0(
                    "dist_config is None but distributed training is enabled; falling back to DistributedDataParallel."
                )
                device_ids = None if self.device.type == "cpu" else [self.device.index]
                self.model = DDP(self.model, device_ids=device_ids)
        else:
            from ..plugins.trainer_plugins.distributed.hub import DistributedPlugin

            self.model = DistributedPlugin(self.args.dist_config.name)(
                self.model,
                self.args.dist_config,
                bf16=self.args.bf16,
            )

    def _init_optimizer(self) -> None:
        """Init optimizer."""
        if self.args.optim_config is None:
            _trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(_trainable_params, lr=self.args.learning_rate)
        else:
            from ..plugins.trainer_plugins.optimizer import OptimizerPlugin

            self.optimizer = OptimizerPlugin(self.args.optim_config.name)(self.model, self.args.optim_config)

    def _init_lr_scheduler(self) -> None:
        """Init lr scheduler."""
        if self.args.lr_scheduler_config is None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: 1.0)
        else:
            from ..plugins.trainer_plugins.lr_scheduler import LRSchedulerPlugin

            self.lr_scheduler = LRSchedulerPlugin(self.args.lr_scheduler_config.name)(
                self.optimizer, self.num_training_steps, self.args.lr_scheduler_config
            )

    def _clip_grad_norm_fsdp2(self, parameters, max_norm: float) -> float:
        """Global grad-norm clipping that is consistent across world sizes.

        ``torch.nn.utils.clip_grad_norm_`` on FSDP2 DTensor params clips using a
        per-rank (local-shard) norm, so the clip coefficient differs by world_size
        (e.g. CP1 vs CP2 get different norms) and the updates diverge. Instead,
        compute the global L2 norm via per-param ``pow(2).sum()`` (DTensor.sum
        all-reduces Shard grads, no-op for Replicate) so it is identical across
        world sizes, then clip every grad with the same global coefficient.
        Falls back to ``clip_grad_norm_`` for non-DTensor (e.g. DDP) grads.
        """
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:  # pragma: no cover
            DTensor = None  # type: ignore[assignment]

        params_with_grad = [p for p in parameters if p.grad is not None]
        if not params_with_grad:
            return 0.0

        if DTensor is not None and isinstance(params_with_grad[0].grad, DTensor):
            # Global L2 norm across the full FSDP2 mesh. Use p.grad.pow(2).sum() per param:
            # DTensor.sum() all-reduces sharded (Shard) grads to a global scalar and is a no-op
            # for replicated (Replicate) grads, so the result is consistent across world sizes.
            # Do NOT use to_local() + manual all_reduce(SUM) -- that double-counts Replicate grads
            # (gives cp1 a 2x norm vs cp2) and makes clipping inconsistent.
            total_sq = torch.zeros((), device=self.device, dtype=torch.float32)
            for p in params_with_grad:
                total_sq = total_sq + p.grad.detach().float().pow(2).sum()
            if isinstance(total_sq, DTensor):
                total_sq = total_sq.full_tensor()
            grad_norm = float(total_sq.sqrt().item())
            clip_coef = max_norm / (grad_norm + 1e-6)
            if clip_coef < 1.0:
                for p in params_with_grad:
                    p.grad.mul_(clip_coef)
            return grad_norm
        else:
            # Non-FSDP2 (DDP / single): grads are replicated, clip_grad_norm_ is correct.
            return float(torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm).item())

    def compute_log_probs(self, model: HFModel, batch: BatchInput) -> Tensor:
        """Compute log probs.

        log_probs: Tensor of shape (batch_size, seq_len - 1)
        """
        batch_size, _ = batch["labels"].shape
        model_inputs = {
            k: v.to(self.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        labels = batch["labels"].to(self.device, non_blocking=True)
        outputs: ModelOutput = model(**model_inputs)
        logits = outputs.logits.float()
        shift_labels = labels[..., 1:].contiguous().view(-1)
        shift_logits = logits[..., :-1, :].contiguous().view(shift_labels.size(0), -1)
        return -F.cross_entropy(shift_logits, shift_labels, reduction="none").view(batch_size, -1)

    @abstractmethod
    def compute_loss(self, batch: BatchInput) -> Tensor:
        """Compute the scalar loss."""
        ...

    def _register_cp_debug_hooks(self, model: HFModel):
        """Register CP precision-debug hooks (no-op unless CP_DEBUG=1).

        Compares CP1 (cp_size=1) vs CP2 (cp_size>1) per-layer tensors. The CP
        group comes from the distributed interface; expected_seq_len defaults to
        cutoff_len (override via CP_DEBUG_SEQ_LEN). When enabled, micro-batches
        are padded to cutoff_len so CP1/CP2 sequence shapes align for all-gather.
        See ``cp_debug/`` and the skill references for the full workflow.
        """
        import os

        if os.environ.get("CP_DEBUG", "0") != "1":
            from .cp_debug import NoOpCPDebugManager

            return NoOpCPDebugManager()

        from .cp_debug import CPDebugConfig, register_cp_debug_hooks

        # CP1 (cp_size=1) may not have a CP dim in the device mesh; fall back to
        # None — all_gather then no-ops and the local (full-seq) tensor is recorded.
        try:
            cp_group = DistributedInterface().get_group(Dim.CP)
        except (KeyError, ValueError):
            cp_group = None
        expected_seq_len = int(os.environ.get("CP_DEBUG_SEQ_LEN", str(self.args.cutoff_len)))
        config = CPDebugConfig(
            enabled=True,
            mode="dump",
            record="both",
            cp_group=cp_group,
            expected_seq_len=expected_seq_len,
            max_steps=int(os.environ.get("CP_DEBUG_MAX_STEPS", "1")),
            dump_dir=os.environ.get("CP_DEBUG_DUMP_DIR", "./cp_debug_dumps"),
            module_filter=os.environ.get("CP_DEBUG_MODULE_FILTER"),
        )
        manager = register_cp_debug_hooks(model, config)
        logger.info_rank0(
            f"[CP_DEBUG] hooks registered: cp_group={'set' if cp_group is not None else 'None'}, "
            f"expected_seq_len={expected_seq_len}, max_steps={config.max_steps}"
        )
        return manager

    def fit(self) -> None:
        """Train the model."""
        self.model.train()
        self.callback_handler.on_train_begin(self.args, self.state)

        epoch = self._resume_epoch
        while self.global_step < self.num_training_steps:
            self.state.epoch = epoch
            self.train_batch_generator.set_epoch(epoch)
            self.callback_handler.on_epoch_begin(self.args, self.state)

            # BatchGenerator is an iterator; each loop step calls its __next__ to produce one optimizer step.
            for micro_batches in self.train_batch_generator:
                self.global_step += 1

                self.state.global_step = self.global_step
                self.callback_handler.on_step_begin(self.args, self.state)

                step_loss = 0
                step_valid_tokens = compute_valid_tokens(micro_batches)
                step_valid_tokens = DistributedInterface().all_reduce(step_valid_tokens, op=ReduceOp.SUM, dim=Dim.ALL)
                num_micro = len(micro_batches)
                for i, micro_batch in enumerate(micro_batches):
                    if self.args.dist_config and self.args.dist_config.get("cp_size", 1) > 1:
                        from ..plugins.model_plugins.parallelization.sequence_parallel import (
                            SequenceParallelLossPlugin,
                        )

                        loss = SequenceParallelLossPlugin("sequence_parallel_loss")(self.model, micro_batch)
                    else:
                        loss = self.compute_loss(micro_batch)
                    raw_loss = loss.item()
                    mini_step_valid_tokens = compute_valid_tokens([micro_batch])
                    # Scale by world_size (= dp_size * cp_size), not dp_size. FSDP2 mean-reduces
                    # gradients over the full shard mesh (world_size), and the SP loss is already a
                    # global full-sequence mean (CP all-gathers log_probs). With *dp_size, CP2's
                    # loss_scaling = (seq/2)*dp/(dp*seq) = 0.5 (its mini is seq/2 because it holds
                    # 1/cp of the sequence), so its gradient is half CP1's -> grad_norm 2x off and
                    # loss diverges. *world_size makes CP2's loss_scaling = (seq/2)*(dp*cp)/(dp*seq)
                    # = 1, matching CP1. No-op for cp_size=1 (world_size == dp_size).
                    loss = loss * mini_step_valid_tokens * (self.dp_size * self.cp_size) / (step_valid_tokens + 1e-6)

                    if self._deepspeed_engine is not None:
                        # deepspeed: set sync_gradients so engine.step() only fires on last micro-batch
                        self._deepspeed_engine.accelerator.sync_gradients = i == num_micro - 1
                        self.cp_debug_manager.set_in_backward(True)
                        self._deepspeed_engine.backward(loss)
                        self.cp_debug_manager.set_in_backward(False)
                    else:
                        self.cp_debug_manager.set_in_backward(True)
                        loss.backward()
                        self.cp_debug_manager.set_in_backward(False)
                    # Read param.grad after backward (NPU-safe; no full_backward_hook).
                    self.cp_debug_manager.collect_param_gradients()
                    step_loss += raw_loss

                if self._deepspeed_engine is not None:
                    # deepspeed: engine.step() already ran inside backward at the sync boundary
                    grad_norm = self._deepspeed_engine.get_grad_norm()
                else:
                    grad_norm = self._clip_grad_norm_fsdp2(self.model.parameters(), self.args.max_grad_norm)

                    if not torch.isfinite(torch.tensor(grad_norm)):  # type: ignore # pyright: ignore [reportUnknownReturnType]
                        logger.warning_rank0(f"Gradient norm is not finite: {grad_norm}")
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                step_loss, grad_norm = DistributedInterface().all_reduce([step_loss, grad_norm])
                DistributedInterface().sync()

                # Update state with step metrics
                current_lr = (
                    self.lr_scheduler.get_last_lr()[0]
                    if hasattr(self.lr_scheduler, "get_last_lr")
                    else self.args.learning_rate
                )
                self.state.loss = step_loss
                self.state.grad_norm = grad_norm
                self.state.learning_rate = current_lr

                self.callback_handler.on_step_end(self.args, self.state)

                # Logging: trainer decides when to log
                if self.global_step % self.args.logging_steps == 0:
                    logs = {
                        "epoch": epoch,
                        "step": self.state.global_step,
                        "loss": step_loss,
                        "grad_norm": grad_norm,
                        "learning_rate": current_lr,
                    }
                    self.callback_handler.on_log(self.args, self.state, logs)

                if self.args.save_steps and self.global_step % self.args.save_steps == 0:
                    self._checkpoint.save(epoch)

                # Check if max_steps is reached
                if self.global_step >= self.num_training_steps:
                    logger.info_rank0(f"Reached max_steps ({self.num_training_steps}), stopping training.")
                    self.callback_handler.on_epoch_end(self.args, self.state)
                    self.callback_handler.on_train_end(self.args, self.state)
                    return

            self.callback_handler.on_epoch_end(self.args, self.state)
            epoch += 1

        self.callback_handler.on_train_end(self.args, self.state)

    def save_model(self) -> None:
        """Save the model."""
        if self.args.dist_config is not None and self.args.dist_config.name in ("deepspeed", "fsdp2"):
            from ..plugins.trainer_plugins.distributed.hub import DistributedPlugin

            DistributedPlugin(self.args.dist_config.name).save_model(
                self.model, self.args.output_dir, self.renderer.processor
            )
        else:
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            model_to_save.save_pretrained(
                self.args.output_dir, state_dict=model_to_save.state_dict(), max_shard_size="4GB"
            )
            self.renderer.processor.save_pretrained(self.args.output_dir, max_shard_size="4GB")
            logger.info_rank0(f"Model saved to {self.args.output_dir}")

        self.callback_handler.on_save(self.args, self.state)
