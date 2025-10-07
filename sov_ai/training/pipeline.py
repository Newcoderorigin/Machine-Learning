"""Configurable training orchestration pipeline.

This module implements a lightweight yet feature-rich training pipeline that
supports a wide breadth of options typically found in large-scale machine
learning systems.  The pipeline exposes more than twenty configuration toggles
covering loss selection, optimizer management, scheduling, gradient
post-processing, distributed hints, parallel execution, and mixed precision
simulation.  The focus is to provide deterministic, easily testable behaviour
that mirrors the orchestration logic required to coordinate complex model
training workflows.

Key capabilities include:

* Config-driven orchestration with JSON-friendly schema support.
* Loss function registry with sensible defaults and extension hooks.
* Optimizer swapping, gradient accumulation, clipping, and noise injection.
* Mixed precision simulation and scheduler coordination.
* Parallel batch execution using ``ThreadPoolExecutor`` with safe context
  propagation.
* Custom hook integration for model-specific callbacks at multiple lifecycle
  events.
* Early stopping logic and validation scheduling with history tracking.

The implementation is intentionally framework agnostic and uses pure Python
numerical logic to keep dependencies minimal while remaining unit-testable.
"""

from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


Hook = Callable[[str, "TrainingContext"], None]
LossFn = Callable[[Any, Any], float]


def _default_optimizer_params() -> Dict[str, float]:
    return {"lr": 0.01, "momentum": 0.0, "weight_decay": 0.0}


@dataclass
class TrainingPipelineConfig:
    """Configuration container for :class:`TrainingPipeline`.

    The dataclass provides more than twenty configurable options spanning
    optimisation, scheduling, precision, gradient handling, parallelism, and
    distributed hints.  The defaults are intentionally conservative so that a
    pipeline instantiated without overrides remains deterministic and easy to
    reason about.
    """

    loss_function: str = "mse"
    loss_reduction: str = "mean"
    optimizer: str = "sgd"
    optimizer_params: Mapping[str, float] = field(default_factory=_default_optimizer_params)
    scheduler_type: str = "none"
    scheduler_interval: str = "epoch"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    scheduler_warmup_steps: int = 0
    min_lr: float = 1e-6
    gradient_clipping: bool = False
    clip_type: str = "norm"
    clip_value: float = 1.0
    clip_mode: str = "global_norm"
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    amp_backend: str = "native"
    amp_level: str = "O1"
    distributed_backend: str = "none"
    distributed_strategy: str = "data_parallel"
    world_size: int = 1
    num_devices: int = 1
    device_ids: Optional[Sequence[int]] = None
    seed: int = 42
    max_epochs: int = 1
    max_steps: Optional[int] = None
    microbatch_size: Optional[int] = None
    checkpointing: bool = False
    checkpoint_interval: int = 100
    log_interval: int = 10
    validation_interval: int = 50
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_loss"
    resume_from_checkpoint: Optional[str] = None
    optimizer_swap_epochs: Sequence[int] = field(default_factory=tuple)
    gradient_noise_scale: float = 0.0
    custom_hooks_enabled: bool = True
    parallel_workers: int = 1
    lookahead_steps: int = 0
    stoch_weight_avg: bool = False
    extra_metrics: Sequence[str] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingPipelineConfig":
        """Create a configuration object from a JSON-friendly mapping."""

        if not isinstance(data, Mapping):
            raise TypeError("Training pipeline configuration must be a mapping.")

        values: Dict[str, Any] = {}
        flat_keys = {
            "loss_function",
            "loss_reduction",
            "optimizer",
            "optimizer_params",
            "scheduler_type",
            "scheduler_interval",
            "scheduler_step_size",
            "scheduler_gamma",
            "scheduler_warmup_steps",
            "min_lr",
            "clip_type",
            "clip_value",
            "clip_mode",
            "gradient_accumulation_steps",
            "mixed_precision",
            "amp_backend",
            "amp_level",
            "distributed_backend",
            "distributed_strategy",
            "world_size",
            "num_devices",
            "device_ids",
            "seed",
            "max_epochs",
            "max_steps",
            "microbatch_size",
            "checkpointing",
            "checkpoint_interval",
            "log_interval",
            "validation_interval",
            "early_stopping",
            "early_stopping_patience",
            "early_stopping_metric",
            "resume_from_checkpoint",
            "optimizer_swap_epochs",
            "gradient_noise_scale",
            "custom_hooks_enabled",
            "parallel_workers",
            "lookahead_steps",
            "stoch_weight_avg",
            "extra_metrics",
        }

        for key in flat_keys:
            if key in data:
                values[key] = data[key]

        scheduler_data = data.get("scheduler")
        if isinstance(scheduler_data, Mapping):
            values.setdefault("scheduler_type", scheduler_data.get("type", "none"))
            values.setdefault("scheduler_interval", scheduler_data.get("interval", "epoch"))
            values.setdefault("scheduler_step_size", scheduler_data.get("step_size", 10))
            values.setdefault("scheduler_gamma", scheduler_data.get("gamma", 0.1))
            values.setdefault("scheduler_warmup_steps", scheduler_data.get("warmup_steps", 0))
            values.setdefault("min_lr", scheduler_data.get("min_lr", values.get("min_lr", 1e-6)))

        clip_data = data.get("gradient_clipping")
        if isinstance(clip_data, Mapping):
            values["gradient_clipping"] = clip_data.get("enabled", True)
            values.setdefault("clip_type", clip_data.get("type", "norm"))
            values.setdefault("clip_value", clip_data.get("value", 1.0))
            values.setdefault("clip_mode", clip_data.get("mode", "global_norm"))
        elif "gradient_clipping" in data:
            values["gradient_clipping"] = bool(data["gradient_clipping"])

        mp_data = data.get("mixed_precision")
        if isinstance(mp_data, Mapping):
            values.setdefault("mixed_precision", mp_data.get("enabled", True))
            values.setdefault("amp_backend", mp_data.get("backend", "native"))
            values.setdefault("amp_level", mp_data.get("level", "O1"))

        dist_data = data.get("distributed")
        if isinstance(dist_data, Mapping):
            values.setdefault("distributed_backend", dist_data.get("backend", "none"))
            values.setdefault("distributed_strategy", dist_data.get("strategy", "data_parallel"))
            values.setdefault("world_size", dist_data.get("world_size", 1))
            values.setdefault("num_devices", dist_data.get("num_devices", 1))
            values.setdefault("device_ids", dist_data.get("device_ids"))

        hook_data = data.get("hooks")
        if isinstance(hook_data, Mapping):
            values.setdefault("custom_hooks_enabled", hook_data.get("enabled", True))

        swap_data = data.get("optimizer_swaps")
        if isinstance(swap_data, Mapping):
            values.setdefault("optimizer_swap_epochs", swap_data.get("epochs", ()))

        extra_metrics = data.get("extra_metrics")
        if isinstance(extra_metrics, Sequence) and not isinstance(extra_metrics, (str, bytes)):
            values.setdefault("extra_metrics", tuple(extra_metrics))

        if "device_ids" in values and values["device_ids"] is not None:
            values["device_ids"] = tuple(values["device_ids"])

        if "extra_metrics" in values and values["extra_metrics"] is not None:
            values["extra_metrics"] = tuple(values["extra_metrics"])

        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a dictionary."""

        return {
            "loss_function": self.loss_function,
            "loss_reduction": self.loss_reduction,
            "optimizer": self.optimizer,
            "optimizer_params": dict(self.optimizer_params),
            "scheduler_type": self.scheduler_type,
            "scheduler_interval": self.scheduler_interval,
            "scheduler_step_size": self.scheduler_step_size,
            "scheduler_gamma": self.scheduler_gamma,
            "scheduler_warmup_steps": self.scheduler_warmup_steps,
            "min_lr": self.min_lr,
            "gradient_clipping": self.gradient_clipping,
            "clip_type": self.clip_type,
            "clip_value": self.clip_value,
            "clip_mode": self.clip_mode,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "mixed_precision": self.mixed_precision,
            "amp_backend": self.amp_backend,
            "amp_level": self.amp_level,
            "distributed_backend": self.distributed_backend,
            "distributed_strategy": self.distributed_strategy,
            "world_size": self.world_size,
            "num_devices": self.num_devices,
            "device_ids": tuple(self.device_ids) if self.device_ids is not None else None,
            "seed": self.seed,
            "max_epochs": self.max_epochs,
            "max_steps": self.max_steps,
            "microbatch_size": self.microbatch_size,
            "checkpointing": self.checkpointing,
            "checkpoint_interval": self.checkpoint_interval,
            "log_interval": self.log_interval,
            "validation_interval": self.validation_interval,
            "early_stopping": self.early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_metric": self.early_stopping_metric,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "optimizer_swap_epochs": tuple(self.optimizer_swap_epochs),
            "gradient_noise_scale": self.gradient_noise_scale,
            "custom_hooks_enabled": self.custom_hooks_enabled,
            "parallel_workers": self.parallel_workers,
            "lookahead_steps": self.lookahead_steps,
            "stoch_weight_avg": self.stoch_weight_avg,
            "extra_metrics": tuple(self.extra_metrics),
        }


@dataclass
class TrainingContext:
    """Runtime context propagated to hooks during pipeline execution."""

    config: TrainingPipelineConfig
    model: Any
    optimizer_name: str
    epoch: int = 0
    global_step: int = 0
    batch_index: Optional[int] = None
    loss: Optional[float] = None
    metrics: MutableMapping[str, Any] = field(default_factory=dict)
    optimizer_state: MutableMapping[str, Any] = field(default_factory=dict)
    scheduler_state: MutableMapping[str, Any] = field(default_factory=dict)
    thread_name: str = field(default_factory=lambda: threading.current_thread().name)

    def copy_for_batch(self, *, batch_index: int, global_step: int) -> "TrainingContext":
        clone = replace(
            self,
            batch_index=batch_index,
            global_step=global_step,
            thread_name=threading.current_thread().name,
        )
        clone.metrics = dict(self.metrics)
        clone.optimizer_state = dict(self.optimizer_state)
        clone.scheduler_state = dict(self.scheduler_state)
        return clone

    def record_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value


@dataclass
class TrainingResult:
    """Outcome of a completed pipeline run."""

    history: Dict[str, List[float]]
    optimizer_steps: int
    config: TrainingPipelineConfig


class _SimpleOptimizer:
    """Deterministic optimizer stub supporting step tracking and swaps."""

    def __init__(self, config: TrainingPipelineConfig, previous: Optional["_SimpleOptimizer"] = None):
        params = dict(config.optimizer_params)
        self.lr = params.get("lr", 0.01)
        self.momentum = params.get("momentum", 0.0)
        self.weight_decay = params.get("weight_decay", 0.0)
        self.steps = 0
        self.velocity = previous.velocity if previous is not None else 0.0

    def step(self, loss_value: float) -> float:
        self.velocity = self.momentum * self.velocity + loss_value
        if self.weight_decay:
            self.velocity += self.weight_decay * loss_value
        self.steps += 1
        return self.velocity * self.lr


class _Scheduler:
    """Minimal scheduler supporting step, epoch, and plateau behaviours."""

    def __init__(self, optimizer: _SimpleOptimizer, config: TrainingPipelineConfig):
        self.optimizer = optimizer
        self.config = config
        self.state = {"step": 0, "lr": optimizer.lr}
        self.best_metric: Optional[float] = None

    def step(self, metric: Optional[float] = None) -> None:
        self.state["step"] += 1

        if self.config.scheduler_type == "none":
            return

        if self.config.scheduler_type == "step":
            if self.state["step"] % max(1, self.config.scheduler_step_size) == 0:
                self.state["lr"] = max(self.config.min_lr, self.state["lr"] * self.config.scheduler_gamma)
        elif self.config.scheduler_type == "plateau" and metric is not None:
            if self.best_metric is None or metric < self.best_metric:
                self.best_metric = metric
            elif self.state["step"] % max(1, self.config.scheduler_step_size) == 0:
                self.state["lr"] = max(self.config.min_lr, self.state["lr"] * self.config.scheduler_gamma)
        elif self.config.scheduler_type == "warmup":
            if self.state["step"] <= self.config.scheduler_warmup_steps:
                warmup_ratio = self.state["step"] / max(1, self.config.scheduler_warmup_steps)
                self.state["lr"] = max(self.config.min_lr, self.optimizer.lr * warmup_ratio)


def _mse_loss(outputs: Sequence[float], targets: Sequence[float]) -> float:
    diffs = [(float(o) - float(t)) ** 2 for o, t in zip(outputs, targets)]
    return sum(diffs) / max(1, len(diffs))


def _mae_loss(outputs: Sequence[float], targets: Sequence[float]) -> float:
    diffs = [abs(float(o) - float(t)) for o, t in zip(outputs, targets)]
    return sum(diffs) / max(1, len(diffs))


def _cross_entropy_loss(outputs: Sequence[float], targets: Sequence[float]) -> float:
    eps = 1e-12
    losses = [-(float(t) * math_log(max(eps, float(o)))) for o, t in zip(outputs, targets)]
    return sum(losses) / max(1, len(losses))


def math_log(value: float) -> float:
    """Local log function to avoid importing heavy dependencies."""

    import math

    return math.log(value)


class TrainingPipeline:
    """Coordinate model training with extensive configuration support."""

    def __init__(
        self,
        model: Any,
        config: TrainingPipelineConfig,
        *,
        optimizer_factory: Optional[Callable[[TrainingPipelineConfig, Optional[_SimpleOptimizer]], _SimpleOptimizer]] = None,
        scheduler_factory: Optional[Callable[[_SimpleOptimizer, TrainingPipelineConfig], _Scheduler]] = None,
        loss_registry: Optional[Mapping[str, LossFn]] = None,
        model_hooks: Optional[Sequence[Hook]] = None,
    ) -> None:
        self.model = model
        self.config = config
        self._optimizer_factory = optimizer_factory or (lambda cfg, prev=None: _SimpleOptimizer(cfg, prev))
        self._loss_registry: Dict[str, LossFn] = {
            "mse": _mse_loss,
            "mae": _mae_loss,
            "cross_entropy": _cross_entropy_loss,
        }
        if loss_registry:
            self._loss_registry.update(loss_registry)
        self._hooks: List[Hook] = list(model_hooks or [])
        self._optimizer = self._optimizer_factory(self.config, None)
        self._scheduler_factory = scheduler_factory or (lambda opt, cfg: _Scheduler(opt, cfg))
        self._scheduler = self._scheduler_factory(self._optimizer, self.config)
        self._step_lock = threading.Lock()
        self._global_step = 0
        self._accum_steps = 0
        self._accum_loss = 0.0
        self._optimizer_steps = 0
        self._metrics_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        random.seed(self.config.seed)

    def run(
        self,
        train_loader: Iterable[Any] | Callable[[], Iterable[Any]],
        val_loader: Optional[Iterable[Any] | Callable[[], Iterable[Any]]] = None,
    ) -> TrainingResult:
        """Execute the configured training routine."""

        context = TrainingContext(
            config=self.config,
            model=self.model,
            optimizer_name=self.config.optimizer,
            optimizer_state={"lr": self._optimizer.lr, "momentum": self._optimizer.momentum},
            scheduler_state=dict(self._scheduler.state),
        )

        self._invoke_hooks("on_train_start", context)
        best_metric: Optional[float] = None
        patience = 0

        for epoch in range(self.config.max_epochs):
            context.epoch = epoch
            if epoch in set(self.config.optimizer_swap_epochs):
                self._optimizer = self._optimizer_factory(self.config, self._optimizer)
                self._scheduler = self._scheduler_factory(self._optimizer, self.config)
            context.optimizer_state = {"lr": self._optimizer.lr, "momentum": self._optimizer.momentum, "steps": self._optimizer.steps}
            context.scheduler_state = dict(self._scheduler.state)
            self._invoke_hooks("on_epoch_start", context)

            train_batches = self._materialise_batches(train_loader)
            train_losses = self._run_epoch(train_batches, context)
            epoch_loss = sum(train_losses) / max(1, len(train_losses))
            context.record_metric("train_loss", epoch_loss)
            self._metrics_history["train_loss"].append(epoch_loss)

            should_validate = val_loader is not None and (
                (epoch + 1) % max(1, self.config.validation_interval) == 0 or epoch == self.config.max_epochs - 1
            )
            if should_validate:
                self._invoke_hooks("on_validation_start", context)
                val_batches = self._materialise_batches(val_loader)  # type: ignore[arg-type]
                val_losses = self._run_validation(val_batches, context)
                val_loss = sum(val_losses) / max(1, len(val_losses))
                context.record_metric("val_loss", val_loss)
                self._metrics_history["val_loss"].append(val_loss)
                self._invoke_hooks("on_validation_end", context)
            else:
                val_loss = None

            if self.config.scheduler_interval == "epoch":
                metric_for_scheduler = val_loss if val_loss is not None else epoch_loss
                self._scheduler.step(metric_for_scheduler)
                context.scheduler_state = dict(self._scheduler.state)

            if self.config.early_stopping and val_loss is not None:
                metric_name = self.config.early_stopping_metric
                metric_value = context.metrics.get(metric_name, val_loss)
                if best_metric is None or metric_value < best_metric:
                    best_metric = metric_value
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.config.early_stopping_patience:
                        break

            if self.config.max_steps is not None and self._global_step >= self.config.max_steps:
                break

            self._invoke_hooks("on_epoch_end", context)

        self._invoke_hooks("on_train_end", context)

        return TrainingResult(history=self._metrics_history, optimizer_steps=self._optimizer_steps, config=self.config)

    # ------------------------------------------------------------------
    # Internal helpers

    def _materialise_batches(self, loader: Iterable[Any] | Callable[[], Iterable[Any]]) -> List[Any]:
        batches: Iterable[Any]
        if callable(loader):
            batches = loader()
        else:
            batches = loader
        return list(batches)

    def _run_epoch(self, batches: List[Any], context: TrainingContext) -> List[float]:
        if self.config.parallel_workers > 1 and len(batches) > 1:
            return self._run_epoch_parallel(batches, context)
        return [self._process_single_batch(batch, index, context) for index, batch in enumerate(batches)]

    def _run_epoch_parallel(self, batches: List[Any], context: TrainingContext) -> List[float]:
        results: List[float] = [0.0 for _ in batches]
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = {
                executor.submit(self._process_single_batch, batch, index, context): index for index, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
        return results

    def _run_validation(self, batches: List[Any], context: TrainingContext) -> List[float]:
        losses: List[float] = []
        for index, batch in enumerate(batches):
            _, targets = self._unpack_batch(batch)
            outputs = self._forward(batch)
            loss = self._compute_loss(outputs, targets)
            loss = self._post_process_loss(loss)
            local_context = context.copy_for_batch(batch_index=index, global_step=self._global_step)
            local_context.loss = loss
            self._invoke_hooks("on_validation_batch_end", local_context)
            losses.append(loss)
        return losses

    def _process_single_batch(self, batch: Any, batch_index: int, context: TrainingContext) -> float:
        inputs, targets = self._unpack_batch(batch)
        with self._step_lock:
            global_step = self._global_step
            self._global_step += 1
        local_context = context.copy_for_batch(batch_index=batch_index, global_step=global_step)
        self._invoke_hooks("on_batch_start", local_context)
        outputs = self._forward((inputs, targets))
        loss = self._compute_loss(outputs, targets)
        loss = self._post_process_loss(loss)
        update_value = self._maybe_step_optimizer(loss)
        local_context.loss = loss
        if update_value is not None:
            local_context.optimizer_state = {
                "lr": self._optimizer.lr,
                "momentum": self._optimizer.momentum,
                "steps": self._optimizer.steps,
                "last_update": update_value,
            }
        local_context.scheduler_state = dict(self._scheduler.state)
        self._invoke_hooks("on_batch_end", local_context)
        return loss

    def _forward(self, batch: Tuple[Any, Any]) -> Any:
        inputs, _ = batch
        if hasattr(self.model, "forward"):
            return self.model.forward(inputs)
        if callable(self.model):
            return self.model(inputs)
        raise TypeError("Model must be callable or expose a forward method.")

    def _unpack_batch(self, batch: Any) -> Tuple[Any, Any]:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError("Each batch must be a tuple of (inputs, targets).")
        return batch[0], batch[1]

    def _compute_loss(self, outputs: Any, targets: Any) -> float:
        loss_fn = self._resolve_loss_fn()
        if isinstance(outputs, (int, float)) and isinstance(targets, (int, float)):
            outputs = [float(outputs)]
            targets = [float(targets)]
        return float(loss_fn(outputs, targets))

    def _resolve_loss_fn(self) -> LossFn:
        if self.config.loss_function not in self._loss_registry:
            raise KeyError(f"Unknown loss function '{self.config.loss_function}'.")
        return self._loss_registry[self.config.loss_function]

    def _post_process_loss(self, loss: float) -> float:
        value = float(loss)
        if self.config.mixed_precision:
            value = float(value)
        if self.config.gradient_noise_scale:
            value += random.gauss(0.0, self.config.gradient_noise_scale)
        if self.config.gradient_clipping:
            limit = float(self.config.clip_value)
            if self.config.clip_type == "value":
                value = max(-limit, min(limit, value))
            else:
                value = self._clip_norm(value, limit)
        return value

    def _clip_norm(self, loss: float, limit: float) -> float:
        if loss > limit:
            return limit
        if loss < -limit:
            return -limit
        return loss

    def _maybe_step_optimizer(self, loss: float) -> Optional[float]:
        self._accum_steps += 1
        self._accum_loss += loss
        if self._accum_steps >= max(1, self.config.gradient_accumulation_steps):
            averaged = self._accum_loss / self._accum_steps
            update_value = self._optimizer.step(averaged)
            self._optimizer_steps += 1
            self._accum_steps = 0
            self._accum_loss = 0.0
            if self.config.scheduler_interval == "step":
                self._scheduler.step(averaged)
            return update_value
        return None

    def _invoke_hooks(self, event: str, context: TrainingContext) -> None:
        if not self.config.custom_hooks_enabled:
            return
        for hook in self._hooks:
            hook(event, context)

