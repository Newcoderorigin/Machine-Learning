import json
import threading
import time
from pathlib import Path

import pytest

from sov_ai.training import TrainingPipeline, TrainingPipelineConfig


class _LinearModel:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def __call__(self, inputs):
        return [self.scale * float(x) for x in inputs]

    forward = __call__


class _SlowModel(_LinearModel):
    def __call__(self, inputs):
        time.sleep(0.01)
        return super().__call__(inputs)


def _load_config() -> TrainingPipelineConfig:
    config_path = Path(__file__).resolve().parents[1] / "config.json"
    data = json.loads(config_path.read_text())
    return TrainingPipelineConfig.from_dict(data["training_pipeline"])


def test_training_pipeline_config_from_json():
    config = _load_config()

    assert config.loss_function == "mae"
    assert config.optimizer == "sgd"
    assert config.gradient_clipping is True
    assert config.parallel_workers == 4
    assert config.device_ids == (0, 1)
    assert config.max_epochs == 5
    assert config.extra_metrics == ("accuracy", "f1")


def test_training_pipeline_sequential_with_hooks():
    config = TrainingPipelineConfig(
        loss_function="mae",
        optimizer="sgd",
        optimizer_params={"lr": 0.1, "momentum": 0.0},
        gradient_clipping=True,
        clip_value=0.2,
        mixed_precision=True,
        gradient_accumulation_steps=2,
        max_epochs=2,
        validation_interval=1,
        early_stopping=True,
        early_stopping_patience=3,
        parallel_workers=1,
        scheduler_type="step",
        scheduler_interval="epoch",
        scheduler_step_size=1,
        scheduler_gamma=0.9,
    )

    model = _LinearModel(scale=1.5)
    train_batches = [
        ([1.0, 2.0], [0.5, 1.0]),
        ([0.2, 0.3], [0.1, 0.1]),
        ([1.1, 1.2], [1.0, 1.2]),
        ([0.8, 0.9], [0.9, 1.1]),
    ]
    val_batches = [([0.5, 0.5], [0.6, 0.4])]

    events = []

    def hook(event, ctx):
        if event == "on_batch_end":
            events.append((ctx.epoch, ctx.batch_index, ctx.thread_name, ctx.loss))

    pipeline = TrainingPipeline(model, config, model_hooks=[hook])
    result = pipeline.run(lambda: train_batches, lambda: val_batches)

    assert len(result.history["train_loss"]) == config.max_epochs
    assert len(events) == len(train_batches) * config.max_epochs
    assert result.optimizer_steps == pytest.approx(len(train_batches) * config.max_epochs / config.gradient_accumulation_steps, rel=0.1)
    assert result.history["val_loss"]


class _SpyPipeline(TrainingPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parallel_invocations = 0

    def _run_epoch_parallel(self, batches, context):  # type: ignore[override]
        self.parallel_invocations += 1
        return super()._run_epoch_parallel(batches, context)


def test_training_pipeline_parallel_execution_invokes_parallel_path():
    config = TrainingPipelineConfig(
        parallel_workers=2,
        max_epochs=1,
        gradient_accumulation_steps=1,
        loss_function="mse",
    )

    model = _SlowModel(scale=2.0)
    batches = [([i, i + 1], [i * 2, (i + 1) * 2]) for i in range(4)]

    thread_names = set()
    thread_lock = threading.Lock()

    def hook(event, ctx):
        if event == "on_batch_end":
            with thread_lock:
                thread_names.add(ctx.thread_name)

    pipeline = _SpyPipeline(model, config, model_hooks=[hook])
    result = pipeline.run(lambda: batches)

    assert pipeline.parallel_invocations == 1
    assert any(name.startswith("ThreadPoolExecutor") for name in thread_names)
    assert result.optimizer_steps == len(batches)
