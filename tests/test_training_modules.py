import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sov_ai.training import (
    DynamicLossAdaptation,
    GradientResonanceOptimization,
    NeuralSymbolicReplayBuffer,
    RecursiveSelfAuditMechanism,
    SymbolicFeedbackIntegration,
)


def test_dynamic_loss_adaptation_balances_losses():
    adaptor = DynamicLossAdaptation(("loss_a", "loss_b"))
    combined, weights = adaptor.update({"loss_a": 2.0, "loss_b": 1.0})
    assert isinstance(combined, float)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    second_combined, second_weights = adaptor.update({"loss_a": 0.5, "loss_b": 1.5})
    assert abs(sum(second_weights.values()) - 1.0) < 1e-6
    assert second_combined != combined


def test_symbolic_feedback_integration_adjusts_outputs():
    integration = SymbolicFeedbackIntegration(symbol_bias={1: 2.0}, scaling=0.5)
    outputs = [0.1, 0.2, 0.3]
    adjusted = integration.apply(outputs, {1: 1.0})
    assert adjusted[1] > outputs[1]


def test_recursive_self_audit_detects_drift_and_overfit():
    audit = RecursiveSelfAuditMechanism(window=10, drift_threshold=0.2, overfit_threshold=0.1)
    for _ in range(5):
        audit.record({"train_loss": 1.0, "val_loss": 1.05})
    for _ in range(5):
        audit.record({"train_loss": 0.5, "val_loss": 0.8})
    result = audit.audit()
    assert result["drift_alert"] is True
    assert result["overfit_alert"] is True


def test_neural_symbolic_replay_buffer_prioritises_difficult_samples():
    buffer = NeuralSymbolicReplayBuffer(capacity=5)
    buffer.add({"text": "easy"}, difficulty=0.1)
    buffer.add({"text": "medium"}, difficulty=0.5)
    buffer.add({"text": "hard"}, difficulty=0.9)
    samples = buffer.sample(2, strategy="topk")
    assert samples[0].sample["text"] == "hard"
    assert samples[1].sample["text"] == "medium"


def test_gradient_resonance_optimization_scales_gradients():
    random.seed(42)
    optimizer = GradientResonanceOptimization(resonance_strength=0.5, min_scale=0.5, max_scale=1.5, backend_hint="python")
    gradients = {"layer1": [1.0, 1.0], "layer2": [-1.0, -1.0]}
    first = optimizer.modulate(gradients)
    second = optimizer.modulate(gradients)
    assert first["layer1"] != second["layer1"]
    assert all(abs(v) <= 1.5 for v in second["layer1"])


class _DummyParam:
    def __init__(self) -> None:
        self.grad = None


class _DummyOptimizer:
    def __init__(self, params):
        self.param_groups = [{"params": list(params)}]
        self.steps = 0

    def step(self):
        self.steps += 1


def test_gradient_resonance_apply_updates_parameter_objects():
    resonance = GradientResonanceOptimization(resonance_strength=0.4, backend_hint="python")
    param_a = _DummyParam()
    param_b = _DummyParam()
    gradients = {param_a: [1.0, -1.0], param_b: [0.25, 0.75]}

    # Prime the running signatures.
    resonance.apply(gradients, _DummyOptimizer([param_a, param_b]))
    param_a.grad = None
    param_b.grad = None

    optimizer = _DummyOptimizer([param_a, param_b])
    scaled = resonance.apply(gradients, optimizer)

    assert optimizer.steps == 1
    assert param_a.grad == scaled[param_a]
    assert param_b.grad == scaled[param_b]


def test_replay_buffer_load_respects_saved_capacity():
    source = NeuralSymbolicReplayBuffer(capacity=2)
    source.add({"id": 1}, difficulty=0.4)
    source.add({"id": 2}, difficulty=0.6)
    state = source.state_dict()

    restored = NeuralSymbolicReplayBuffer(capacity=5)
    restored.load_state_dict(state)

    assert restored._capacity == 2
    assert restored._items.maxlen == 2
    assert len(restored) == 2

    restored.add({"id": 3}, difficulty=0.8)

    assert len(restored) == 2
    ids = [item.sample["id"] for item in restored._items]
    assert 3 in ids and 1 not in ids
