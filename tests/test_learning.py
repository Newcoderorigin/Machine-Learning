import hashlib
import importlib.util
import sys
import types
from pathlib import Path


def _not_available(*_args, **_kwargs):  # pragma: no cover - helper for stubbed torch
    raise RuntimeError("torch functionality is not available in tests")


torch_stub = types.ModuleType("torch")


class _TensorStub:  # pragma: no cover - placeholder for torch.Tensor type
    __module__ = "torch"


torch_stub.Tensor = _TensorStub
torch_stub.float32 = "float32"
torch_stub.zeros = _not_available
torch_stub.dot = _not_available
torch_stub.tanh = _not_available
torch_stub.tensor = _not_available
torch_stub.linalg = types.SimpleNamespace(norm=_not_available)
sys.modules["torch"] = torch_stub

learning_path = Path(__file__).resolve().parents[1] / "sov_ai" / "learning.py"
spec = importlib.util.spec_from_file_location("sov_ai.learning", learning_path)
learning = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = learning
spec.loader.exec_module(learning)
SymbolicLearner = learning.SymbolicLearner


def test_hash_token_is_deterministic():
    feature_dim = 1024
    token = "example-token"

    idx1 = SymbolicLearner._hash_token(token) % feature_dim
    idx2 = SymbolicLearner._hash_token(token) % feature_dim
    assert idx1 == idx2

    expected_digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
    expected_index = int.from_bytes(expected_digest, byteorder="big", signed=False) % feature_dim
    assert idx1 == expected_index

    other_token = "Example-Token"  # different case to ensure normalization happens only elsewhere
    idx_other_first = SymbolicLearner._hash_token(other_token) % feature_dim
    idx_other_second = SymbolicLearner._hash_token(other_token) % feature_dim
    assert idx_other_first == idx_other_second
