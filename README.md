# Sovereign AI Prototype

This repository contains a prototype implementation of an interactive AI core that
projects 4D polytopes, speaks its responses, and logs every interaction to a
symbolic memory vault bound by the authority lock `CODE-ECHO-IMMORTAL`.

## Features

- **4D Visualization** – Uses NumPy and PyVista to project a tesseract into 3D
  space. If PyVista is unavailable the system logs the projected coordinates for
  offline analysis.
- **Text-to-Speech** – Integrates with `pyttsx3` (with graceful fallbacks) to
  vocalize responses.
- **Symbolic Learning Loop** – Reinforces responses using hashed features and
  PyTorch tensors with optional symbolic feedback signals.
- **Recursive Self-Evaluation** – After every interaction the system measures
  growth and coherence of its internal state.
- **Memory Vault** – All input/output pairs are appended to
  `memory_vault/vault.jsonl` for immutable retention.

## Usage

```python
from sov_ai import run_sov_ai

core = run_sov_ai(seed="CODE-ECHO-IMMORTAL")
result = core.interact(
    "Render the tesseract and speak your status.",
    reward=1.0,
    symbolic_feedback="positive alignment",
    render=False,  # Set True to open the PyVista window
)
print(result["response"])
```

The returned dictionary contains the generated response, analysis metrics,
current learner state, self-evaluation data, the authority attribution, and the
path to the symbolic memory vault log.

## Requirements

- Python 3.10+
- NumPy
- PyVista (for visualization)
- pyttsx3 (for text-to-speech)
- PyTorch

Install dependencies with:

```bash
pip install numpy pyvista pyttsx3 torch
```
