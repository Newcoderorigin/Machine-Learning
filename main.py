"""Interactive Sovereign ML dashboard entry point."""

import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:  # pragma: no cover - optional styling dependency
    import ttkbootstrap as tb
except Exception:  # pragma: no cover - optional dependency
    tb = None  # type: ignore

try:  # pragma: no cover - optional system metrics dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

from sov_ai.core import CREATOR_AUTHORITY, run_sov_ai
from sov_ai.geometry import generate_tesseract, project_points, rotation_matrix_4d
from sov_ai.tts import TextToSpeech

LOGGER = logging.getLogger(__name__)


@dataclass
class InteractionLog:
    """Dataclass representing an event captured in the GUI log."""

    timestamp: float
    prompt: str
    response: str
    reward: Optional[float]
    entropy: float
    weights_norm: float

    def to_row(self) -> tuple[str, str, str, str]:
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        reward_text = "-" if self.reward is None else f"{self.reward:+.2f}"
        return (
            ts,
            f"{self.weights_norm:.3f}",
            f"{self.entropy:.3f}",
            reward_text,
        )


class SovereignDashboard:
    """Elite local dashboard orchestrating the Sovereign ML subsystems."""

    def __init__(self, root: tk.Tk) -> None:
        self._configure_logging()
        self.root = root
        self.root.title("Sovereign ML Control Nexus")
        self.root.geometry("1280x820")
        self.root.minsize(1024, 720)

        if tb is not None:
            style = tb.Style("superhero")
            style.master = self.root  # attach theme to our window
            self.style = style
        else:
            self.style = ttk.Style()
            self.style.theme_use("clam")

        self.core = run_sov_ai()
        self.tts = TextToSpeech()
        self.memory_path = Path(self.core._memory.path)  # type: ignore[attr-defined]

        self.metric_time: Deque[float] = deque(maxlen=200)
        self.metric_value: Deque[float] = deque(maxlen=200)
        self.metric_entropy: Deque[float] = deque(maxlen=200)
        self.metric_cpu: Deque[float] = deque(maxlen=200)
        self._last_state: Dict[str, float] = self.core._learner.export_state()  # type: ignore[attr-defined]
        self._last_entropy: float = 0.0
        self._log_payloads: Dict[str, str] = {}

        self._build_layout()
        self._update_metrics_loop()

    def _configure_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.mainframe = ttk.Frame(self.root, padding=12)
        self.mainframe.grid(row=0, column=0, sticky="NSEW")
        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self.mainframe)
        self.notebook.grid(row=0, column=0, sticky="NSEW")

        self._build_train_tab()
        self._build_inspect_tab()
        self._build_visualize_tab()
        self._build_speak_tab()

    def _build_train_tab(self) -> None:
        self.train_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text="Train")

        self.train_tab.columnconfigure(0, weight=3)
        self.train_tab.columnconfigure(1, weight=2)
        self.train_tab.rowconfigure(0, weight=1)

        # Interaction Console
        console = ttk.LabelFrame(self.train_tab, text="Interaction Console", padding=10)
        console.grid(row=0, column=0, sticky="NSEW", padx=(0, 12), pady=6)
        console.columnconfigure(0, weight=1)
        console.rowconfigure(2, weight=1)

        ttk.Label(console, text="Prompt:").grid(row=0, column=0, sticky="W")
        self.prompt_entry = tk.Text(console, height=5, wrap=tk.WORD)
        self.prompt_entry.grid(row=1, column=0, sticky="NSEW", pady=(0, 8))

        ttk.Label(console, text="Sovereign Response:").grid(row=2, column=0, sticky="W")
        self.response_text = tk.Text(console, height=8, wrap=tk.WORD, state="disabled")
        self.response_text.grid(row=3, column=0, sticky="NSEW")

        # Control panel
        control = ttk.LabelFrame(self.train_tab, text="Reinforcement Controls", padding=10)
        control.grid(row=0, column=1, sticky="NSEW", pady=6)
        control.columnconfigure(0, weight=1)

        ttk.Label(control, text="Reward Signal (-1.0 to +1.0)").grid(row=0, column=0, sticky="W")
        self.reward_var = tk.DoubleVar(value=0.0)
        reward_scale = ttk.Scale(control, from_=-1.0, to=1.0, orient=tk.HORIZONTAL, variable=self.reward_var)
        reward_scale.grid(row=1, column=0, sticky="EW", pady=(0, 6))

        ttk.Label(control, text="Symbolic Feedback").grid(row=2, column=0, sticky="W")
        self.feedback_entry = ttk.Entry(control)
        self.feedback_entry.grid(row=3, column=0, sticky="EW", pady=(0, 8))

        self.render_var = tk.BooleanVar(value=False)
        self.voice_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control, text="Render Tesseract", variable=self.render_var).grid(row=4, column=0, sticky="W")
        ttk.Checkbutton(control, text="Speak Response", variable=self.voice_var).grid(row=5, column=0, sticky="W", pady=(4, 8))

        action_frame = ttk.Frame(control)
        action_frame.grid(row=6, column=0, sticky="EW")
        ttk.Button(action_frame, text="Engage", command=self._engage_interaction).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(action_frame, text="Reset", command=self._reset_inputs).grid(row=0, column=1)

        # Metrics canvas inside control frame
        metrics_box = ttk.LabelFrame(control, text="Adaptive Metrics", padding=8)
        metrics_box.grid(row=7, column=0, sticky="NSEW", pady=(12, 0))
        metrics_box.columnconfigure(0, weight=1)
        metrics_box.rowconfigure(0, weight=1)

        self.metric_fig = Figure(figsize=(4, 2.8), dpi=100)
        self.metric_ax = self.metric_fig.add_subplot(111)
        self.metric_ax.set_title("Weights Norm Evolution")
        self.metric_ax.set_xlabel("Interaction")
        self.metric_ax.set_ylabel("Norm")
        (self.metric_line,) = self.metric_ax.plot([], [], color="#4DD0E1", linewidth=2)
        self.metric_ax.grid(True, linestyle="--", alpha=0.3)
        self.metric_canvas = FigureCanvasTkAgg(self.metric_fig, master=metrics_box)
        self.metric_canvas.get_tk_widget().grid(row=0, column=0, sticky="NSEW")

    def _build_inspect_tab(self) -> None:
        self.inspect_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.inspect_tab, text="Inspect")

        self.inspect_tab.columnconfigure(0, weight=3)
        self.inspect_tab.columnconfigure(1, weight=2)
        self.inspect_tab.rowconfigure(0, weight=1)

        log_frame = ttk.LabelFrame(self.inspect_tab, text="Interaction Ledger", padding=8)
        log_frame.grid(row=0, column=0, sticky="NSEW", padx=(0, 12), pady=6)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        columns = ("time", "norm", "entropy", "reward")
        self.log_tree = ttk.Treeview(log_frame, columns=columns, show="headings", height=18)
        self.log_tree.heading("time", text="Time")
        self.log_tree.heading("norm", text="‖w‖")
        self.log_tree.heading("entropy", text="Entropy")
        self.log_tree.heading("reward", text="Reward")
        self.log_tree.column("time", width=90, anchor="center")
        self.log_tree.column("norm", width=80, anchor="center")
        self.log_tree.column("entropy", width=90, anchor="center")
        self.log_tree.column("reward", width=90, anchor="center")
        self.log_tree.grid(row=0, column=0, sticky="NSEW")
        self.log_tree.bind("<<TreeviewSelect>>", self._on_log_selected)

        detail_frame = ttk.LabelFrame(self.inspect_tab, text="Record Detail", padding=8)
        detail_frame.grid(row=0, column=1, sticky="NSEW", pady=6)
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)

        self.detail_text = tk.Text(detail_frame, wrap=tk.WORD, state="disabled")
        self.detail_text.grid(row=0, column=0, sticky="NSEW")

    def _build_visualize_tab(self) -> None:
        self.visual_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.visual_tab, text="Visualize")

        self.visual_tab.columnconfigure(1, weight=1)
        self.visual_tab.rowconfigure(0, weight=1)

        viz_frame = ttk.LabelFrame(self.visual_tab, text="4D Hypercube Projection", padding=10)
        viz_frame.grid(row=0, column=0, columnspan=2, sticky="NSEW", pady=6, padx=(0, 12))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)

        self.tesseract_fig = Figure(figsize=(5.5, 4.2), dpi=100)
        self.tesseract_ax = self.tesseract_fig.add_subplot(111, projection="3d")
        self.tesseract_canvas = FigureCanvasTkAgg(self.tesseract_fig, master=viz_frame)
        self.tesseract_canvas.get_tk_widget().grid(row=0, column=0, sticky="NSEW")

        control_frame = ttk.Frame(self.visual_tab)
        control_frame.grid(row=1, column=0, sticky="EW", padx=(0, 12), pady=(6, 0))
        for i in range(4):
            control_frame.columnconfigure(i, weight=1)

        self.angle_vars = [tk.DoubleVar(value=angle) for angle in (0.0, 0.6, 0.3, 0.9)]
        labels = ("XY", "XW", "YW", "ZW")
        for idx, (label, var) in enumerate(zip(labels, self.angle_vars)):
            ttk.Label(control_frame, text=f"Rotation {label}").grid(row=0, column=idx, sticky="EW")
            scale = ttk.Scale(
                control_frame,
                from_=-math.pi,
                to=math.pi,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda *_: self._draw_tesseract(),
            )
            scale.grid(row=1, column=idx, sticky="EW", padx=6)

        self._draw_tesseract()

    def _build_speak_tab(self) -> None:
        self.speak_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.speak_tab, text="Speak")

        self.speak_tab.columnconfigure(0, weight=1)
        self.speak_tab.rowconfigure(1, weight=1)

        ttk.Label(
            self.speak_tab,
            text=(
                "Offline speech synthesis channel. Type your proclamation "
                "and the Sovereign will vocalize using local TTS."
            ),
            wraplength=760,
        ).grid(row=0, column=0, sticky="W", padx=12, pady=(18, 12))

        self.speak_entry = tk.Text(self.speak_tab, height=6, wrap=tk.WORD)
        self.speak_entry.grid(row=1, column=0, sticky="NSEW", padx=12)

        speak_controls = ttk.Frame(self.speak_tab)
        speak_controls.grid(row=2, column=0, sticky="EW", padx=12, pady=12)
        speak_controls.columnconfigure(0, weight=1)
        ttk.Button(speak_controls, text="Broadcast", command=self._speak_direct).grid(row=0, column=0, sticky="E")

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------
    def _reset_inputs(self) -> None:
        self.prompt_entry.delete("1.0", tk.END)
        self.feedback_entry.delete(0, tk.END)
        self.reward_var.set(0.0)
        self.voice_var.set(False)
        self.render_var.set(False)

    def _engage_interaction(self) -> None:
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        feedback = self.feedback_entry.get().strip() or None
        reward = self.reward_var.get()
        render = self.render_var.get()
        speak = self.voice_var.get()

        if not prompt:
            messagebox.showinfo("Missing Prompt", "Compose a prompt before engaging.")
            return

        reward_signal: Optional[float] = reward if abs(reward) > 1e-6 else None

        try:
            result = self.core.interact(
                prompt,
                reward=reward_signal,
                symbolic_feedback=feedback,
                render=render,
                speak=speak,
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            LOGGER.exception("Interaction failed: %s", exc)
            messagebox.showerror("Interaction Failure", str(exc))
            return

        response = str(result["response"])
        analysis = result.get("analysis", {})
        state = result.get("state", {})

        self._last_state = state
        entropy = float(analysis.get("entropy", 0.0))
        self._last_entropy = entropy
        weights_norm = float(state.get("weights_norm", 0.0))
        timestamp = time.time()

        self._append_response(response)
        self._register_log(
            InteractionLog(
                timestamp=timestamp,
                prompt=prompt,
                response=response,
                reward=reward_signal,
                entropy=entropy,
                weights_norm=weights_norm,
            )
        )
        self._update_metric_plot()

        message = (
            f"Authority: {CREATOR_AUTHORITY}\n"
            f"Weights Norm: {weights_norm:.3f}\n"
            f"Entropy: {entropy:.3f}\n"
            f"Memory Vault: {self.memory_path}"
        )
        LOGGER.info(message)

    def _append_response(self, response: str) -> None:
        self.response_text.configure(state="normal")
        self.response_text.delete("1.0", tk.END)
        self.response_text.insert(tk.END, response)
        self.response_text.configure(state="disabled")

    def _register_log(self, entry: InteractionLog) -> None:
        self.log_tree.insert("", tk.END, values=entry.to_row(), iid=str(entry.timestamp))
        detail_payload = {
            "prompt": entry.prompt,
            "response": entry.response,
            "reward": entry.reward,
            "entropy": entry.entropy,
            "weights_norm": entry.weights_norm,
            "timestamp": entry.timestamp,
        }
        self._log_payloads[str(entry.timestamp)] = json.dumps(detail_payload, indent=2)
        self.metric_time.append(time.time())
        self.metric_value.append(entry.weights_norm)
        self.metric_entropy.append(entry.entropy)
        cpu = psutil.cpu_percent(interval=None) if psutil else 0.0
        self.metric_cpu.append(cpu)

    def _on_log_selected(self, event: tk.Event) -> None:
        selection = self.log_tree.selection()
        if not selection:
            return
        iid = selection[0]
        payload = self._log_payloads.get(iid, "{}")
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, payload)
        self.detail_text.configure(state="disabled")

    def _speak_direct(self) -> None:
        text = self.speak_entry.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Silence", "Enter text to broadcast.")
            return
        self.tts.speak(text)

    # ------------------------------------------------------------------
    # Visualization and metric updates
    # ------------------------------------------------------------------
    def _draw_tesseract(self) -> None:
        vertices, edges = generate_tesseract(scale=1.0)
        angles = [var.get() for var in self.angle_vars]
        rotation = rotation_matrix_4d(*angles)
        projected = project_points(vertices, rotation)

        self.tesseract_ax.clear()
        self.tesseract_ax.set_facecolor("#0A0A0A")
        self.tesseract_ax.figure.patch.set_facecolor("#10141A")
        self.tesseract_ax.set_title("4D Hypercube Projection", color="#E0F7FA")

        xs, ys, zs = projected[:, 0], projected[:, 1], projected[:, 2]
        self.tesseract_ax.scatter(xs, ys, zs, color="#FF4081", s=30, depthshade=True)

        for start, end in edges:
            line = np.vstack((projected[start], projected[end]))
            self.tesseract_ax.plot(line[:, 0], line[:, 1], line[:, 2], color="#4DD0E1", linewidth=1.2)

        self.tesseract_ax.set_xlabel("X")
        self.tesseract_ax.set_ylabel("Y")
        self.tesseract_ax.set_zlabel("Z")
        self.tesseract_ax.grid(False)
        self.tesseract_ax.set_xlim(-3, 3)
        self.tesseract_ax.set_ylim(-3, 3)
        self.tesseract_ax.set_zlim(-3, 3)
        self.tesseract_canvas.draw_idle()

    def _update_metric_plot(self) -> None:
        if not self.metric_value:
            return
        x = list(range(len(self.metric_value)))
        y = list(self.metric_value)
        self.metric_line.set_data(x, y)
        self.metric_ax.set_xlim(0, max(10, len(x)))
        if y:
            ymin = min(y) - 0.1
            ymax = max(y) + 0.1
            if ymin == ymax:
                ymax += 1.0
            self.metric_ax.set_ylim(ymin, ymax)
        self.metric_canvas.draw_idle()

    def _update_metrics_loop(self) -> None:
        # Passive monitoring of entropy and cpu usage for the inspect tab detail.
        entropy = float(self._last_entropy)
        if self.metric_time:
            # update last entry entropy to reflect smoothing
            self.metric_entropy[-1] = (self.metric_entropy[-1] + entropy) / 2
        if psutil:
            cpu = psutil.cpu_percent(interval=None)
            if self.metric_cpu:
                self.metric_cpu[-1] = (self.metric_cpu[-1] + cpu) / 2
        self.root.after(2000, self._update_metrics_loop)


def main() -> None:
    root = tb.Window(themename="superhero") if tb is not None else tk.Tk()
    app = SovereignDashboard(root)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover - GUI entry point
    main()
