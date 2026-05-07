"""Ephemeral Textual tuner for LLM punctuation summary figures.

Run with:
    uv run --with textual experiments/llm_sentences_last_layer/scripts/plot_llm_punctuation_figures_tui.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Label, Static

from plot_llm_punctuation_figures import DEFAULT_CONFIG, FigureConfig, plot_all


SUMMARY_PATH = Path("experiments/llm_sentences_last_layer/results/summary/summary.json")
OUTPUT_DIR = Path("experiments/llm_sentences_last_layer/results/summary/figures")


class LLMFigureTuner(App):
    """Small interactive control surface for exporting the LLM figure set."""

    CSS = """
    Screen {
        padding: 1 2;
    }
    #controls {
        width: 56;
        padding-right: 2;
    }
    Input {
        margin-bottom: 1;
    }
    #status {
        border: round $accent;
        padding: 1 2;
        height: 8;
    }
    Button {
        margin-right: 1;
    }
    """

    BINDINGS = [
        ("e", "export", "Export"),
        ("r", "reset", "Reset"),
        ("z", "undo", "Undo"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.config = DEFAULT_CONFIG
        self.previous_config: FigureConfig | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="controls"):
                yield Label("Canvas")
                yield Input(str(self.config.dpi), id="dpi", placeholder="dpi")
                yield Input(str(self.config.panel_height), id="panel_height", placeholder="panel height")
                yield Input(str(self.config.three_panel_width), id="three_panel_width", placeholder="three-panel width")
                yield Input(str(self.config.single_panel_width), id="single_panel_width", placeholder="single-panel width")
                yield Label("Typography")
                yield Input(str(self.config.tick_fontsize), id="tick_fontsize", placeholder="tick fontsize")
                yield Input(str(self.config.label_fontsize), id="label_fontsize", placeholder="label fontsize")
                yield Input(str(self.config.title_fontsize), id="title_fontsize", placeholder="title fontsize")
                yield Input(str(self.config.legend_fontsize), id="legend_fontsize", placeholder="legend fontsize")
                with Horizontal():
                    yield Button("Export", id="export", variant="primary")
                    yield Button("Reset", id="reset")
                    yield Button("Undo", id="undo")
            yield Static(self.status_text(), id="status")
        yield Footer()

    def status_text(self) -> str:
        return (
            f"Summary: {SUMMARY_PATH}\n"
            f"Output: {OUTPUT_DIR}\n"
            f"Config: dpi={self.config.dpi}, panel_height={self.config.panel_height}, "
            f"three_panel_width={self.config.three_panel_width}, single_panel_width={self.config.single_panel_width}\n"
            "Press e to export, r to reset, z to undo."
        )

    def refresh_status(self, message: str | None = None) -> None:
        status = self.query_one("#status", Static)
        text = self.status_text()
        if message:
            text = f"{message}\n\n{text}"
        status.update(text)

    def read_config(self) -> FigureConfig:
        def input_value(widget_id: str, cast):
            raw = self.query_one(f"#{widget_id}", Input).value
            return cast(raw)

        return replace(
            self.config,
            dpi=input_value("dpi", int),
            panel_height=input_value("panel_height", float),
            three_panel_width=input_value("three_panel_width", float),
            single_panel_width=input_value("single_panel_width", float),
            tick_fontsize=input_value("tick_fontsize", float),
            label_fontsize=input_value("label_fontsize", float),
            title_fontsize=input_value("title_fontsize", float),
            legend_fontsize=input_value("legend_fontsize", float),
        )

    def write_inputs(self) -> None:
        for field_name in (
            "dpi",
            "panel_height",
            "three_panel_width",
            "single_panel_width",
            "tick_fontsize",
            "label_fontsize",
            "title_fontsize",
            "legend_fontsize",
        ):
            self.query_one(f"#{field_name}", Input).value = str(getattr(self.config, field_name))

    @on(Button.Pressed, "#export")
    def export_button(self) -> None:
        self.action_export()

    @on(Button.Pressed, "#reset")
    def reset_button(self) -> None:
        self.action_reset()

    @on(Button.Pressed, "#undo")
    def undo_button(self) -> None:
        self.action_undo()

    def action_export(self) -> None:
        try:
            new_config = self.read_config()
        except ValueError as exc:
            self.refresh_status(f"Invalid numeric input: {exc}")
            return
        self.previous_config = self.config
        self.config = new_config
        outputs = plot_all(SUMMARY_PATH, OUTPUT_DIR, self.config)
        self.refresh_status(f"Exported {len(outputs)} PNG files.")

    def action_reset(self) -> None:
        self.previous_config = self.config
        self.config = DEFAULT_CONFIG
        self.write_inputs()
        self.refresh_status("Reset to defaults.")

    def action_undo(self) -> None:
        if self.previous_config is None:
            self.refresh_status("No previous config to restore.")
            return
        self.config, self.previous_config = self.previous_config, self.config
        self.write_inputs()
        self.refresh_status("Restored previous config.")


if __name__ == "__main__":
    LLMFigureTuner().run()
