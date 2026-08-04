"""Tests for CLI --fit-text wiring."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from typer.testing import CliRunner

import pdf_modifier.interfaces.cli as cli_module
from pdf_modifier.interfaces.cli import app

runner = CliRunner()


def _fake_modifier(captured: dict[str, Any]) -> type:
    result = MagicMock()
    result.output_path = "/out.pdf"
    result.replacements_made = 1
    result.pages_modified = 1
    result.failed_insertions = 0
    result.warnings = []

    class FakeModifier:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        def process(self, *args: Any, **kwargs: Any) -> MagicMock:
            return result

    return FakeModifier


def test_modify_wires_fit_text(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(cli_module, "PDFModifier", _fake_modifier(captured))

    result = runner.invoke(
        app, ["modify", "in.pdf", "out.pdf", "-r", "old=new", "--fit-text"]
    )
    assert result.exit_code == 0
    assert captured["fit_text"] is True


def test_modify_fit_text_defaults_off(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(cli_module, "PDFModifier", _fake_modifier(captured))

    result = runner.invoke(app, ["modify", "in.pdf", "out.pdf", "-r", "old=new"])
    assert result.exit_code == 0
    assert captured["fit_text"] is False


def test_batch_wires_fit_text(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    batch_result = MagicMock()
    batch_result.results = []
    batch_result.errors = []
    batch_result.successful = 0
    batch_result.failed = 0
    batch_result.total_files = 1

    def fake_batch_process(*args: Any, **kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return batch_result

    monkeypatch.setattr(cli_module, "batch_process", fake_batch_process)

    result = runner.invoke(
        app, ["batch", "in.pdf", "-o", "out", "-r", "old=new", "--fit-text"]
    )
    assert result.exit_code == 0
    assert captured["fit_text"] is True


def test_modify_reports_failed_insertions(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    fake_cls = _fake_modifier(captured)
    monkeypatch.setattr(cli_module, "PDFModifier", fake_cls)

    result_obj = fake_cls("x", "y").process()
    result_obj.failed_insertions = 2

    result = runner.invoke(app, ["modify", "in.pdf", "out.pdf", "-r", "old=new"])
    assert result.exit_code == 0
    assert "Failed insertions" in result.output
