"""Tests for MCP modifier option wiring."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

import pdf_modifier.interfaces.mcp as mcp_module
from pdf_modifier.interfaces.mcp import batch_modify_pdf_content, modify_pdf_content

if TYPE_CHECKING:
    from pathlib import Path


def test_modify_wires_custom_fonts_and_rgb_fill(monkeypatch: Any, tmp_path: Path) -> None:
    """The single-file MCP tool forwards font mappings and parsed RGB fill."""
    captured: dict[str, Any] = {}
    result = MagicMock()
    result.model_dump_json.return_value = "{}"

    class FakeModifier:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        def process(self, *args: Any, **kwargs: Any) -> MagicMock:
            return result

    monkeypatch.setattr(mcp_module, "PDFModifier", FakeModifier)
    custom_fonts = {"HeBo": str(tmp_path / "font.ttf")}

    response = modify_pdf_content(
        "/input.pdf",
        "/output.pdf",
        {"old": "new"},
        custom_fonts=custom_fonts,
        redact_fill=[0.1, 0.2, 0.3],
    )

    assert response == "{}"
    assert captured["custom_fonts"] is custom_fonts
    assert captured["redact_fill"] == (0.1, 0.2, 0.3)


def test_batch_wires_custom_fonts_and_none_fill(monkeypatch: Any, tmp_path: Path) -> None:
    """The batch MCP tool converts the literal 'none' to a disabled fill."""
    captured: dict[str, Any] = {}
    result = MagicMock()
    result.model_dump_json.return_value = "{}"

    def fake_batch_process(*args: Any, **kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return result

    monkeypatch.setattr(mcp_module, "batch_process", fake_batch_process)
    custom_fonts = {"helv": str(tmp_path / "font.ttf")}

    response = batch_modify_pdf_content(
        ["/input.pdf"],
        "/output",
        {"old": "new"},
        custom_fonts=custom_fonts,
        redact_fill="none",
    )

    assert response == "{}"
    assert captured["custom_fonts"] is custom_fonts
    assert captured["redact_fill"] is None


@pytest.mark.parametrize(
    "redact_fill",
    [[0.1, 0.2], [0.1, 0.2, 1.5], "invalid", 7, ["a", "b", "c"]],
)
def test_modify_rejects_invalid_redact_fill(redact_fill: Any) -> None:
    """Invalid fill values surface as an error JSON, not a crash."""
    response = modify_pdf_content(
        "/input.pdf",
        "/output.pdf",
        {"old": "new"},
        redact_fill=redact_fill,
    )

    data = json.loads(response)
    assert data["success"] is False
    assert "redact_fill" in data["message"]


def test_modify_wires_fit_text(monkeypatch: Any) -> None:
    """The single-file MCP tool forwards fit_text to PDFModifier."""
    captured: dict[str, Any] = {}
    result = MagicMock()
    result.model_dump_json.return_value = "{}"

    class FakeModifier:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        def process(self, *args: Any, **kwargs: Any) -> MagicMock:
            return result

    monkeypatch.setattr(mcp_module, "PDFModifier", FakeModifier)

    modify_pdf_content("/input.pdf", "/output.pdf", {"old": "new"}, fit_text=True)
    assert captured["fit_text"] is True

    modify_pdf_content("/input.pdf", "/output.pdf", {"old": "new"})
    assert captured["fit_text"] is False


def test_batch_wires_fit_text(monkeypatch: Any) -> None:
    """The batch MCP tool forwards fit_text to batch_process."""
    captured: dict[str, Any] = {}
    result = MagicMock()
    result.model_dump_json.return_value = "{}"

    def fake_batch_process(*args: Any, **kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return result

    monkeypatch.setattr(mcp_module, "batch_process", fake_batch_process)

    batch_modify_pdf_content(["/input.pdf"], "/output", {"old": "new"}, fit_text=True)
    assert captured["fit_text"] is True
