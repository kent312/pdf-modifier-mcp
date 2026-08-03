"""Tests for configurable PDF redaction fill colors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, call

import fitz
import pytest

from pdf_modifier.core.models import ModificationResult, ReplacementSpec
from pdf_modifier.core.modifier import PDFModifier, batch_process

if TYPE_CHECKING:
    from pathlib import Path


def _apply_redactions(modifier: PDFModifier) -> MagicMock:
    """Apply both multi-bbox and single-bbox annotations to a mock page."""
    page = MagicMock()
    modifier._insert_replacement = MagicMock()  # type: ignore[method-assign]
    first = fitz.Rect(1, 2, 3, 4)
    second = fitz.Rect(5, 6, 7, 8)
    single = fitz.Rect(9, 10, 11, 12)
    items: list[dict[str, Any]] = [
        {"bboxes": [first, second]},
        {"bbox": single},
    ]

    modifier._apply_replacements_to_page(page, items)

    page.add_redact_annot.assert_has_calls(
        [
            call(first, fill=modifier._redact_fill),
            call(second, fill=modifier._redact_fill),
            call(single, fill=modifier._redact_fill),
        ]
    )
    return page


class TestModifierRedactFill:
    """Tests for PDFModifier redact_fill validation and use."""

    def test_default_fill_remains_white(self) -> None:
        modifier = PDFModifier("input.pdf", "output.pdf")

        page = _apply_redactions(modifier)

        assert modifier._redact_fill == (1.0, 1.0, 1.0)
        assert page.add_redact_annot.call_count == 3

    def test_custom_rgb_fill_is_used(self) -> None:
        modifier = PDFModifier("input.pdf", "output.pdf", redact_fill=(0.9, 0.8, 0.7))

        page = _apply_redactions(modifier)

        assert modifier._redact_fill == (0.9, 0.8, 0.7)
        assert page.add_redact_annot.call_count == 3

    def test_none_disables_fill(self) -> None:
        modifier = PDFModifier("input.pdf", "output.pdf", redact_fill=None)
        modifier._insert_replacement = MagicMock()  # type: ignore[method-assign]
        page = MagicMock()

        modifier._apply_replacements_to_page(page, [{"bbox": fitz.Rect(1, 2, 3, 4)}])

        page.add_redact_annot.assert_called_once_with(fitz.Rect(1, 2, 3, 4), fill=False)

    @pytest.mark.parametrize(
        "redact_fill",
        [(-0.1, 0.0, 0.0), (0.0, 1.1, 0.0), (0.0, 0.0, float("nan")), 7, "abc", ("a", "b", "c")],
    )
    def test_out_of_range_fill_raises(self, redact_fill: Any) -> None:
        with pytest.raises(ValueError, match="redact_fill"):
            PDFModifier("input.pdf", "output.pdf", redact_fill=redact_fill)

    def test_none_fill_works_with_real_pdf(self, tmp_path: Path) -> None:
        """A real fitz page accepts fill=False and the replacement lands."""
        src = tmp_path / "in.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello World")
        doc.save(str(src))
        doc.close()

        out = tmp_path / "out.pdf"
        modifier = PDFModifier(src, out, redact_fill=None)
        result = modifier.process(ReplacementSpec(replacements={"Hello": "Goodbye"}))

        assert result.success
        check = fitz.open(str(out))
        text = check[0].get_text()
        check.close()
        assert "Goodbye" in text
        assert "Hello" not in text


def test_batch_process_propagates_redact_fill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """batch_process passes the configured fill to each PDFModifier."""
    captured: list[tuple[float, float, float] | None] = []

    class FakeModifier:
        def __init__(
            self, *args: Any, redact_fill: tuple[float, float, float] | None, **kwargs: Any
        ):
            captured.append(redact_fill)

        def process(self, spec: ReplacementSpec) -> ModificationResult:
            return ModificationResult(
                success=True,
                input_path="input.pdf",
                output_path="output.pdf",
                replacements_made=1,
                pages_modified=1,
            )

    monkeypatch.setattr("pdf_modifier.core.modifier.PDFModifier", FakeModifier)
    input_path = tmp_path / "input.pdf"
    spec = ReplacementSpec(replacements={"old": "new"})

    result = batch_process(
        [input_path],
        tmp_path / "output",
        spec,
        redact_fill=(0.1, 0.2, 0.3),
    )

    assert result.successful == 1
    assert captured == [(0.1, 0.2, 0.3)]
