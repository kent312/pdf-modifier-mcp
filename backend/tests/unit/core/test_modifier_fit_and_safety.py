"""Tests for fit_text width fitting and insertion failure safety."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import fitz

from pdf_modifier.core.models import ReplacementSpec
from pdf_modifier.core.modifier import MIN_FIT_FONTSIZE, PDFModifier

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _create_test_pdf(tmp_path: Path, text: str = "Hello World") -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontname="helv", fontsize=12)
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def _spans(pdf_path: Path, needle: str) -> list[dict[str, Any]]:
    """Collect spans containing the needle text."""
    doc = fitz.open(str(pdf_path))
    spans = [
        span
        for block in doc[0].get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
        if needle in span["text"]
    ]
    doc.close()
    return spans


def _span_sizes(pdf_path: Path, needle: str) -> list[float]:
    """Collect font sizes of spans containing the needle text."""
    return [span["size"] for span in _spans(pdf_path, needle)]


class TestFitText:
    """Tests for the fit_text option."""

    LONG = "This replacement is much longer than the original text was"

    def test_long_replacement_is_shrunk(self, tmp_path: Path) -> None:
        source = _create_test_pdf(tmp_path, "Hello World")
        output = tmp_path / "out.pdf"

        original_bbox = _spans(source, "Hello World")[0]["bbox"]

        modifier = PDFModifier(str(source), str(output), fit_text=True)
        result = modifier.process(
            ReplacementSpec(replacements={"Hello World": "Hello World plus a tail"})
        )

        assert result.replacements_made == 1
        assert not result.warnings  # moderate shrink must not hit the clamp
        spans = _spans(output, "plus a tail")
        assert spans, "replacement text not found in output"
        assert all(span["size"] < 12 for span in spans)
        # The fitted text must actually stay within the original bbox width.
        assert all(span["bbox"][2] <= original_bbox[2] + 1.0 for span in spans)

    def test_extreme_shrink_clamps_to_minimum(self, tmp_path: Path) -> None:
        """A huge replacement into a tiny bbox clamps at MIN_FIT_FONTSIZE."""
        source = _create_test_pdf(tmp_path, "I")
        output = tmp_path / "out.pdf"

        modifier = PDFModifier(str(source), str(output), fit_text=True)
        result = modifier.process(ReplacementSpec(replacements={"I": self.LONG * 3}))

        assert result.replacements_made == 1
        sizes = _span_sizes(output, "This replacement")
        assert sizes
        assert all(abs(size - MIN_FIT_FONTSIZE) < 0.1 for size in sizes)
        assert any("clamped" in w for w in result.warnings)

    def test_degenerate_bbox_keeps_size(self, tmp_path: Path) -> None:
        """A zero-width bbox skips fitting and keeps the original size."""
        modifier = PDFModifier("dummy.pdf", "out.pdf", fit_text=True)
        item = {"size": 12.0, "bbox": (72.0, 72.0, 72.0, 84.0), "text": "abc", "fontname": "helv"}
        assert modifier._fit_fontsize(item) == 12.0

    def test_measurement_failure_keeps_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A font that cannot be measured falls back to the original size."""
        modifier = PDFModifier("dummy.pdf", "out.pdf", fit_text=True)

        def _fail(fontname: str, fontfile: str | None) -> fitz.Font:
            raise RuntimeError("simulated measurement failure")

        monkeypatch.setattr(modifier, "_load_font", _fail)
        item = {"size": 12.0, "bbox": (72.0, 72.0, 200.0, 84.0), "text": "abc", "fontname": "helv"}
        assert modifier._fit_fontsize(item) == 12.0

    def test_default_keeps_original_size(self, tmp_path: Path) -> None:
        source = _create_test_pdf(tmp_path, "Hi")
        output = tmp_path / "out.pdf"

        modifier = PDFModifier(str(source), str(output))
        modifier.process(ReplacementSpec(replacements={"Hi": self.LONG}))

        sizes = _span_sizes(output, "This replacement")
        assert sizes
        assert all(abs(size - 12) < 0.1 for size in sizes)

    def test_short_replacement_keeps_size(self, tmp_path: Path) -> None:
        source = _create_test_pdf(tmp_path, "Hello World")
        output = tmp_path / "out.pdf"

        modifier = PDFModifier(str(source), str(output), fit_text=True)
        modifier.process(ReplacementSpec(replacements={"Hello World": "Hi"}))

        sizes = _span_sizes(output, "Hi")
        assert sizes
        assert all(abs(size - 12) < 0.1 for size in sizes)

    def test_fit_with_cjk_replacement(self, tmp_path: Path) -> None:
        """CJK auto-routing and fitting compose: long Japanese text shrinks."""
        source = _create_test_pdf(tmp_path, "Hi")
        output = tmp_path / "out.pdf"

        long_ja = "これは元のテキストよりずっと長い日本語の置換文です"
        modifier = PDFModifier(str(source), str(output), fit_text=True)
        result = modifier.process(ReplacementSpec(replacements={"Hi": long_ja}))

        assert result.replacements_made == 1
        doc = fitz.open(str(output))
        text = doc[0].get_text("text")
        doc.close()
        assert "日本語の置換文" in text
        sizes = _span_sizes(output, "日本語")
        assert sizes and all(size < 12 for size in sizes)


class TestInsertionSafety:
    """Tests for preflight font validation and guarded insertion."""

    def test_broken_font_file_skips_item_before_redaction(self, tmp_path: Path) -> None:
        """A corrupt font file must not destroy the original text."""
        source = _create_test_pdf(tmp_path, "Hello World")
        output = tmp_path / "out.pdf"

        fake_font = tmp_path / "broken.ttf"
        fake_font.write_bytes(b"this is not a real font")

        modifier = PDFModifier(
            str(source),
            str(output),
            custom_fonts={"helv": str(fake_font)},
        )
        result = modifier.process(ReplacementSpec(replacements={"Hello World": "Goodbye"}))

        assert result.success is True
        assert result.replacements_made == 0
        assert result.pages_modified == 0
        assert any("cannot load font file" in w for w in result.warnings)

        # Original text is untouched because redaction never ran.
        doc = fitz.open(str(output))
        text = doc[0].get_text("text")
        doc.close()
        assert "Hello World" in text

    def test_insert_failure_warns_instead_of_aborting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _create_test_pdf(tmp_path, "Hello World")
        output = tmp_path / "out.pdf"

        def _fail(self: fitz.Page, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated insert failure")

        monkeypatch.setattr(fitz.Page, "insert_text", _fail)

        modifier = PDFModifier(str(source), str(output))
        result = modifier.process(ReplacementSpec(replacements={"Hello World": "Goodbye"}))

        assert result.success is True
        assert result.replacements_made == 0
        assert result.pages_modified == 1  # redaction was applied
        assert result.failed_insertions == 1
        assert any("Could not insert replacement" in w for w in result.warnings)

        # The redaction-only outcome is visible: old text gone, new text absent.
        doc = fitz.open(str(output))
        text = doc[0].get_text("text")
        doc.close()
        assert "Hello World" not in text
        assert "Goodbye" not in text

    def test_partial_insert_failure_counts_and_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One failing insertion does not stop the others."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "First", fontname="helv", fontsize=12)
        page.insert_text((72, 144), "Second", fontname="helv", fontsize=12)
        source = tmp_path / "two.pdf"
        doc.save(str(source))
        doc.close()
        output = tmp_path / "out.pdf"

        real_insert_text = fitz.Page.insert_text

        def _fail_selectively(self: fitz.Page, point: Any, text: str, **kwargs: Any) -> Any:
            if "Beta" in text:
                raise RuntimeError("simulated insert failure")
            return real_insert_text(self, point, text, **kwargs)

        monkeypatch.setattr(fitz.Page, "insert_text", _fail_selectively)

        modifier = PDFModifier(str(source), str(output))
        result = modifier.process(
            ReplacementSpec(replacements={"First": "Alpha", "Second": "Beta"})
        )

        assert result.success is True
        assert result.replacements_made == 1
        assert result.failed_insertions == 1

        out = fitz.open(str(output))
        text = out[0].get_text("text")
        out.close()
        assert "Alpha" in text
        assert "Beta" not in text

    def test_partial_failure_processes_remaining_items(self, tmp_path: Path) -> None:
        """One broken-font item is skipped while others are replaced."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "KeepFont", fontname="helv", fontsize=12)
        page.insert_text((72, 144), "Target", fontname="cour", fontsize=12)
        source = tmp_path / "two.pdf"
        doc.save(str(source))
        doc.close()

        fake_font = tmp_path / "broken.ttf"
        fake_font.write_bytes(b"not a font")

        # Broken font maps only to Courier; Helvetica item stays Base 14.
        modifier = PDFModifier(
            str(source),
            str(tmp_path / "out.pdf"),
            custom_fonts={"Cour": str(fake_font)},
        )
        result = modifier.process(
            ReplacementSpec(replacements={"KeepFont": "Replaced", "Target": "Dropped"})
        )

        assert result.success is True
        assert result.replacements_made == 1
        assert any("cannot load font file" in w for w in result.warnings)

        out = fitz.open(str(tmp_path / "out.pdf"))
        text = out[0].get_text("text")
        out.close()
        assert "Replaced" in text
        assert "Target" in text  # skipped item kept its original text
