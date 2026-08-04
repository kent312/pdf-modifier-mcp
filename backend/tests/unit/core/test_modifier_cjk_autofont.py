"""Tests for automatic CJK routing to the bundled Noto Sans JP font."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import fitz

from pdf_modifier.core.models import ReplacementSpec
from pdf_modifier.core.modifier import PDFModifier
from pdf_modifier.fonts import bundled_font_path

if TYPE_CHECKING:
    import pytest


def _create_test_pdf(tmp_path: Path, text: str = "Hello World") -> Path:
    """Create a simple PDF with Base 14 Helvetica text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontname="helv", fontsize=12)
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


class TestResolveFontProps:
    """Unit tests for PDFModifier._resolve_font_props CJK routing."""

    def _modifier(self, **kwargs: object) -> PDFModifier:
        return PDFModifier("dummy.pdf", "out.pdf", **kwargs)  # type: ignore[arg-type]

    def test_cjk_text_routes_to_bundled_font(self) -> None:
        modifier = self._modifier()
        props = modifier._resolve_font_props("Helvetica", None, "こんにちは")
        assert props.fontfile == bundled_font_path()
        assert props.embed is True

    def test_cjk_bold_routes_to_bundled_bold(self) -> None:
        modifier = self._modifier()
        props = modifier._resolve_font_props("Helvetica-Bold", None, "太字テスト")
        assert props.fontfile == bundled_font_path(bold=True)

    def test_cjk_bold_flag_routes_to_bundled_bold(self) -> None:
        modifier = self._modifier()
        props = modifier._resolve_font_props("UnknownFont", {"bold": 1}, "太字テスト")
        assert props.fontfile == bundled_font_path(bold=True)

    def test_cjk_italic_degrades_to_regular(self) -> None:
        """No italic CJK variant is bundled; italic degrades to Regular."""
        modifier = self._modifier()
        props = modifier._resolve_font_props("Helvetica-Oblique", {"italic": 1}, "斜体テスト")
        assert props.fontfile == bundled_font_path()
        assert props.is_italic is True

    def test_cjk_mono_degrades_to_regular(self) -> None:
        """No monospaced CJK variant is bundled; mono degrades to Regular."""
        modifier = self._modifier()
        props = modifier._resolve_font_props("Courier", {"mono": 1}, "等幅テスト")
        assert props.fontfile == bundled_font_path()

    def test_supplementary_plane_kanji_routes(self) -> None:
        """Extension B kanji (e.g. 𠮷) also trigger routing."""
        modifier = self._modifier()
        props = modifier._resolve_font_props("Helvetica", None, "𠮷野家")
        assert props.fontfile == bundled_font_path()

    def test_non_cjk_text_keeps_base14(self) -> None:
        modifier = self._modifier()
        props = modifier._resolve_font_props("Helvetica", None, "Goodbye")
        assert props.fontfile is None
        assert props.fontname == "helv"

    def test_custom_font_takes_precedence(self, tmp_path: Path) -> None:
        # FontResolver validates the file exists with a .ttf suffix; reuse the bundled bytes.
        custom = tmp_path / "custom.ttf"
        custom.write_bytes(Path(bundled_font_path()).read_bytes())

        modifier = self._modifier(custom_fonts={"helv": str(custom)})
        props = modifier._resolve_font_props("Helvetica", None, "こんにちは")
        assert props.fontfile == str(custom)

    def test_missing_bundled_font_falls_back_with_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pdf_modifier.core.modifier as modifier_module

        def _raise(*, bold: bool = False) -> str:
            raise FileNotFoundError("bundled font resource not found")

        monkeypatch.setattr(modifier_module, "bundled_font_path", _raise)
        modifier = self._modifier()
        props = modifier._resolve_font_props("Helvetica", None, "こんにちは")
        assert props.fontfile is None
        assert props.fontname == "helv"
        assert any("bundled CJK font unavailable" in w for w in modifier._warnings)


class TestCjkEndToEnd:
    """End-to-end: Japanese replacement text renders via the bundled font."""

    def test_japanese_replacement_embeds_bundled_font(self, tmp_path: Path) -> None:
        source = _create_test_pdf(tmp_path, "Hello World")
        output = tmp_path / "output.pdf"

        modifier = PDFModifier(str(source), str(output))
        spec = ReplacementSpec(replacements={"Hello World": "こんにちは世界"})
        result = modifier.process(spec)

        assert result.success is True
        assert result.replacements_made == 1

        doc = fitz.open(str(output))
        text = doc[0].get_text("text")
        fonts = doc[0].get_fonts(full=True)
        doc.close()

        assert "こんにちは世界" in text
        assert "Hello World" not in text
        # Custom embedded TTFs appear as Type0; Base 14 fonts are Type1.
        assert any(f[2] == "Type0" for f in fonts), f"expected embedded font, got: {fonts}"

    def test_english_replacement_unaffected(self, tmp_path: Path) -> None:
        source = _create_test_pdf(tmp_path, "Hello World")
        output = tmp_path / "output.pdf"

        modifier = PDFModifier(str(source), str(output))
        spec = ReplacementSpec(replacements={"Hello World": "Goodbye Moon"})
        result = modifier.process(spec)

        assert result.success is True
        doc = fitz.open(str(output))
        text = doc[0].get_text("text")
        fonts = doc[0].get_fonts(full=True)
        doc.close()

        assert "Goodbye Moon" in text
        # No CJK → no bundled font embedding (no Type0 fonts).
        assert not any(f[2] == "Type0" for f in fonts)
