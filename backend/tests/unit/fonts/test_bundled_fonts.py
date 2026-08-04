"""Tests for the bundled Noto Sans JP fonts package."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from pdf_modifier import fonts
from pdf_modifier.fonts import bundled_font_path, contains_cjk


class TestContainsCjk:
    """Tests for CJK character detection."""

    @pytest.mark.parametrize(
        "text",
        [
            "こんにちは",  # Hiragana
            "カタカナ",  # Katakana
            "漢字",  # CJK unified ideographs
            "、。「」",  # CJK punctuation
            "ＡＢＣ１２３",  # Full-width forms
            "ｱｲｳ",  # Half-width Katakana
            "㐂",  # Extension A (U+3402)
            "𠮷野家",  # Supplementary plane, extension B (U+20BB7)
            "㈱",  # Enclosed CJK (U+3231)
            "㌔",  # CJK compatibility (U+3314)
            "mixed 日本語 text",  # Mixed Latin and CJK
        ],
    )
    def test_detects_cjk(self, text: str) -> None:
        assert contains_cjk(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "Hello World",
            "123 !@# äöü éàç",
            "Ω α β γ",  # Greek is not CJK
            # Hangul is deliberately excluded: the bundled Japanese font
            # has no Hangul glyphs, so routing would not help.
            "한국어",
        ],
    )
    def test_rejects_non_cjk(self, text: str) -> None:
        assert contains_cjk(text) is False

    def test_ranges_match_bundled_font_coverage(self) -> None:
        """Representative characters of each detected range have glyphs."""
        font = fitz.Font(fontfile=bundled_font_path())
        representatives = "あア日。Ａｱ㊤㌔〆𠮷"
        for char in representatives:
            assert contains_cjk(char), f"not detected: {char!r}"
            assert font.has_glyph(ord(char)), f"no glyph for: {char!r}"


class TestBundledFontPath:
    """Tests for the bundled font path accessor."""

    def test_regular_font_exists(self) -> None:
        path = Path(bundled_font_path())
        assert path.is_file()
        assert path.suffix == ".ttf"
        assert "Regular" in path.name

    def test_bold_font_exists(self) -> None:
        path = Path(bundled_font_path(bold=True))
        assert path.is_file()
        assert "Bold" in path.name

    def test_regular_and_bold_differ(self) -> None:
        assert bundled_font_path() != bundled_font_path(bold=True)

    def test_path_is_cached(self) -> None:
        assert bundled_font_path() is bundled_font_path()

    def test_missing_resource_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fonts, "_REGULAR", "Missing.ttf")
        with pytest.raises(FileNotFoundError, match="Missing.ttf"):
            bundled_font_path.__wrapped__(bold=False)

    def test_font_name_metadata(self) -> None:
        """Bundled instances carry correct name tables and bold flags."""
        regular = fitz.Font(fontfile=bundled_font_path())
        bold = fitz.Font(fontfile=bundled_font_path(bold=True))
        assert regular.name == "Noto Sans JP Regular"
        assert not regular.is_bold
        assert bold.name == "Noto Sans JP Bold"
        assert bold.is_bold

    def test_license_bundled(self) -> None:
        license_path = Path(bundled_font_path()).parent / "OFL.txt"
        assert license_path.is_file()
        assert "SIL OPEN FONT LICENSE" in license_path.read_text()
