"""Bundled CJK fonts (Noto Sans JP, SIL OFL 1.1 — see OFL.txt).

Provides filesystem paths to the bundled Noto Sans JP static TTF instances
and a helper to detect CJK text that Base 14 fonts cannot render.

Only Regular and Bold weights are bundled: italic and monospaced styles
are not available for CJK and degrade to the upright proportional design.
"""

from __future__ import annotations

import atexit
import threading
from contextlib import ExitStack
from functools import cache
from importlib.resources import as_file, files

_REGULAR = "NotoSansJP-Regular.ttf"
_BOLD = "NotoSansJP-Bold.ttf"

# Keeps resources extracted by as_file() alive for the process lifetime,
# so the returned paths stay valid even under zipimport-style installs.
# The lock prevents concurrent first calls from double-extracting.
_resource_stack = ExitStack()
_resource_lock = threading.Lock()
atexit.register(_resource_stack.close)

# Unicode ranges the bundled Japanese font is expected to cover (verified
# against its cmap). Japanese-oriented: Hangul is deliberately excluded
# because Noto Sans JP has no Hangul glyphs. Han ranges marked "partial"
# follow Adobe-Japan1 coverage — common JIS ideographs are present.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3000, 0x303F),  # CJK symbols and punctuation
    (0x3040, 0x30FF),  # Hiragana and Katakana
    (0x3190, 0x319F),  # Kanbun
    (0x31C0, 0x31EF),  # CJK strokes
    (0x31F0, 0x31FF),  # Katakana phonetic extensions
    (0x3200, 0x33FF),  # Enclosed CJK letters and CJK compatibility
    (0x3400, 0x4DBF),  # CJK unified ideographs extension A (partial)
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xFE10, 0xFE1F),  # Vertical forms
    (0xFE30, 0xFE4F),  # CJK compatibility forms
    (0xFF00, 0xFFEF),  # Half-width and full-width forms
    (0x20000, 0x2FA1F),  # CJK extensions B+ / compatibility supplement (partial)
)


def contains_cjk(text: str) -> bool:
    """Return True if the text contains a character the bundled font targets.

    Detection is Japanese-oriented: kana, Han ideographs (including
    supplementary-plane extensions), CJK punctuation, and full/half-width
    forms. Hangul-only text returns False — the bundled font cannot
    render it, so routing would not help.
    """
    return any(
        start <= ord(char) <= end for char in text for start, end in _CJK_RANGES
    )


@cache
def bundled_font_path(*, bold: bool = False) -> str:
    """Return the filesystem path to a bundled Noto Sans JP TTF.

    Args:
        bold: Return the Bold instance instead of Regular.

    Returns:
        Absolute path to the TTF file, valid for the process lifetime.

    Raises:
        FileNotFoundError: If the bundled font resource is missing.
    """
    name = _BOLD if bold else _REGULAR
    resource = files(__package__).joinpath(name)
    if not resource.is_file():
        raise FileNotFoundError(f"bundled font resource not found: {name}")
    with _resource_lock:
        return str(_resource_stack.enter_context(as_file(resource)))
