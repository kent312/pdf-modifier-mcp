"""PDF text replacement engine with style preservation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import fitz

from ..fonts import bundled_font_path, contains_cjk
from ..logger import setup_logging
from .exceptions import (
    FileSizeExceededError,
    PDFModifierError,
    PDFNotFoundError,
    PDFPasswordError,
    PDFReadError,
)
from .font_resolver import FontResolver
from .models import BatchResult, FontProperties, ModificationResult, ReplacementSpec

logger = setup_logging(__name__)

DEFAULT_MAX_FILE_SIZE_BYTES: int = 100 * 1024 * 1024  # 100 MB

# Lower bound for fit_text shrinking: below this the text is unreadable,
# so fitting clamps here and warns that the replacement may overflow.
MIN_FIT_FONTSIZE: float = 4.0


class PDFModifier:
    """
    PDF text replacement engine with style preservation.

    Supports:
    - Exact text matching
    - Regex pattern matching
    - Hyperlink creation (text|URL syntax)
    - Font style preservation (Base 14 fonts)

    Example:
        >>> spec = ReplacementSpec(replacements={"old": "new"})
        >>> modifier = PDFModifier("input.pdf", "output.pdf")
        >>> result = modifier.process(spec)
        >>> print(result.replacements_made)

        # Or with context manager:
        >>> with PDFModifier("input.pdf", "output.pdf") as modifier:
        ...     result = modifier.process(spec)
    """

    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
        password: str | None = None,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        custom_fonts: dict[str, str] | None = None,
        redact_fill: tuple[float, float, float] | None = (1.0, 1.0, 1.0),
        fit_text: bool = False,
    ) -> None:
        self.input_path = Path(input_path).absolute()
        self.output_path = Path(output_path).absolute()
        self.password = password
        self.max_file_size = max_file_size
        self._custom_fonts = self._validate_custom_fonts(custom_fonts or {})
        self._redact_fill = self._validate_redact_fill(redact_fill)
        self._fit_text = fit_text
        self._font_resolver = FontResolver()
        self._font_cache: dict[str, fitz.Font] = {}

        if self.input_path == self.output_path:
            raise ValueError("Input and output paths cannot be the same. Risk of file corruption.")

        self._doc: fitz.Document | None = None
        self._warnings: list[str] = []
        self._failed_insertions = 0

    @staticmethod
    def _parse_flags(raw_flags: int | dict[str, int] | None) -> dict[str, int] | None:
        """Convert PyMuPDF int flags to dict, or pass through if already dict."""
        if raw_flags is None:
            return None
        if isinstance(raw_flags, dict):
            return raw_flags
        # PyMuPDF int flags bitmask:
        # TEXT_FONT_BOLD=1, TEXT_FONT_ITALIC=2, TEXT_FONT_SERIFED=4,
        # TEXT_FONT_MONOSPACED=8, TEXT_FONT_SUPERSCRIPT=16
        flags: dict[str, int] = {}
        if raw_flags & 1:
            flags["bold"] = 1
        if raw_flags & 2:
            flags["italic"] = 1
        if raw_flags & 4:
            flags["serif"] = 1
        if raw_flags & 8:
            flags["mono"] = 1
        return flags

    @staticmethod
    def _validate_custom_fonts(custom_fonts: dict[str, str]) -> dict[str, str]:
        """Validate that all custom font files exist and are valid TTF/OTF."""
        for _alias, path in custom_fonts.items():
            p = Path(path)
            if not p.is_file():
                raise ValueError(f"font file does not exist: {path}")
            if p.suffix.lower() not in (".ttf", ".otf"):
                raise ValueError(f"font file must be .ttf or .otf: {path}")
        return custom_fonts

    @staticmethod
    def _validate_redact_fill(
        redact_fill: tuple[float, float, float] | None,
    ) -> tuple[float, float, float] | None:
        """Validate an optional RGB redaction fill color."""
        if redact_fill is None:
            return None
        if not isinstance(redact_fill, tuple | list) or len(redact_fill) != 3:
            raise ValueError("redact_fill must contain three RGB values between 0.0 and 1.0")
        try:
            components = tuple(float(component) for component in redact_fill)
        except (TypeError, ValueError) as e:
            raise ValueError("redact_fill must contain three RGB values between 0.0 and 1.0") from e
        if any(not 0.0 <= component <= 1.0 for component in components):
            raise ValueError("redact_fill must contain three RGB values between 0.0 and 1.0")
        return (components[0], components[1], components[2])

    def _open_doc(self) -> fitz.Document:
        """Safely open the document with password authentication if required."""
        if not self.input_path.exists():
            raise PDFNotFoundError(
                f"PDF file not found: {self.input_path}",
                {"path": str(self.input_path)},
            )

        file_size = self.input_path.stat().st_size
        if file_size > self.max_file_size:
            size_mb = file_size / (1024 * 1024)
            limit_mb = self.max_file_size / (1024 * 1024)
            raise FileSizeExceededError(
                f"PDF file is {size_mb:.1f} MB, exceeds limit of {limit_mb:.0f} MB",
                {
                    "path": str(self.input_path),
                    "size_bytes": file_size,
                    "limit_bytes": self.max_file_size,
                },
            )

        try:
            doc = fitz.open(self.input_path)
            if doc.needs_pass:
                if not self.password:
                    raise PDFPasswordError("PDF is password protected. Please provide a password.")
                if not doc.authenticate(self.password):
                    raise PDFPasswordError("Incorrect password provided for the PDF.")
            return doc
        except PDFPasswordError:
            raise
        except Exception as e:
            raise PDFReadError(f"Cannot open PDF: {e}", {"path": str(self.input_path)}) from e

    def __enter__(self) -> PDFModifier:
        self._doc = self._open_doc()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Explicitly close the document."""
        if self._doc:
            self._doc.close()
            self._doc = None

    def _load_font(self, fontname: str, fontfile: str | None) -> fitz.Font:
        """Load (and cache) a fitz.Font for measurement and preflight."""
        key = fontfile or fontname
        font = self._font_cache.get(key)
        if font is None:
            font = fitz.Font(fontname=fontname, fontfile=fontfile)
            self._font_cache[key] = font
        return font

    def _preflight_fonts(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop items whose custom font file cannot be loaded.

        Runs before redaction so a broken font never destroys original text:
        skipped items keep their original content and produce a warning.
        """
        valid: list[dict[str, Any]] = []
        for item in items:
            fontfile = item.get("fontfile")
            if fontfile:
                try:
                    self._load_font(item["fontname"], fontfile)
                except Exception as e:
                    msg = (
                        f"Skipping replacement '{item['text']}': "
                        f"cannot load font file {fontfile}: {e}"
                    )
                    logger.warning(msg)
                    self._warnings.append(msg)
                    continue
            valid.append(item)
        return valid

    def _apply_replacements_to_page(
        self,
        page: fitz.Page,
        items: list[dict[str, Any]],
    ) -> tuple[int, bool]:
        """Apply replacements to a single page.

        Returns:
            Tuple of (successful insertions, whether the page was touched).
        """
        items = self._preflight_fonts(items)
        if not items:
            return 0, False

        for item in items:
            if "bboxes" in item:
                for bbox in item["bboxes"]:
                    page.add_redact_annot(
                        bbox,
                        fill=self._redact_fill if self._redact_fill is not None else False,
                    )
            else:
                page.add_redact_annot(
                    item["bbox"],
                    fill=self._redact_fill if self._redact_fill is not None else False,
                )

        page.apply_redactions()

        inserted = sum(1 for item in items if self._insert_replacement(page, item))
        return inserted, True

    def _process_pages(
        self,
        spec: ReplacementSpec,
        pages: tuple[int, int] | None = None,
    ) -> tuple[int, set[int]]:
        """Process pages and return (total_replacements, pages_modified).

        Args:
            spec: Replacement specification.
            pages: Optional (start, end) 1-indexed inclusive range.
                   None processes all pages.

        Raises:
            ValueError: If page range is invalid or out of bounds.
        """
        doc = self._doc
        assert doc is not None, "Document must be opened before processing"
        total_pages = len(doc)

        if pages is not None:
            if not pages:
                raise ValueError("Page range must have at least one page")
            if len(pages) != 2:
                raise ValueError("Page range must be a (start, end) tuple")
            start, end = pages
            if start < 1 or end < 1:
                raise ValueError("Page numbers must be 1-indexed")
            if start > end:
                raise ValueError("Page range start must not exceed end")
            if end > total_pages:
                raise ValueError(f"Page {end} exceeds document total of {total_pages} pages")
            page_indices = range(start - 1, end)
        else:
            page_indices = range(total_pages)

        total = 0
        pages_modified: set[int] = set()

        for page_num in page_indices:
            page = doc[page_num]
            items = self._collect_replacements(page, spec)
            if not items:
                continue
            inserted, touched = self._apply_replacements_to_page(page, items)
            if touched:
                pages_modified.add(page_num)
            total += inserted

        return total, pages_modified

    def _save_and_log(self) -> None:
        """Save the modified document and log the result."""
        self._doc.save(str(self.output_path))  # type: ignore[union-attr]
        logger.info("Saved %s", self.output_path)

    def process(
        self,
        spec: ReplacementSpec,
        pages: tuple[int, int] | None = None,
    ) -> ModificationResult:
        """
        Execute all replacements and return structured result.

        Batches redactions per page for efficiency.

        Args:
            spec: ReplacementSpec containing replacements and options.
            pages: Optional (start, end) 1-indexed inclusive page range.
                   None processes all pages.

        Returns:
            ModificationResult with success status and statistics.

        Raises:
            PDFReadError: If the PDF cannot be opened.
            ValueError: If page range is invalid.
        """
        doc_opened_here = False
        self._failed_insertions = 0
        if not self._doc:
            self._doc = self._open_doc()
            doc_opened_here = True

        try:
            total, pages_modified = self._process_pages(spec, pages)
            self._save_and_log()
        except ValueError:
            raise
        except PDFModifierError:
            raise
        except Exception as e:
            raise PDFReadError(f"Failed to process PDF pages: {e}") from e
        finally:
            if doc_opened_here:
                self.close()

        return ModificationResult(
            success=True,
            input_path=str(self.input_path),
            output_path=str(self.output_path),
            replacements_made=total,
            pages_modified=len(pages_modified),
            failed_insertions=self._failed_insertions,
            warnings=self._warnings,
        )

    def _get_font_properties(self, font_name: str) -> tuple[str, str]:
        """
        Map PDF font names to PyMuPDF Base 14 font codes.

        Returns:
            Tuple of (font_code for insert_text, font_name for width calculation)
        """
        name_lower = font_name.lower()

        if "courier" in name_lower:
            if "bold" in name_lower:
                return ("CoBo", "Courier-Bold")
            return ("Cour", "Courier")
        elif "times" in name_lower or "serif" in name_lower:
            if "bold" in name_lower:
                return ("TiBo", "Times-Bold")
            return ("TiRo", "Times-Roman")
        elif "bold" in name_lower:
            return ("HeBo", "Helvetica-Bold")
        return ("helv", "Helvetica")

    def _convert_color(
        self, color_input: int | list[float] | tuple[float, ...]
    ) -> tuple[float, float, float]:
        """Convert PyMuPDF color to RGB float tuple (0.0-1.0)."""
        if isinstance(color_input, int):
            r = ((color_input >> 16) & 0xFF) / 255.0
            g = ((color_input >> 8) & 0xFF) / 255.0
            b = (color_input & 0xFF) / 255.0
            return (r, g, b)
        elif isinstance(color_input, list | tuple) and len(color_input) >= 3:
            return tuple(c if c <= 1.0 else c / 255.0 for c in color_input[:3])  # type: ignore[return-value]
        return (0.0, 0.0, 0.0)

    def _resolve_replacement(
        self,
        replacement_raw: str,
        matched_text: str,
        target: str,
        spec: ReplacementSpec,
    ) -> tuple[str, str | None]:
        """Parse replacement text and URL from raw replacement string."""
        replacement_text = replacement_raw
        url: str | None = None

        if "|" in replacement_raw:
            candidate_text, candidate_url = replacement_raw.rsplit("|", 1)
            candidate_url = candidate_url.strip()
            if candidate_url == "void(0)" or candidate_url.startswith(
                ("http://", "https://", "mailto:", "javascript:")
            ):
                replacement_text = candidate_text
                url = "javascript:void(0)" if candidate_url == "void(0)" else candidate_url

        if spec.use_regex and spec.compiled_patterns:
            pattern = spec.compiled_patterns[target]
            new_text = pattern.sub(replacement_text, matched_text)
        else:
            new_text = matched_text.replace(target, replacement_text)

        return new_text, url

    def _resolve_font_props(
        self,
        font_name: str,
        font_flags: dict[str, int] | None,
        text: str,
    ) -> FontProperties:
        """Resolve font properties, auto-routing CJK text to the bundled font.

        Custom fonts supplied by the caller take precedence. The bundled
        Noto Sans JP is used only when the replacement text contains CJK
        characters and resolution produced no font file (a Base 14 font or
        the built-in Helvetica fallback), which cannot render CJK glyphs.

        Only Regular and Bold are bundled: italic or monospaced styles
        degrade to the upright proportional design for CJK text.
        """
        props = self._font_resolver.resolve(
            font_name,
            font_flags=font_flags,
            custom_fonts=self._custom_fonts,
        )
        if props.fontfile is None and contains_cjk(text):
            try:
                fontfile = bundled_font_path(bold=props.is_bold)
            except FileNotFoundError as e:
                msg = f"CJK replacement text but bundled CJK font unavailable: {e}"
                logger.warning(msg)
                self._warnings.append(msg)
                return props
            return props.model_copy(update={"fontfile": fontfile, "embed": True})
        return props

    def _match_single_span(
        self,
        span: dict[str, Any],
        spec: ReplacementSpec,
    ) -> dict[str, Any] | None:
        """Check if a single span matches any replacement target."""
        text = span["text"].strip()
        original = span["text"]

        for target, replacement_raw in spec.replacements.items():
            match_found = False
            if spec.use_regex and spec.compiled_patterns:
                pattern = spec.compiled_patterns[target]
                if pattern.search(text):
                    match_found = True
            elif target in text:
                match_found = True

            if match_found:
                new_text, url = self._resolve_replacement(replacement_raw, original, target, spec)
                # Convert int flags to dict if needed
                raw_flags = span.get("flags")
                font_flags = self._parse_flags(raw_flags)
                font_props = self._resolve_font_props(span["font"], font_flags, new_text)
                return {
                    "bbox": span["bbox"],
                    "origin": span["origin"],
                    "text": new_text,
                    "url": url,
                    "fontname": font_props.fontname,
                    "fontfile": font_props.fontfile,
                    "size": span["size"],
                    "color": span["color"],
                }
        return None

    def _collect_replacements(
        self,
        page: fitz.Page,
        spec: ReplacementSpec,
    ) -> list[dict[str, Any]]:
        """Scan page and collect items to replace.

        Two-pass approach:
        1. Single-span matching (fast path for most cases).
        2. Cross-span matching per line.
        """
        items: list[dict[str, Any]] = []
        matched_span_ids: set[int] = set()
        blocks = page.get_text("dict")["blocks"]

        # Pass 1: single-span matching
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    item = self._match_single_span(span, spec)
                    if item:
                        items.append(item)
                        matched_span_ids.add(id(span))
                        break

        # Pass 2: cross-span matching
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                if len(line["spans"]) < 2:
                    continue
                items.extend(self._match_across_spans(line["spans"], spec, matched_span_ids))

        return items

    def _build_merged_text(
        self,
        spans: list[dict[str, Any]],
    ) -> tuple[str, list[tuple[int, int]]]:
        """Build concatenated text and offset map from spans."""
        merged = ""
        span_ranges: list[tuple[int, int]] = []
        for span in spans:
            start = len(merged)
            merged += span["text"]
            span_ranges.append((start, len(merged)))
        return merged, span_ranges

    def _find_matches_in_merged(
        self,
        merged: str,
        target: str,
        spec: ReplacementSpec,
    ) -> list[tuple[int, int]]:
        """Find all match positions in merged text."""
        matches: list[tuple[int, int]] = []
        if spec.use_regex and spec.compiled_patterns:
            pattern = spec.compiled_patterns[target]
            for m in pattern.finditer(merged):
                matches.append((m.start(), m.end()))
        else:
            start = 0
            while True:
                idx = merged.find(target, start)
                if idx == -1:
                    break
                matches.append((idx, idx + len(target)))
                start = idx + 1
        return matches

    def _build_cross_span_item(
        self,
        spans: list[dict[str, Any]],
        involved: list[int],
        merged: str,
        m_start: int,
        m_end: int,
        replacement_raw: str,
        target: str,
        spec: ReplacementSpec,
    ) -> dict[str, Any]:
        """Build a replacement item for a cross-span match."""
        first_span = spans[involved[0]]
        bboxes = [tuple(spans[i]["bbox"]) for i in involved]
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)

        matched_text = merged[m_start:m_end]
        new_text, url = self._resolve_replacement(replacement_raw, matched_text, target, spec)
        raw_flags = first_span.get("flags")
        font_flags = self._parse_flags(raw_flags)
        font_props = self._resolve_font_props(first_span["font"], font_flags, new_text)

        return {
            "bbox": (x0, y0, x1, y1),
            "bboxes": bboxes,
            "origin": first_span["origin"],
            "text": new_text,
            "url": url,
            "fontname": font_props.fontname,
            "fontfile": font_props.fontfile,
            "size": first_span["size"],
            "color": first_span["color"],
        }

    def _match_across_spans(
        self,
        spans: list[dict[str, Any]],
        spec: ReplacementSpec,
        matched_span_ids: set[int],
    ) -> list[dict[str, Any]]:
        """Match replacement targets across concatenated span texts."""
        items: list[dict[str, Any]] = []

        merged, span_ranges = self._build_merged_text(spans)
        if not merged.strip():
            return items

        for target, replacement_raw in spec.replacements.items():
            matches = self._find_matches_in_merged(merged, target, spec)

            for m_start, m_end in matches:
                involved = [
                    i
                    for i, (s_start, s_end) in enumerate(span_ranges)
                    if s_start < m_end and s_end > m_start
                ]

                if len(involved) < 2:
                    continue
                if any(id(spans[i]) in matched_span_ids for i in involved):
                    continue

                item = self._build_cross_span_item(
                    spans,
                    involved,
                    merged,
                    m_start,
                    m_end,
                    replacement_raw,
                    target,
                    spec,
                )
                items.append(item)

                for i in involved:
                    matched_span_ids.add(id(spans[i]))

        return items

    def _insert_link(
        self,
        page: fitz.Page,
        item: dict[str, Any],
        link_url: str,
    ) -> None:
        """Insert a hyperlink for the replacement text."""
        try:
            fontname = item["fontname"]
            fontfile = item.get("fontfile")
            if fontfile:
                fontname = f"__custom_{Path(fontfile).stem}__"
            font = fitz.Font(fontname=fontname, fontfile=fontfile)
            text_width = font.text_length(item["text"], fontsize=item["size"])
            x0 = item["origin"][0]
            y_baseline = item["origin"][1]
            link_rect = fitz.Rect(
                x0,
                y_baseline - item["size"],
                x0 + text_width,
                y_baseline + (item["size"] * 0.25),
            )
            page.insert_link({"kind": fitz.LINK_URI, "from": link_rect, "uri": link_url})
        except Exception as e:
            msg = f"Could not add link for '{item['text']}': {e}"
            logger.warning(msg)
            self._warnings.append(msg)

    def _fit_fontsize(self, item: dict[str, Any]) -> float:
        """Shrink the font size so the replacement text fits the original bbox.

        The shrink is clamped at ``MIN_FIT_FONTSIZE``; when the clamp kicks
        in a warning is recorded because the text may still overflow.
        For cross-span matches the reference width is the merged bbox of the
        involved spans, which can be wider than the exact matched region.
        """
        size: float = item["size"]
        bbox = item["bbox"]
        available: float = float(bbox[2]) - float(bbox[0])
        if available <= 0:
            logger.debug("Degenerate bbox for '%s'; skipping fit", item["text"])
            return size
        try:
            font = self._load_font(item["fontname"], item.get("fontfile"))
            width = float(font.text_length(item["text"], fontsize=size))
        except Exception as e:
            logger.debug("Could not measure text for fit: %s", e)
            return size
        if width <= available:
            return size
        fitted = size * available / width
        if fitted < MIN_FIT_FONTSIZE:
            msg = (
                f"Replacement '{item['text']}' needs fontsize {fitted:.2f} to fit; "
                f"clamped to {MIN_FIT_FONTSIZE} — text may overflow its region."
            )
            logger.warning(msg)
            self._warnings.append(msg)
            fitted = MIN_FIT_FONTSIZE if size >= MIN_FIT_FONTSIZE else size
        logger.debug("Fitting '%s': fontsize %.2f -> %.2f", item["text"], size, fitted)
        return fitted

    def _insert_replacement(self, page: fitz.Page, item: dict[str, Any]) -> bool:
        """Insert replacement text with original styling.

        Returns:
            True if the text was inserted, False if insertion failed
            (a warning is recorded; the original text is already redacted).
        """
        color = self._convert_color(item["color"])
        fontname = item["fontname"]
        fontfile = item.get("fontfile")

        if self._fit_text:
            item["size"] = self._fit_fontsize(item)

        # PyMuPDF ignores fontfile when fontname is a Base 14 font.
        # When a custom fontfile is provided, use a non-Base14 fontname
        # so PyMuPDF treats it as a custom embedded font.
        if fontfile:
            fontname = f"__custom_{Path(fontfile).stem}__"

        try:
            page.insert_text(
                item["origin"],
                item["text"],
                fontname=fontname,
                fontsize=item["size"],
                color=color,
                fontfile=fontfile,
            )
        except Exception as e:
            msg = (
                f"Could not insert replacement '{item['text']}': {e}. "
                "The original text was already redacted, leaving a blank region."
            )
            logger.warning(msg)
            self._warnings.append(msg)
            self._failed_insertions += 1
            return False

        link_url = item["url"]
        if not link_url and item["text"].strip().startswith(("http://", "https://")):
            link_url = item["text"].strip()
        if link_url:
            self._insert_link(page, item, link_url)
        return True


def batch_process(
    file_paths: Sequence[str | Path],
    output_dir: str | Path,
    spec: ReplacementSpec,
    password: str | None = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    custom_fonts: dict[str, str] | None = None,
    redact_fill: tuple[float, float, float] | None = (1.0, 1.0, 1.0),
    fit_text: bool = False,
) -> BatchResult:
    """
    Apply the same replacements to multiple PDF files.

    Each file is processed independently; failures in one file do not
    affect the rest of the batch. Output files are written to
    ``output_dir`` using the same filename as the input.

    Args:
        file_paths: List of input PDF file paths.
        output_dir: Directory where modified PDFs will be saved.
        spec: ReplacementSpec containing replacements and options.
        password: Optional password for encrypted PDFs.
        custom_fonts: Optional map of alias -> font file path.
        redact_fill: Optional RGB redaction fill. None disables the fill.
                     Defaults to white (1.0, 1.0, 1.0).
        fit_text: If True, shrink replacement text to fit the original bbox width.

    Returns:
        BatchResult with per-file results and aggregate statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ModificationResult] = []
    errors: list[dict[str, str]] = []

    for file_path in file_paths:
        file_path = Path(file_path)
        output_path = output_dir / file_path.name

        if file_path.absolute() == output_path.absolute():
            errors.append({"file": str(file_path), "error": "Input and output paths are the same"})
            continue

        try:
            modifier = PDFModifier(
                str(file_path),
                str(output_path),
                password=password,
                max_file_size=max_file_size,
                custom_fonts=custom_fonts,
                redact_fill=redact_fill,
                fit_text=fit_text,
            )
            result = modifier.process(spec)
            results.append(result)
        except Exception as e:
            logger.warning("Batch: failed to process %s: %s", file_path, e)
            error_code = getattr(e, "code", "UNKNOWN")
            errors.append({"file": str(file_path), "error": f"[{error_code}] {e}"})

    return BatchResult(
        total_files=len(file_paths),
        successful=len(results),
        failed=len(errors),
        results=results,
        errors=errors,
    )
