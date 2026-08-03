"""Tests for CLI redaction fill parsing."""

from __future__ import annotations

import pytest
import typer

from pdf_modifier.interfaces.cli import _parse_redact_fill


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, (1.0, 1.0, 1.0)),
        ("0.9,0.8,0.7", (0.9, 0.8, 0.7)),
        ("none", None),
        ("NONE", None),
    ],
)
def test_parse_redact_fill_valid_values(
    value: str | None, expected: tuple[float, float, float] | None
) -> None:
    assert _parse_redact_fill(value) == expected


@pytest.mark.parametrize("value", ["red", "0.1,0.2", "0.0,0.5,1.1", "nan,0,0"])
def test_parse_redact_fill_invalid_values_raise_exit(value: str) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        _parse_redact_fill(value)

    assert exc_info.value.exit_code == 1
