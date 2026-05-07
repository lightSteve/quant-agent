"""Formatting utilities for financial figures."""

from __future__ import annotations

from typing import Optional


def fmt_number(value: Optional[float], currency: str = "") -> str:
    """Format large financial figures using Korean units (조, 억, 만)."""
    if value is None:
        return "N/A"
    prefix = currency + " " if currency else ""
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}{prefix}{abs_val / 1e12:.2f}조"
    if abs_val >= 1e8:
        return f"{sign}{prefix}{abs_val / 1e8:.1f}억"
    if abs_val >= 1e4:
        return f"{sign}{prefix}{abs_val / 1e4:.1f}만"
    return f"{sign}{prefix}{abs_val:,.0f}"


def fmt_usd(value: Optional[float]) -> str:
    """Format a dollar amount."""
    if value is None:
        return "N/A"
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:.1f}K"
    return f"${value:.2f}"


def fmt_pct(value: Optional[float], already_pct: bool = False) -> str:
    """Format as percentage string."""
    if value is None:
        return "N/A"
    pct = value if already_pct else value * 100
    return f"{pct:+.2f}%" if pct != 0 else "0.00%"


def fmt_ratio(value: Optional[float], decimals: int = 2) -> str:
    """Format a ratio (PER / PBR / etc.)."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}배"


def score_color(score: float, max_score: float = 10.0) -> str:
    """Return a hex color based on score fraction."""
    frac = score / max_score
    if frac >= 0.7:
        return "#00c853"   # green
    if frac >= 0.5:
        return "#ffd600"   # amber
    return "#d50000"       # red
