"""분기별 재무 데이터 캐싱 모듈.

새 분기 보고서 발표일이 지나면 자동으로 캐시가 무효화됩니다.
한국 기준 발표 일정: 3/15(Q4결산), 5/15(Q1), 8/14(Q2), 11/14(Q3)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / "cache"


def _current_quarter(d: date | None = None) -> str:
    """현재 분기 키. 예: '2025Q2'"""
    if d is None:
        d = date.today()
    q = (d.month - 1) // 3 + 1
    return f"{d.year}Q{q}"


def _next_report_date(today: date | None = None) -> date:
    """다음 분기 재무제표 발표 예정일 (한국 기준)"""
    if today is None:
        today = date.today()
    schedule = [(3, 15), (5, 15), (8, 14), (11, 14)]
    for month, day in schedule:
        candidate = date(today.year, month, day)
        if candidate > today:
            return candidate
    return date(today.year + 1, 3, 15)


def _safe_key(ticker: str) -> str:
    return ticker.replace(".", "_").replace("/", "_")


# ── DataFrame / Series 직렬화 ──────────────────

def _df_to_json(df: pd.DataFrame) -> dict | None:
    if df is None or df.empty:
        return None
    try:
        return json.loads(df.to_json(orient="split", date_format="iso", default_handler=str))
    except Exception:
        return None


def _json_to_df(d: dict | None) -> pd.DataFrame:
    if not d:
        return pd.DataFrame()
    try:
        df = pd.DataFrame(
            data=d.get("data", []),
            index=d.get("index", []),
            columns=d.get("columns", []),
        )
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
        try:
            df.columns = pd.to_datetime(df.columns)
        except Exception:
            pass
        return df
    except Exception:
        return pd.DataFrame()


def _series_to_json(s: pd.Series) -> dict | None:
    if s is None or s.empty:
        return None
    try:
        return {str(k): float(v) for k, v in s.items()}
    except Exception:
        return None


def _json_to_series(d: dict | None) -> pd.Series:
    if not d:
        return pd.Series(dtype=float)
    s = pd.Series({k: v for k, v in d.items()})
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        pass
    return s


# ── 공개 API ───────────────────────────────────

def is_valid(ticker: str) -> bool:
    """캐시가 현재 분기 내에 유효한지 확인"""
    path = CACHE_DIR / f"{_safe_key(ticker)}.json"
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("quarter") == _current_quarter()
    except Exception:
        return False


def load(ticker: str) -> Optional[Dict[str, Any]]:
    """캐시 로드. 성공 시 메타 + data dict 반환."""
    path = CACHE_DIR / f"{_safe_key(ticker)}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        d = raw["data"]
        d["financials"]    = _json_to_df(d.get("financials"))
        d["price_history"] = _json_to_df(d.get("price_history"))
        d["peer_df"]       = _json_to_df(d.get("peer_df"))
        d["profit_margins"] = _json_to_series(d.get("profit_margins"))
        for key in ("stock", "sp500"):
            if key in d.get("dca_results", {}):
                d["dca_results"][key]["prices"] = _json_to_series(
                    d["dca_results"][key].get("prices")
                )
        return {
            "cached_at":    raw.get("cached_at", ""),
            "quarter":      raw.get("quarter", ""),
            "next_refresh": raw.get("next_refresh", ""),
            "data": d,
        }
    except Exception:
        return None


def save(ticker: str, data: Dict[str, Any]) -> None:
    """분석 결과를 분기 캐시에 저장"""
    CACHE_DIR.mkdir(exist_ok=True)
    today = date.today()

    serializable: Dict[str, Any] = {k: v for k, v in data.items()}
    serializable["financials"]    = _df_to_json(data.get("financials", pd.DataFrame()))
    serializable["price_history"] = _df_to_json(data.get("price_history", pd.DataFrame()))
    serializable["peer_df"]       = _df_to_json(data.get("peer_df", pd.DataFrame()))
    serializable["profit_margins"] = _series_to_json(
        data.get("profit_margins", pd.Series(dtype=float))
    )

    dca: Dict[str, Any] = {}
    for key, entry in data.get("dca_results", {}).items():
        e = {k: v for k, v in entry.items()}
        e["prices"] = _series_to_json(entry.get("prices", pd.Series(dtype=float)))
        dca[key] = e
    serializable["dca_results"] = dca

    payload = {
        "ticker":       ticker,
        "cached_at":    today.isoformat(),
        "quarter":      _current_quarter(today),
        "next_refresh": _next_report_date(today).isoformat(),
        "data":         serializable,
    }
    path = CACHE_DIR / f"{_safe_key(ticker)}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
