"""
SQLite 영속 저장소 — 섹터별 전수 분석 결과.

테이블:
  stock_queue      : 분석 대기열 (pending → in_progress → done)
  sector_analysis  : 완료된 AI 분석 결과
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Streamlit Cloud: /mount/src/ 는 read-only → /tmp/ 사용
# 로컬 개발: 프로젝트 루트에 저장
if os.path.exists("/mount/src"):
    DB_PATH = "/tmp/sector_analysis.db"
else:
    _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH = os.path.join(_BASE, "sector_analysis.db")

_write_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    """테이블이 없으면 생성."""
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS stock_queue (
                ticker       TEXT PRIMARY KEY,
                company_name TEXT,
                sector       TEXT,
                market       TEXT,
                priority     REAL DEFAULT 999.0,
                status       TEXT DEFAULT 'pending',
                added_at     TEXT
            );

            CREATE TABLE IF NOT EXISTS sector_analysis (
                ticker             TEXT PRIMARY KEY,
                company_name       TEXT,
                sector             TEXT,
                sector_kr          TEXT,
                market             TEXT,
                per                REAL,
                pbr                REAL,
                profit_margin      REAL,
                revenue_growth     REAL,
                appeal_score       REAL,
                buy_signal         INTEGER DEFAULT 0,
                ai_summary         TEXT,
                key_strength       TEXT,
                key_risk           TEXT,
                investment_horizon TEXT,
                analyzed_at        TEXT
            );
        """)


def add_stocks_to_queue(stocks: List[Dict[str, Any]]) -> int:
    """큐에 종목 추가. 이미 있는 종목(ticker)은 건너뜀. 반환값: 신규 추가 수."""
    added = 0
    now = datetime.now().isoformat()
    with _write_lock:
        with _conn() as conn:
            for s in stocks:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO stock_queue
                        (ticker, company_name, sector, market, priority, status, added_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        s["ticker"],
                        s.get("company_name", ""),
                        s.get("sector", ""),
                        s.get("market", "KOSPI"),
                        s.get("priority", 999.0),
                        now,
                    ),
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    added += 1
    return added


def get_next_pending(reanalysis_days: int = 7) -> Optional[Dict[str, Any]]:
    """
    분석 대기 종목 1개 반환 (우선순위 오름차순).
    이미 분석됐더라도 reanalysis_days 이전이면 재분석 대상.
    """
    cutoff = (datetime.now() - timedelta(days=reanalysis_days)).isoformat()
    with _conn() as conn:
        row = conn.execute(
            """
            SELECT q.ticker, q.company_name, q.sector, q.market
            FROM   stock_queue q
            LEFT JOIN sector_analysis a ON q.ticker = a.ticker
            WHERE  q.status = 'pending'
              AND  (a.ticker IS NULL OR a.analyzed_at < ?)
            ORDER BY q.priority ASC
            LIMIT 1
            """,
            (cutoff,),
        ).fetchone()
        if row:
            return dict(row)
    return None


def mark_in_progress(ticker: str) -> None:
    with _write_lock:
        with _conn() as conn:
            conn.execute(
                "UPDATE stock_queue SET status='in_progress' WHERE ticker=?", (ticker,)
            )


def mark_pending(ticker: str) -> None:
    """오류 발생 시 pending으로 되돌려 재시도 허용."""
    with _write_lock:
        with _conn() as conn:
            conn.execute(
                "UPDATE stock_queue SET status='pending' WHERE ticker=?", (ticker,)
            )


def save_result(result: Dict[str, Any]) -> None:
    """분석 결과 upsert 저장 후 큐 상태 done 처리."""
    now = datetime.now().isoformat()
    with _write_lock:
        with _conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sector_analysis
                    (ticker, company_name, sector, sector_kr, market,
                     per, pbr, profit_margin, revenue_growth,
                     appeal_score, buy_signal, ai_summary,
                     key_strength, key_risk, investment_horizon, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result["ticker"],
                    result.get("company_name", ""),
                    result.get("sector", ""),
                    result.get("sector_kr", ""),
                    result.get("market", "KOSPI"),
                    result.get("per"),
                    result.get("pbr"),
                    result.get("profit_margin"),
                    result.get("revenue_growth"),
                    result.get("appeal_score"),
                    int(bool(result.get("buy_signal", False))),
                    result.get("ai_summary", ""),
                    result.get("key_strength", ""),
                    result.get("key_risk", ""),
                    result.get("investment_horizon", ""),
                    now,
                ),
            )
            conn.execute(
                "UPDATE stock_queue SET status='done' WHERE ticker=?",
                (result["ticker"],),
            )


def get_all_results(sector_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """분석 완료 결과 조회. appeal_score 내림차순."""
    with _conn() as conn:
        if sector_filter and sector_filter != "전체":
            rows = conn.execute(
                """
                SELECT * FROM sector_analysis
                WHERE sector_kr = ? OR sector = ?
                ORDER BY appeal_score DESC NULLS LAST
                """,
                (sector_filter, sector_filter),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sector_analysis ORDER BY appeal_score DESC NULLS LAST"
            ).fetchall()
        return [dict(r) for r in rows]


def get_queue_stats() -> Dict[str, int]:
    """큐 및 분석 현황 통계."""
    with _conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM stock_queue").fetchone()[0]
        done = conn.execute(
            "SELECT COUNT(*) FROM stock_queue WHERE status='done'"
        ).fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM stock_queue WHERE status='pending'"
        ).fetchone()[0]
        analyzed = conn.execute("SELECT COUNT(*) FROM sector_analysis").fetchone()[0]
        buy_signals = conn.execute(
            "SELECT COUNT(*) FROM sector_analysis WHERE buy_signal=1"
        ).fetchone()[0]
    return {
        "total": total,
        "done": done,
        "pending": pending,
        "analyzed": analyzed,
        "buy_signals": buy_signals,
    }


def get_sectors() -> List[str]:
    """분석 결과에서 섹터 목록 추출."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT COALESCE(NULLIF(sector_kr,''), sector) AS s "
            "FROM sector_analysis WHERE s IS NOT NULL AND s != '' ORDER BY s"
        ).fetchall()
        return [r["s"] for r in rows]


def reset_in_progress() -> None:
    """서버 재시작 시 in_progress 종목을 pending으로 복구."""
    with _write_lock:
        with _conn() as conn:
            conn.execute(
                "UPDATE stock_queue SET status='pending' WHERE status='in_progress'"
            )


def reset_all() -> None:
    """큐·결과 전체 초기화 (UI에서 사용자 확인 후 호출)."""
    with _write_lock:
        with _conn() as conn:
            conn.execute("DELETE FROM stock_queue")
            conn.execute("DELETE FROM sector_analysis")
