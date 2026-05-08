"""
영속 저장소 — 섹터별 전수 분석 결과.

백엔드 자동 선택:
  - SUPABASE_DB_URL (또는 DATABASE_URL) 환경변수 있음
      → PostgreSQL (Supabase) — Streamlit Cloud 영구 저장
  - 없음
      → SQLite (로컬 개발, 프로젝트 루트에 sector_analysis.db 저장)

Streamlit Cloud secrets.toml 예시:
  SUPABASE_DB_URL = "postgresql://postgres.[ref]:[password]@..."
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional

# ── 백엔드 감지 ──────────────────────────────────────────────────────────────
_PG_URL: str = (
    os.getenv("SUPABASE_DB_URL", "")
    or os.getenv("DATABASE_URL", "")
)
_USE_PG: bool = bool(_PG_URL)

# Streamlit Cloud 환경 감지 (/mount/src 존재 = 파일시스템 read-only)
_ON_STREAMLIT_CLOUD: bool = os.path.exists("/mount/src")

if _ON_STREAMLIT_CLOUD and not _USE_PG:
    raise EnvironmentError(
        "Streamlit Cloud 배포 환경에서 SUPABASE_DB_URL 또는 DATABASE_URL 환경변수가 설정되지 않았습니다.\n"
        "Streamlit Cloud → Settings → Secrets 에서 SUPABASE_DB_URL을 추가해주세요."
    )

_write_lock = threading.Lock()

# ── SQLite 경로 (로컬 전용) ───────────────────────────────────────────────────
if _USE_PG:
    DB_PATH = ""   # PostgreSQL 사용 시 미사용
else:
    import sqlite3 as _sqlite3
    _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH = os.path.join(_BASE, "sector_analysis.db")


# ── 내부: 커서 컨텍스트 매니저 ───────────────────────────────────────────────
@contextmanager
def _get_cursor() -> Generator:
    """백엔드에 맞는 DB 커서를 열고 자동 commit / rollback / close."""
    if _USE_PG:
        import urllib.parse
        import psycopg2
        import psycopg2.extras
        # Python 3.14 urlparse가 비밀번호 내 대괄호를 IPv6로 잘못 해석 → 직접 파싱
        _url = _PG_URL
        if _url.startswith("postgresql://"):
            _url = _url[len("postgresql://"):]
        elif _url.startswith("postgres://"):
            _url = _url[len("postgres://"):]
        _at = _url.rfind("@")                        # 마지막 @ 기준으로 분리
        _userinfo = _url[:_at]
        _hostinfo  = _url[_at + 1:]
        _ci = _userinfo.index(":")                   # 첫 번째 : 기준으로 user/pass
        _pg_user = urllib.parse.unquote(_userinfo[:_ci])
        _pg_pass = urllib.parse.unquote(_userinfo[_ci + 1:])
        _host, _, _rest = _hostinfo.partition(":")
        _port_s, _, _dbname = _rest.partition("/")
        _pg_port = int(_port_s) if _port_s.isdigit() else 6543
        _pg_db   = _dbname or "postgres"
        conn = psycopg2.connect(
            host=_host,
            port=_pg_port,
            user=_pg_user,
            password=_pg_pass,
            dbname=_pg_db,
            connect_timeout=10,
            sslmode="require",
        )
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()
    else:
        conn = _sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = _sqlite3.Row
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        finally:
            cur.close()
            conn.close()


def _ph() -> str:
    """SQL 플레이스홀더 반환 — PostgreSQL: %s, SQLite: ?"""
    return "%s" if _USE_PG else "?"


def _row_to_dict(row) -> Dict[str, Any]:
    """백엔드 무관하게 dict 변환 (RealDictRow / sqlite3.Row 모두 처리)."""
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    return dict(row)


# ── 공개 API ─────────────────────────────────────────────────────────────────

def init_db() -> None:
    """테이블이 없으면 생성."""
    with _get_cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_queue (
                ticker       TEXT PRIMARY KEY,
                company_name TEXT,
                sector       TEXT,
                market       TEXT,
                priority     REAL DEFAULT 999.0,
                status       TEXT DEFAULT 'pending',
                added_at     TEXT
            )
        """)
        cur.execute("""
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
            )
        """)


def add_stocks_to_queue(stocks: List[Dict[str, Any]]) -> int:
    """
    큐에 종목 추가.
    - 신규 종목: pending으로 삽입
    - 기존 종목: status를 pending으로 리셋 (재분석 대상으로 복구)
    반환값: 처리된 종목 수
    """
    now = datetime.now().isoformat()
    p = _ph()
    with _write_lock:
        with _get_cursor() as cur:
            for s in stocks:
                if _USE_PG:
                    cur.execute(
                        f"""
                        INSERT INTO stock_queue
                            (ticker, company_name, sector, market, priority, status, added_at)
                        VALUES ({p}, {p}, {p}, {p}, {p}, 'pending', {p})
                        ON CONFLICT (ticker) DO UPDATE SET
                            status   = CASE WHEN stock_queue.status = 'in_progress'
                                           THEN 'in_progress' ELSE 'pending' END,
                            added_at = EXCLUDED.added_at
                        """,
                        (
                            s["ticker"], s.get("company_name", ""), s.get("sector", ""),
                            s.get("market", "KOSPI"), s.get("priority", 999.0), now,
                        ),
                    )
                else:
                    cur.execute(
                        f"""
                        INSERT INTO stock_queue
                            (ticker, company_name, sector, market, priority, status, added_at)
                        VALUES ({p}, {p}, {p}, {p}, {p}, 'pending', {p})
                        ON CONFLICT(ticker) DO UPDATE SET
                            status   = CASE WHEN status = 'in_progress'
                                           THEN 'in_progress' ELSE 'pending' END,
                            added_at = excluded.added_at
                        """,
                        (
                            s["ticker"], s.get("company_name", ""), s.get("sector", ""),
                            s.get("market", "KOSPI"), s.get("priority", 999.0), now,
                        ),
                    )
    return len(stocks)


def get_next_pending(reanalysis_days: int = 7) -> Optional[Dict[str, Any]]:
    """
    분석 대기 종목 1개 반환 (우선순위 오름차순).
    - status='pending' 인 종목 (분석 미완료)
    - status='done' 이면서 analyzed_at이 reanalysis_days 보다 오래된 종목 (재분석)
    in_progress 상태는 현재 처리 중이므로 제외.
    """
    cutoff = (datetime.now() - timedelta(days=reanalysis_days)).isoformat()
    p = _ph()
    with _get_cursor() as cur:
        cur.execute(
            f"""
            SELECT q.ticker, q.company_name, q.sector, q.market
            FROM   stock_queue q
            LEFT JOIN sector_analysis a ON q.ticker = a.ticker
            WHERE  q.status != 'in_progress'
              AND  (a.ticker IS NULL OR a.analyzed_at < {p})
            ORDER BY q.priority ASC
            LIMIT 1
            """,
            (cutoff,),
        )
        row = cur.fetchone()
        return _row_to_dict(row) if row else None


def mark_in_progress(ticker: str) -> None:
    p = _ph()
    with _write_lock:
        with _get_cursor() as cur:
            cur.execute(
                f"UPDATE stock_queue SET status='in_progress' WHERE ticker={p}",
                (ticker,),
            )


def mark_pending(ticker: str) -> None:
    """오류 발생 시 pending으로 되돌려 재시도 허용."""
    p = _ph()
    with _write_lock:
        with _get_cursor() as cur:
            cur.execute(
                f"UPDATE stock_queue SET status='pending' WHERE ticker={p}",
                (ticker,),
            )


def save_result(result: Dict[str, Any]) -> None:
    """분석 결과 upsert 저장 후 큐 상태 done 처리."""
    now = datetime.now().isoformat()
    p = _ph()
    vals = (
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
    )
    with _write_lock:
        with _get_cursor() as cur:
            if _USE_PG:
                cur.execute(
                    f"""
                    INSERT INTO sector_analysis
                        (ticker, company_name, sector, sector_kr, market,
                         per, pbr, profit_margin, revenue_growth,
                         appeal_score, buy_signal, ai_summary,
                         key_strength, key_risk, investment_horizon, analyzed_at)
                    VALUES ({p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p})
                    ON CONFLICT (ticker) DO UPDATE SET
                        company_name       = EXCLUDED.company_name,
                        sector             = EXCLUDED.sector,
                        sector_kr          = EXCLUDED.sector_kr,
                        market             = EXCLUDED.market,
                        per                = EXCLUDED.per,
                        pbr                = EXCLUDED.pbr,
                        profit_margin      = EXCLUDED.profit_margin,
                        revenue_growth     = EXCLUDED.revenue_growth,
                        appeal_score       = EXCLUDED.appeal_score,
                        buy_signal         = EXCLUDED.buy_signal,
                        ai_summary         = EXCLUDED.ai_summary,
                        key_strength       = EXCLUDED.key_strength,
                        key_risk           = EXCLUDED.key_risk,
                        investment_horizon = EXCLUDED.investment_horizon,
                        analyzed_at        = EXCLUDED.analyzed_at
                    """,
                    vals,
                )
            else:
                cur.execute(
                    f"""
                    INSERT OR REPLACE INTO sector_analysis
                        (ticker, company_name, sector, sector_kr, market,
                         per, pbr, profit_margin, revenue_growth,
                         appeal_score, buy_signal, ai_summary,
                         key_strength, key_risk, investment_horizon, analyzed_at)
                    VALUES ({p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p})
                    """,
                    vals,
                )
            cur.execute(
                f"UPDATE stock_queue SET status='done' WHERE ticker={p}",
                (result["ticker"],),
            )


def get_all_results(sector_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """분석 완료 결과 조회. appeal_score 내림차순."""
    p = _ph()
    with _get_cursor() as cur:
        if sector_filter and sector_filter != "전체":
            cur.execute(
                f"""
                SELECT * FROM sector_analysis
                WHERE sector_kr = {p} OR sector = {p}
                ORDER BY appeal_score DESC NULLS LAST
                """,
                (sector_filter, sector_filter),
            )
        else:
            cur.execute(
                "SELECT * FROM sector_analysis ORDER BY appeal_score DESC NULLS LAST"
            )
        rows = cur.fetchall()
        return [_row_to_dict(r) for r in rows]


def get_queue_stats() -> Dict[str, int]:
    """큐 및 분석 현황 통계."""
    with _get_cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM stock_queue")
        total = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM stock_queue WHERE status='done'")
        done = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM stock_queue WHERE status='pending'")
        pending = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM sector_analysis")
        analyzed = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM sector_analysis WHERE buy_signal=1")
        buy_signals = cur.fetchone()

    def _int(row) -> int:
        if row is None:
            return 0
        if isinstance(row, dict):
            return int(list(row.values())[0] or 0)
        return int(row[0] or 0)

    return {
        "total":       _int(total),
        "done":        _int(done),
        "pending":     _int(pending),
        "analyzed":    _int(analyzed),
        "buy_signals": _int(buy_signals),
    }


def get_sectors() -> List[str]:
    """분석 결과에서 섹터 목록 추출."""
    with _get_cursor() as cur:
        cur.execute(
            "SELECT DISTINCT COALESCE(NULLIF(sector_kr,''), sector) AS s "
            "FROM sector_analysis WHERE s IS NOT NULL AND s != '' ORDER BY s"
        )
        rows = cur.fetchall()
        return [_row_to_dict(r).get("s", "") for r in rows if r]


def reset_in_progress() -> None:
    """서버 재시작 시 in_progress 종목을 pending으로 복구."""
    with _write_lock:
        with _get_cursor() as cur:
            cur.execute(
                "UPDATE stock_queue SET status='pending' WHERE status='in_progress'"
            )


def reset_all() -> None:
    """큐·결과 전체 초기화 (UI에서 사용자 확인 후 호출)."""
    with _write_lock:
        with _get_cursor() as cur:
            cur.execute("DELETE FROM stock_queue")
            cur.execute("DELETE FROM sector_analysis")
