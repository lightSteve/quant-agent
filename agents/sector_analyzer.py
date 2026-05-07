"""
KOSPI 전 종목 섹터별 전수 분석 — 백그라운드 스레드 워커.

흐름:
  1. FinanceDataReader로 KOSPI 종목 목록 수집 (없으면 대표 20종목 fallback)
  2. SQLite 큐에 적재 (우선순위: 낮은 PBR = 높은 우선순위 → 저평가 후보 먼저)
  3. 백그라운드 스레드가 40초 간격으로 1종목씩 yfinance + AI 분석
  4. 결과 SQLite 저장 → UI가 언제든 조회 가능
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import yfinance as yf

from agents.analyst_agent import AnalystAgent
from data.sector_db import (
    add_stocks_to_queue,
    get_next_pending,
    init_db,
    mark_in_progress,
    mark_pending,
    reset_in_progress,
    save_result,
)

logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────
ANALYSIS_INTERVAL_SECS = 40   # 종목 간 대기 (API rate-limit 준수)
REANALYSIS_DAYS = 7            # 재분석 주기 (일)
IDLE_WAIT_SECS = 60            # 큐 소진 시 대기

SECTOR_KR: Dict[str, str] = {
    "Technology": "기술",
    "Financial Services": "금융",
    "Healthcare": "헬스케어",
    "Consumer Cyclical": "소비재(경기)",
    "Industrials": "산업재",
    "Communication Services": "통신서비스",
    "Energy": "에너지",
    "Basic Materials": "소재",
    "Consumer Defensive": "소비재(필수)",
    "Real Estate": "부동산",
    "Utilities": "유틸리티",
    "Semiconductors": "반도체",
    "Automotive": "자동차",
}


# ── KOSPI 종목 목록 수집 ────────────────────────

def _fetch_kospi_stocks() -> List[Dict[str, Any]]:
    """FinanceDataReader로 KOSPI 전 종목 반환. 실패 시 대표 20종목 fallback."""
    try:
        import FinanceDataReader as fdr  # optional dependency

        df = fdr.StockListing("KOSPI")
        stocks: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            symbol = str(row.get("Symbol", "")).strip()
            if not symbol or not symbol.isdigit():
                continue
            ticker = f"{symbol}.KS"
            name = str(row.get("Name", symbol))
            sector = str(row.get("Sector", "") or "")
            # 시가총액 역수 = 우선순위 (데이터가 많은 대형주 먼저)
            try:
                mc = float(row.get("MarketCap", 0) or 0)
                priority = 1.0 / mc if mc > 0 else 999.0
            except Exception:
                priority = 999.0
            stocks.append(
                {
                    "ticker": ticker,
                    "company_name": name,
                    "sector": sector,
                    "market": "KOSPI",
                    "priority": priority,
                }
            )
        logger.info("KOSPI 종목 %d개 로드 완료", len(stocks))
        return stocks
    except ImportError:
        logger.warning("FinanceDataReader 미설치 → 대표 종목 20개 사용")
        return _fallback_stocks()
    except Exception as exc:
        logger.error("KOSPI 종목 로드 실패: %s", exc)
        return _fallback_stocks()


def _fallback_stocks() -> List[Dict[str, Any]]:
    """FinanceDataReader 없을 때 대표 KOSPI 종목."""
    rows = [
        ("005930", "삼성전자",          "Technology"),
        ("000660", "SK하이닉스",         "Semiconductors"),
        ("207940", "삼성바이오로직스",    "Healthcare"),
        ("006400", "삼성SDI",            "Technology"),
        ("051910", "LG화학",             "Basic Materials"),
        ("035420", "NAVER",              "Communication Services"),
        ("000270", "기아",               "Automotive"),
        ("068270", "셀트리온",           "Healthcare"),
        ("105560", "KB금융",             "Financial Services"),
        ("055550", "신한지주",           "Financial Services"),
        ("035720", "카카오",             "Communication Services"),
        ("028260", "삼성물산",           "Industrials"),
        ("066570", "LG전자",             "Technology"),
        ("096770", "SK이노베이션",       "Energy"),
        ("003550", "LG",                 "Industrials"),
        ("034730", "SK",                 "Energy"),
        ("012330", "현대모비스",         "Automotive"),
        ("032830", "삼성생명",           "Financial Services"),
        ("011200", "HMM",                "Industrials"),
        ("010950", "S-Oil",              "Energy"),
    ]
    return [
        {
            "ticker": f"{code}.KS",
            "company_name": name,
            "sector": sector,
            "market": "KOSPI",
            "priority": float(i),
        }
        for i, (code, name, sector) in enumerate(rows)
    ]


# ── yfinance 지표 수집 ────────────────────────

def _fetch_metrics(ticker: str) -> Dict[str, Any]:
    """yfinance에서 주요 재무 지표 수집."""
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "per":            info.get("trailingPE"),
            "pbr":            info.get("priceToBook"),
            "profit_margin":  info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "roe":            info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "dividend_yield": info.get("dividendYield"),
            "sector":         info.get("sector", ""),
            "company_name":   info.get("longName") or info.get("shortName", ""),
        }
    except Exception as exc:
        logger.debug("%s 지표 수집 실패: %s", ticker, exc)
        return {}


# ── 모듈 레벨 싱글톤 ─────────────────────────
# Streamlit은 매 인터랙션마다 스크립트를 재실행하므로,
# 백그라운드 스레드는 모듈 레벨에 보관해야 유지된다.

_analyzer: Optional["SectorAnalyzer"] = None
_singleton_lock = threading.Lock()


def get_analyzer() -> "SectorAnalyzer":
    global _analyzer
    with _singleton_lock:
        if _analyzer is None:
            _analyzer = SectorAnalyzer()
    return _analyzer


# ── SectorAnalyzer ──────────────────────────

class SectorAnalyzer:
    """백그라운드에서 KOSPI 종목을 순차 AI 분석하는 워커."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._api_key = ""
        self._provider = "github"
        self._current_ticker = ""
        self._last_error = ""
        init_db()
        reset_in_progress()

    # ── 공개 인터페이스 ─────────────────────

    def start(self, api_key: str, provider: str = "github") -> bool:
        """분석 시작. 이미 실행 중이거나 api_key 없으면 False."""
        if self.is_running() or not api_key:
            return False
        self._api_key = api_key
        self._provider = provider
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="sector-analyzer"
        )
        self._thread.start()
        logger.info("섹터 분석기 시작 (provider=%s)", provider)
        return True

    def stop(self) -> None:
        """분석 중지 요청. 현재 분석 중인 종목 완료 후 종료."""
        self._stop_event.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def load_queue(self) -> int:
        """KOSPI 종목 목록을 가져와 큐에 적재. 반환: 신규 추가 수."""
        stocks = _fetch_kospi_stocks()
        return add_stocks_to_queue(stocks)

    @property
    def current_ticker(self) -> str:
        return self._current_ticker

    @property
    def last_error(self) -> str:
        return self._last_error

    # ── 백그라운드 루프 ─────────────────────

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                stock = get_next_pending(REANALYSIS_DAYS)
                if stock is None:
                    self._current_ticker = ""
                    self._stop_event.wait(IDLE_WAIT_SECS)
                    continue

                ticker = stock["ticker"]
                self._current_ticker = ticker
                mark_in_progress(ticker)

                try:
                    self._analyze_one(stock)
                except Exception as exc:
                    self._last_error = f"{ticker}: {str(exc)[:120]}"
                    logger.error("%s 분석 오류: %s", ticker, exc)
                    mark_pending(ticker)  # 다음 주기에 재시도
                finally:
                    self._current_ticker = ""

                # rate-limit 준수 대기
                self._stop_event.wait(ANALYSIS_INTERVAL_SECS)

            except Exception as exc:
                logger.error("루프 오류: %s", exc)
                self._stop_event.wait(10)

        logger.info("섹터 분석기 종료")

    def _analyze_one(self, stock: Dict[str, Any]) -> None:
        ticker = stock["ticker"]
        company_name = stock.get("company_name", ticker)
        sector = stock.get("sector", "")

        # 1. 재무 지표 수집
        metrics = _fetch_metrics(ticker)
        if metrics.get("company_name"):
            company_name = metrics["company_name"]
        if metrics.get("sector"):
            sector = metrics["sector"]

        sector_kr = SECTOR_KR.get(sector, sector)

        # 유효 지표 없으면 "데이터 없음"으로 저장 후 스킵
        has_data = any(
            metrics.get(k) is not None
            for k in ("per", "pbr", "profit_margin")
        )
        if not has_data:
            logger.debug("%s 유효 지표 없음, 스킵", ticker)
            save_result(
                {
                    "ticker": ticker,
                    "company_name": company_name,
                    "sector": sector,
                    "sector_kr": sector_kr,
                    "market": stock.get("market", "KOSPI"),
                    "ai_summary": "재무 데이터 없음 (상장폐지 또는 조회 불가)",
                }
            )
            return

        # 2. AI 매수 매력도 분석
        agent = AnalystAgent(self._api_key, provider=self._provider)
        metrics_for_ai = {
            "PER":          metrics.get("per"),
            "PBR":          metrics.get("pbr"),
            "순이익마진":   f"{(metrics.get('profit_margin') or 0) * 100:.1f}%",
            "매출성장률":   f"{(metrics.get('revenue_growth') or 0) * 100:.1f}%",
            "ROE":          f"{(metrics.get('roe') or 0) * 100:.1f}%",
            "부채비율(D/E)": metrics.get("debt_to_equity"),
            "배당수익률":   f"{(metrics.get('dividend_yield') or 0) * 100:.2f}%",
        }
        ai = agent.analyze_buy_appeal(
            ticker=ticker,
            company_name=company_name,
            sector=sector_kr or sector,
            metrics=metrics_for_ai,
        )

        # 3. 결과 저장
        save_result(
            {
                "ticker": ticker,
                "company_name": company_name,
                "sector": sector,
                "sector_kr": sector_kr,
                "market": stock.get("market", "KOSPI"),
                "per":             metrics.get("per"),
                "pbr":             metrics.get("pbr"),
                "profit_margin":   metrics.get("profit_margin"),
                "revenue_growth":  metrics.get("revenue_growth"),
                "appeal_score":    ai.get("appeal_score"),
                "buy_signal":      ai.get("buy_signal", False),
                "ai_summary":      ai.get("summary", ""),
                "key_strength":    ai.get("key_strength", ""),
                "key_risk":        ai.get("key_risk", ""),
                "investment_horizon": ai.get("investment_horizon", ""),
            }
        )
        logger.info(
            "✓ %s(%s) 분석 완료 — 매력도: %s",
            company_name, ticker, ai.get("appeal_score"),
        )
