"""
KOSPI 전 종목 섹터별 전수 분석 — 백그라운드 스레드 워커.

흐름:
  1. FinanceDataReader로 KOSPI 종목 목록 수집 (없으면 대표 20종목 fallback)
  2. SQLite 큐에 적재 (우선순위: 낮은 PBR = 높은 우선순위 → 저평가 후보 먼저)
  3. 백그라운드 스레드가 15초 간격으로 1종목씩 yfinance + AI 분석
  4. 결과는 성공/실패 무관하게 항상 SQLite 저장 → UI가 언제든 조회 가능
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import yfinance as yf

from agents.analyst_agent import AnalystAgent
from data.sector_db import (
    add_stocks_to_queue,
    get_next_pending,
    init_db,
    mark_in_progress,
    reset_in_progress,
    save_result,
)

logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────
ANALYSIS_INTERVAL_SECS = 15   # 종목 간 대기 (첫 종목은 즉시 처리)
REANALYSIS_DAYS = 7            # 재분석 주기 (일)
IDLE_WAIT_SECS = 60            # 큐 소진 시 대기
MAX_LOG_LINES = 30             # UI 로그 버퍼 크기

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
    """yfinance에서 주요 재무 지표 수집 (타임아웃 15초)."""
    import threading as _t

    result: Dict[str, Any] = {}
    exc_holder: list = []

    def _fetch():
        try:
            info = yf.Ticker(ticker).info or {}
            result.update({
                "per":            info.get("trailingPE"),
                "pbr":            info.get("priceToBook"),
                "profit_margin":  info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "roe":            info.get("returnOnEquity"),
                "debt_to_equity": info.get("debtToEquity"),
                "dividend_yield": info.get("dividendYield"),
                "sector":         info.get("sector", ""),
                "company_name":   info.get("longName") or info.get("shortName", ""),
            })
        except Exception as exc:
            exc_holder.append(exc)

    t = _t.Thread(target=_fetch, daemon=True)
    t.start()
    t.join(timeout=15)
    if not result and not exc_holder:
        logger.debug("%s yfinance 타임아웃 (15초)", ticker)
    elif exc_holder:
        logger.debug("%s 지표 수집 실패: %s", ticker, exc_holder[0])
    return result


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
        self._log_buffer: deque = deque(maxlen=MAX_LOG_LINES)
        init_db()
        reset_in_progress()
        self._log("워커 초기화 완료")

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
        self._log(f"분석 시작 (provider={provider})")
        return True

    def stop(self) -> None:
        self._stop_event.set()
        self._log("중지 요청됨")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def load_queue(self) -> int:
        stocks = _fetch_kospi_stocks()
        n = add_stocks_to_queue(stocks)
        self._log(f"큐 적재: 신규 {n}개 / 전체 {len(stocks)}개")
        return n

    @property
    def current_ticker(self) -> str:
        return self._current_ticker

    @property
    def last_error(self) -> str:
        return self._last_error

    def get_log(self) -> str:
        return "\n".join(self._log_buffer)

    # ── 내부 유틸 ───────────────────────────

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_buffer.append(f"[{ts}] {msg}")

    # ── 백그라운드 루프 ─────────────────────

    def _run_loop(self) -> None:
        """핵심 변경: 모든 예외를 내부에서 처리하고 항상 save_result 호출."""
        first = True
        while not self._stop_event.is_set():
            try:
                stock = get_next_pending(REANALYSIS_DAYS)
                if stock is None:
                    self._current_ticker = ""
                    self._log("큐 소진 — 대기 중...")
                    self._stop_event.wait(IDLE_WAIT_SECS)
                    continue

                ticker = stock["ticker"]
                self._current_ticker = ticker
                mark_in_progress(ticker)
                self._log(f"▶ 분석 시작: {stock.get('company_name', ticker)} ({ticker})")

                self._analyze_one(stock)  # 내부에서 항상 save_result 호출

                self._current_ticker = ""

                # 첫 종목은 즉시 처리, 이후 간격 준수
                if first:
                    first = False
                else:
                    self._stop_event.wait(ANALYSIS_INTERVAL_SECS)

            except Exception as exc:
                logger.error("루프 레벨 오류: %s", exc)
                self._log(f"루프 오류: {str(exc)[:80]}")
                self._stop_event.wait(10)

        self._log("워커 종료")

    def _analyze_one(self, stock: Dict[str, Any]) -> None:
        """
        핵심 변경: try/finally로 save_result를 항상 보장.
        성공·실패·타임아웃 모두 sector_analysis에 기록되어 분석완료 수가 증가.
        """
        ticker = stock["ticker"]
        company_name = stock.get("company_name", ticker)
        sector = stock.get("sector", "")

        # 항상 저장될 기본 결과 dict
        result: Dict[str, Any] = {
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "sector_kr": SECTOR_KR.get(sector, sector),
            "market": stock.get("market", "KOSPI"),
            "ai_summary": "",
        }

        try:
            # 1. 재무 지표 수집
            metrics = _fetch_metrics(ticker)
            if metrics.get("company_name"):
                result["company_name"] = metrics["company_name"]
            if metrics.get("sector"):
                result["sector"] = metrics["sector"]
                result["sector_kr"] = SECTOR_KR.get(metrics["sector"], metrics["sector"])

            result.update({
                "per":            metrics.get("per"),
                "pbr":            metrics.get("pbr"),
                "profit_margin":  metrics.get("profit_margin"),
                "revenue_growth": metrics.get("revenue_growth"),
            })

            has_data = any(
                metrics.get(k) is not None
                for k in ("per", "pbr", "profit_margin")
            )

            if not has_data:
                result["ai_summary"] = "재무 데이터 없음 (Yahoo Finance 조회 불가)"
                self._log(f"  → {ticker} 재무 데이터 없음")
            else:
                # 2. AI 매수 매력도 분석 (실패해도 재무 데이터만으로 저장)
                try:
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
                        company_name=result["company_name"],
                        sector=result.get("sector_kr") or sector,
                        metrics=metrics_for_ai,
                    )
                    result.update({
                        "appeal_score":       ai.get("appeal_score"),
                        "buy_signal":         ai.get("buy_signal", False),
                        "ai_summary":         ai.get("summary", ""),
                        "key_strength":       ai.get("key_strength", ""),
                        "key_risk":           ai.get("key_risk", ""),
                        "investment_horizon": ai.get("investment_horizon", ""),
                    })
                    self._log(
                        f"  ✓ {result['company_name']} 완료 "
                        f"(매력도={ai.get('appeal_score')}, "
                        f"매수={'✅' if ai.get('buy_signal') else '⚪'})"
                    )
                except Exception as ai_exc:
                    err = str(ai_exc)[:100]
                    self._last_error = f"{ticker} AI: {err}"
                    result["ai_summary"] = f"AI 분석 실패: {err}"
                    self._log(f"  ⚠ {ticker} AI 오류: {err}")

        except Exception as exc:
            err = str(exc)[:100]
            self._last_error = f"{ticker}: {err}"
            result["ai_summary"] = f"수집 오류: {err}"
            self._log(f"  ✗ {ticker} 오류: {err}")

        finally:
            # 성공/실패 무관 항상 저장 → analyzed 카운트 증가
            try:
                save_result(result)
            except Exception as save_exc:
                self._log(f"  ✗ {ticker} DB 저장 실패: {str(save_exc)[:80]}")
                logger.error("%s save_result 실패: %s", ticker, save_exc)
