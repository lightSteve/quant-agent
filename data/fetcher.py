"""
Financial data fetcher using yfinance.
Supports Korean (KS/KQ) and US stocks.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf


# Sector → representative peer tickers (US market as reference)
SECTOR_PEERS: Dict[str, list[str]] = {
    "Technology": ["MSFT", "AAPL", "GOOGL", "META", "NVDA"],
    "Semiconductors": ["NVDA", "AMD", "INTC", "TSM", "AVGO"],
    "Consumer Electronics": ["AAPL", "SONY", "SSNLF", "LG", "MSFT"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS"],
    "Healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV"],
    "Energy": ["XOM", "CVX", "BP", "SHEL", "TTE"],
    "Consumer Cyclical": ["AMZN", "TSLA", "NKE", "MCD", "SBUX"],
    "Industrials": ["CAT", "BA", "HON", "MMM", "GE"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "VZ"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    "Real Estate": ["AMT", "PLD", "EQIX", "CCI", "SPG"],
    "Basic Materials": ["LIN", "APD", "ECL", "SHW", "NEM"],
    "Automotive": ["TSLA", "TM", "F", "GM", "STLA"],
}


class FinancialDataFetcher:
    """Fetch and compute financial metrics for a given ticker."""

    def __init__(self, raw_ticker: str) -> None:
        raw_norm = raw_ticker.strip().upper().replace(" ", "")
        # 6자리 한국 코드는 코스피/코스닥 자동 감지
        if "." not in raw_norm and raw_norm.isdigit() and len(raw_norm) == 6:
            self.ticker = self._detect_korean_exchange(raw_norm)
        else:
            self.ticker = self._resolve_ticker(raw_ticker)
        self._yf = yf.Ticker(self.ticker)
        self._info: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Ticker resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_ticker(raw: str) -> str:
        raw = raw.strip().upper().replace(" ", "")
        if "." in raw:
            return raw
        if raw.isdigit():
            if len(raw) == 6:
                return f"{raw}.KS"   # KOSPI default
            if len(raw) == 4:
                return f"{raw}.T"    # Tokyo
        return raw

    @staticmethod
    def _detect_korean_exchange(code: str) -> str:
        """실제 history 데이터가 낙는 거래소를 반환 (KS 우선, 없으면 KQ 시도)."""
        for suffix in (".KS", ".KQ"):
            try:
                hist = yf.Ticker(f"{code}{suffix}").history(period="5d")
                if hist is not None and not hist.empty:
                    return f"{code}{suffix}"
            except Exception:
                pass
            time.sleep(0.3)
        return f"{code}.KS"  # 감지 실패 시 KOSPI 폴백

    # ------------------------------------------------------------------
    # Info cache
    # ------------------------------------------------------------------

    @property
    def info(self) -> Dict[str, Any]:
        if self._info is None:
            for attempt in range(3):
                try:
                    data = self._yf.info or {}
                    if data and len(data) > 5:
                        self._info = data
                        break
                except Exception:
                    pass
                time.sleep(1.5)
            if self._info is None:
                self._info = {}
        return self._info

    # ------------------------------------------------------------------
    # Company metadata
    # ------------------------------------------------------------------

    def get_company_name(self) -> str:
        return self.info.get("longName") or self.info.get("shortName") or self.ticker

    def get_sector(self) -> str:
        return self.info.get("sector", "Unknown")

    def get_industry(self) -> str:
        return self.info.get("industry", "Unknown")

    def get_currency(self) -> str:
        return self.info.get("currency", "USD")

    # ------------------------------------------------------------------
    # Income statement (3 years)
    # ------------------------------------------------------------------

    def get_financials_3yr(self) -> pd.DataFrame:
        """Annual income statement, most recent 3 columns."""
        try:
            df = self._yf.financials
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.iloc[:, :3]
            wanted = [
                "Total Revenue",
                "Operating Income",
                "Net Income",
                "Gross Profit",
                "EBIT",
                "EBITDA",
            ]
            rows = [r for r in wanted if r in df.index]
            return df.loc[rows] if rows else df.head(6)
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Balance sheet
    # ------------------------------------------------------------------

    def get_balance_sheet_3yr(self) -> pd.DataFrame:
        try:
            df = self._yf.balance_sheet
            if df is None or df.empty:
                return pd.DataFrame()
            return df.iloc[:, :3]
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Live price (history 기반 — KR 종목 info 불안정 대응)
    # ------------------------------------------------------------------

    def get_live_price(self) -> Optional[float]:
        """yfinance history 기반 현재가. info보다 신뢰도 높음."""
        try:
            hist = self._yf.history(period="2d")
            if hist is not None and not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Key valuation / financial metrics
    # ------------------------------------------------------------------

    def get_key_metrics(self) -> Dict[str, Any]:
        i = self.info
        # history 기반 현재가 우선 — KR 종목에서 info 가격이 지연되는 문제 방지
        current_price = self.get_live_price() or i.get("currentPrice") or i.get("regularMarketPrice")
        return {
            "pe_ratio": i.get("trailingPE"),
            "forward_pe": i.get("forwardPE"),
            "pb_ratio": i.get("priceToBook"),
            "ps_ratio": i.get("priceToSalesTrailingTwelveMonths"),
            "market_cap": i.get("marketCap"),
            "enterprise_value": i.get("enterpriseValue"),
            "dividend_yield": i.get("dividendYield"),
            "roe": i.get("returnOnEquity"),
            "roa": i.get("returnOnAssets"),
            "profit_margin": i.get("profitMargins"),
            "operating_margin": i.get("operatingMargins"),
            "revenue_growth": i.get("revenueGrowth"),
            "earnings_growth": i.get("earningsGrowth"),
            "debt_to_equity": i.get("debtToEquity"),
            "current_ratio": i.get("currentRatio"),
            "quick_ratio": i.get("quickRatio"),
            "beta": i.get("beta"),
            "fifty_two_week_high": i.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": i.get("fiftyTwoWeekLow"),
            "current_price": current_price,
        }

    # ------------------------------------------------------------------
    # Price history
    # ------------------------------------------------------------------

    def get_price_history(self, period: str = "3y") -> pd.DataFrame:
        try:
            return self._yf.history(period=period)
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Net profit margin trend
    # ------------------------------------------------------------------

    def calculate_net_profit_margin(self) -> pd.Series:
        fin = self.get_financials_3yr()
        if fin.empty:
            return pd.Series(dtype=float)
        try:
            revenue = fin.loc["Total Revenue"]
            net_income = fin.loc["Net Income"]
            margin = (net_income / revenue * 100).round(2)
            return margin
        except (KeyError, ZeroDivisionError):
            return pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # Peer comparison
    # ------------------------------------------------------------------

    def get_peer_metrics(self) -> pd.DataFrame:
        """Fetch key metrics for sector peers."""
        sector = self.get_sector()
        peers = SECTOR_PEERS.get(sector, SECTOR_PEERS.get("Technology", []))

        rows = []
        for ticker in peers[:5]:
            try:
                p_info = yf.Ticker(ticker).info or {}
                rows.append(
                    {
                        "ticker": ticker,
                        "name": (p_info.get("shortName") or ticker)[:20],
                        "pe_ratio": p_info.get("trailingPE"),
                        "pb_ratio": p_info.get("priceToBook"),
                        "profit_margin": p_info.get("profitMargins"),
                        "roe": p_info.get("returnOnEquity"),
                    }
                )
            except Exception:
                continue

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # DCA simulation
    # ------------------------------------------------------------------

    def calculate_dca_comparison(
        self,
        monthly_amount: float = 300_000,
        years: int = 3,
    ) -> Dict[str, Any]:
        """
        Simulate monthly DCA into this stock vs S&P 500 (SPY).
        Returns percentage returns (currency-agnostic comparison).
        """
        end = pd.Timestamp.now(tz="UTC")
        start = end - pd.DateOffset(years=years)
        results: Dict[str, Any] = {}

        def _simulate(hist: pd.DataFrame, label: str) -> None:
            if hist.empty:
                return
            prices = hist["Close"].resample("ME").last().dropna()
            if len(prices) < 2:
                return
            shares = (monthly_amount / prices).cumsum()
            total_invested = monthly_amount * len(prices)
            final_value_units = shares.iloc[-1] * prices.iloc[-1]
            ret_pct = (final_value_units - total_invested) / total_invested * 100
            results[label] = {
                "total_invested": total_invested,
                "final_value_units": final_value_units,
                "return_pct": round(float(ret_pct), 2),
                "prices": prices,
                "monthly_amounts": monthly_amount,
            }

        try:
            hist = self._yf.history(start=start, end=end)
            _simulate(hist, "stock")
        except Exception:
            pass

        try:
            spy_hist = yf.Ticker("SPY").history(start=start, end=end)
            _simulate(spy_hist, "sp500")
        except Exception:
            pass

        return results
