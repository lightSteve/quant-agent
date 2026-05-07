"""
Rule-based quantitative scoring engine.
Scores a stock 0–5 across four dimensions:
  PER valuation (0–1.5), PBR valuation (0–1.5),
  Profitability (0–1.0), Financial health (0–1.0)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


class QuantScorer:
    """Deterministic quant scoring — no external API calls."""

    _MAX = {"per": 1.5, "pbr": 1.5, "profitability": 1.0, "financial_health": 1.0}

    # ------------------------------------------------------------------
    # Peer averages
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_peer_averages(peer_df: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Compute median metrics from peer DataFrame."""
        if peer_df is None or peer_df.empty:
            return {}
        result: Dict[str, float] = {}
        mapping = {
            "pe_ratio": "avg_pe",
            "pb_ratio": "avg_pb",
            "profit_margin": "avg_margin",
            "roe": "avg_roe",
        }
        for col, key in mapping.items():
            if col in peer_df.columns:
                valid = peer_df[col].dropna()
                valid = valid[valid > 0]
                if not valid.empty:
                    result[key] = float(valid.median())
        return result

    # ------------------------------------------------------------------
    # Main scoring
    # ------------------------------------------------------------------

    def calculate_score(
        self,
        metrics: Dict[str, Any],
        peer_averages: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Returns:
          total_score   float  0–5
          breakdown     dict   category → score
          explanations  dict   category → human-readable reason
          grade         str    letter grade with label
        """
        breakdown: Dict[str, float] = {}
        explanations: Dict[str, str] = {}

        # --- PER ---
        per_score, per_exp = self._score_per(
            metrics.get("pe_ratio"), peer_averages.get("avg_pe")
        )
        breakdown["per"] = per_score
        explanations["per"] = per_exp

        # --- PBR ---
        pbr_score, pbr_exp = self._score_pbr(
            metrics.get("pb_ratio"), peer_averages.get("avg_pb")
        )
        breakdown["pbr"] = pbr_score
        explanations["pbr"] = pbr_exp

        # --- Profitability ---
        prof_score, prof_exp = self._score_profitability(metrics.get("profit_margin"))
        breakdown["profitability"] = prof_score
        explanations["profitability"] = prof_exp

        # --- Financial health ---
        health_score, health_exp = self._score_health(
            metrics.get("debt_to_equity"), metrics.get("current_ratio")
        )
        breakdown["financial_health"] = health_score
        explanations["financial_health"] = health_exp

        total = round(sum(breakdown.values()), 2)
        return {
            "total_score": total,
            "max_score": 5.0,
            "breakdown": breakdown,
            "explanations": explanations,
            "grade": self._grade(total),
        }

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------

    def _score_per(
        self,
        per: Optional[float],
        peer_avg: Optional[float],
    ) -> tuple[float, str]:
        if per is None or per <= 0:
            return 0.0, "PER 데이터 없음 또는 음수 (적자 기업)"

        if peer_avg and peer_avg > 0:
            ratio = per / peer_avg
            if ratio < 0.50:
                return 1.5, f"PER {per:.1f} — 업계 평균 {peer_avg:.1f}의 절반 이하 (강한 저평가)"
            if ratio < 0.75:
                return 1.0, f"PER {per:.1f} — 업계 평균 {peer_avg:.1f}보다 낮음 (저평가)"
            if ratio < 1.00:
                return 0.5, f"PER {per:.1f} — 업계 평균 {peer_avg:.1f}에 근접 (약간 저평가)"
            if ratio < 1.25:
                return 0.25, f"PER {per:.1f} — 업계 평균 {peer_avg:.1f} 수준 (적정)"
            return 0.0, f"PER {per:.1f} — 업계 평균 {peer_avg:.1f}보다 높음 (고평가)"

        # Absolute thresholds when no peer data
        if per < 10:
            return 1.5, f"PER {per:.1f} — 매우 낮음 (저평가 가능성 높음)"
        if per < 15:
            return 1.0, f"PER {per:.1f} — 낮음 (저평가)"
        if per < 20:
            return 0.5, f"PER {per:.1f} — 보통 수준"
        if per < 25:
            return 0.25, f"PER {per:.1f} — 약간 높음"
        return 0.0, f"PER {per:.1f} — 높음 (고평가 주의)"

    def _score_pbr(
        self,
        pbr: Optional[float],
        peer_avg: Optional[float],
    ) -> tuple[float, str]:
        if pbr is None or pbr <= 0:
            return 0.0, "PBR 데이터 없음"

        if peer_avg and peer_avg > 0:
            ratio = pbr / peer_avg
            if ratio < 0.50:
                return 1.5, f"PBR {pbr:.2f} — 업계 평균 {peer_avg:.2f}의 절반 이하 (강한 저평가)"
            if ratio < 0.75:
                return 1.0, f"PBR {pbr:.2f} — 업계 평균 {peer_avg:.2f}보다 낮음 (저평가)"
            if ratio < 1.00:
                return 0.5, f"PBR {pbr:.2f} — 업계 평균 {peer_avg:.2f}에 근접"
            return 0.0, f"PBR {pbr:.2f} — 업계 평균 {peer_avg:.2f} 이상 (고평가)"

        if pbr < 1.0:
            return 1.5, f"PBR {pbr:.2f} — 장부가치 이하 (강한 저평가 신호)"
        if pbr < 1.5:
            return 1.0, f"PBR {pbr:.2f} — 낮은 수준 (저평가)"
        if pbr < 2.0:
            return 0.5, f"PBR {pbr:.2f} — 보통 수준"
        return 0.0, f"PBR {pbr:.2f} — 높음"

    def _score_profitability(
        self, profit_margin: Optional[float]
    ) -> tuple[float, str]:
        if profit_margin is None:
            return 0.0, "수익성 데이터 없음"
        pct = profit_margin * 100
        if pct >= 15:
            return 1.0, f"순이익률 {pct:.1f}% — 우수"
        if pct >= 10:
            return 0.75, f"순이익률 {pct:.1f}% — 양호"
        if pct >= 5:
            return 0.5, f"순이익률 {pct:.1f}% — 보통"
        if pct > 0:
            return 0.25, f"순이익률 {pct:.1f}% — 낮음"
        return 0.0, f"순이익률 {pct:.1f}% — 적자"

    def _score_health(
        self,
        debt_to_equity: Optional[float],
        current_ratio: Optional[float],
    ) -> tuple[float, str]:
        score = 0.0
        notes: list[str] = []

        if debt_to_equity is not None:
            if debt_to_equity < 0.5:
                score += 0.5
                notes.append(f"부채비율 {debt_to_equity:.2f} (매우 건전)")
            elif debt_to_equity < 1.0:
                score += 0.35
                notes.append(f"부채비율 {debt_to_equity:.2f} (건전)")
            elif debt_to_equity < 2.0:
                score += 0.15
                notes.append(f"부채비율 {debt_to_equity:.2f} (보통)")
            else:
                notes.append(f"부채비율 {debt_to_equity:.2f} (높음, 주의)")

        if current_ratio is not None:
            if current_ratio >= 2.0:
                score += 0.5
                notes.append(f"유동비율 {current_ratio:.2f} (매우 양호)")
            elif current_ratio >= 1.5:
                score += 0.35
                notes.append(f"유동비율 {current_ratio:.2f} (양호)")
            elif current_ratio >= 1.0:
                score += 0.15
                notes.append(f"유동비율 {current_ratio:.2f} (보통)")
            else:
                notes.append(f"유동비율 {current_ratio:.2f} (낮음, 주의)")

        exp = " | ".join(notes) if notes else "재무건전성 데이터 없음"
        return round(min(score, 1.0), 2), exp

    # ------------------------------------------------------------------
    # Grade
    # ------------------------------------------------------------------

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 4.5:
            return "A+ (강한 매수)"
        if score >= 4.0:
            return "A (매수)"
        if score >= 3.0:
            return "B (매수 고려)"
        if score >= 2.0:
            return "C (관망)"
        if score >= 1.0:
            return "D (신중)"
        return "F (매수 불가)"
