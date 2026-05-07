"""
퀀트 투자 분석 에이전트 — Streamlit 대시보드
데이터 수집 → AI 해석 → 저평가 분석 → 투자 점수 & DCA 전략
"""

from __future__ import annotations

import io
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from agents.analyst_agent import AnalystAgent
from agents.quant_scorer import QuantScorer
from data.fetcher import FinancialDataFetcher
from utils.formatters import fmt_number, fmt_pct, fmt_ratio, fmt_usd, score_color
from utils import cache_manager

load_dotenv()

def _get_default_token() -> str:
    """로컬 .env 또는 Streamlit Cloud secrets에서 토큰 읽기."""
    try:
        return st.secrets.get("GITHUB_TOKEN", "") or os.getenv("GITHUB_TOKEN", "")
    except Exception:
        return os.getenv("GITHUB_TOKEN", "")

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="퀀트 투자 분석 에이전트",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #0d2137 0%, #1e4d7b 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.8rem;
        color: white;
    }
    .hero h1 { margin: 0 0 0.3rem 0; font-size: 2rem; }
    .hero p  { margin: 0; opacity: 0.85; font-size: 1rem; }
    .card {
        background: #f4f7fb;
        padding: 1.2rem 1.4rem;
        border-radius: 10px;
        border-left: 4px solid #1e4d7b;
        margin-bottom: 0.8rem;
    }
    .score-big {
        font-size: 3.4rem;
        font-weight: 800;
        text-align: center;
        line-height: 1.1;
    }
    .step-badge {
        display: inline-block;
        background: #1e4d7b;
        color: white;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _years_from_df(df: pd.DataFrame) -> list[str]:
    return [
        str(c.year) if hasattr(c, "year") else str(c)
        for c in df.columns
    ]


def _safe_divide(a: float, b: float) -> float | None:
    return a / b if b and b != 0 else None


def _sanitize_json(obj):
    """Recursively convert Timestamp keys/values to str for json.dumps compatibility."""
    if isinstance(obj, dict):
        return {str(k) if hasattr(k, "year") else k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if hasattr(obj, "year"):  # Timestamp, date, datetime
        return str(obj)
    return obj


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

def build_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## ⚙️ 설정")

        api_key = st.text_input(
            "GitHub Personal Access Token (PAT)",
            type="password",
            value=_get_default_token(),
            help="Fine-grained PAT 필요 — Permissions → Models: Read-only 이상 설정",
        )
        st.caption("👉 [PAT 발급하기](https://github.com/settings/tokens/new) — **Fine-grained token** → Account permissions → **Models: Read-only** 선택 필수")

        st.markdown("---")
        st.markdown("## 🔍 분석 대상")

        mode = st.radio("입력 방식", ["종목 코드 입력", "재무제표 이미지 업로드"])

        ticker_input = None
        uploaded_image = None

        if mode == "종목 코드 입력":
            ticker_input = st.text_input(
                "종목 코드",
                placeholder="예: 005930  AAPL  TSLA  0700.HK",
                help="한국 주식: 6자리 코드 자동 인식 | 미국: 영문 티커",
            )
        else:
            uploaded_image = st.file_uploader(
                "재무제표 이미지",
                type=["png", "jpg", "jpeg"],
                help="AI Vision이 표에서 수치를 자동 추출합니다.",
            )
            if uploaded_image:
                st.image(uploaded_image, use_column_width=True)

        st.markdown("---")
        st.markdown("## 💰 DCA 시뮬레이션")

        monthly_amount = st.number_input(
            "월 적립금 (원)",
            min_value=10_000,
            max_value=10_000_000,
            value=300_000,
            step=50_000,
            format="%d",
        )
        dca_years = st.selectbox("시뮬레이션 기간", [1, 3, 5], index=1)

        st.markdown("---")
        force_refresh = st.checkbox(
            "🔄 새 데이터로 재분석",
            value=False,
            help="저장된 캐시를 무시하고 최신 데이터로 다시 분석합니다",
        )
        analyze = st.button("🚀 분석 시작", use_container_width=True, type="primary")

    return {
        "api_key": api_key,
        "provider": "github",
        "mode": mode,
        "ticker_input": ticker_input,
        "uploaded_image": uploaded_image,
        "monthly_amount": monthly_amount,
        "dca_years": dca_years,
        "force_refresh": force_refresh,
        "analyze": analyze,
    }


# ──────────────────────────────────────────────
# Welcome screen
# ──────────────────────────────────────────────

def show_welcome() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>📊 퀀트 투자 분석 에이전트</h1>
          <p>세계 퀀트 대회 우승자의 투자 철학 · 데이터 기반 저평가 우량주 발굴 시스템</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    steps = [
        ("📡", "1단계", "데이터 수집", "재무제표 3개년 자동 수집 또는 이미지 OCR"),
        ("💬", "2단계", "AI 해석", "복잡한 수치를 쉬운 말로 번역"),
        ("🔍", "3단계", "저평가 분석", "PER·PBR 비교 & 체계적/비체계적 위험 판별"),
        ("🏆", "4단계", "투자 점수", "1~10점 매력도 & DCA 전략 제안"),
    ]
    for col, (icon, badge, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="card">
                  <span class="step-badge">{badge}</span><br>
                  <b>{icon} {title}</b><br>
                  <small>{desc}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        ---
        **지원 종목**
        - 🇰🇷 한국: `005930` (삼성전자)  `000660` (SK하이닉스)  `005380` (현대차)
        - 🇺🇸 미국: `AAPL`  `TSLA`  `NVDA`  `MSFT`
        - 🌏 기타: `0700.HK` (텐센트)  `7203.T` (도요타)

        > ⚠️ 투자 결과에 대한 책임은 투자자 본인에게 있습니다. 본 시스템은 참고 자료입니다.
        """,
    )


# ──────────────────────────────────────────────
# Tab 1 – Financial data charts
# ──────────────────────────────────────────────

def render_tab_financials(
    fetcher: FinancialDataFetcher | None,
    financials: pd.DataFrame,
    metrics: dict,
    profit_margins: pd.Series,
    price_history: pd.DataFrame,
    company_name: str,
    image_data: dict | None,
) -> None:
    st.markdown('<span class="step-badge">1단계 · 데이터 수집</span>', unsafe_allow_html=True)
    st.subheader("📊 3개년 재무 추이")

    if not financials.empty:
        years = _years_from_df(financials)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("매출액", "영업이익", "순이익률 (%)", "밸류에이션 지표"),
            vertical_spacing=0.18,
            horizontal_spacing=0.12,
        )

        def _bar(row_name: str, row: int, col: int, color: str, unit_label: str) -> None:
            if row_name not in financials.index:
                return
            vals = financials.loc[row_name].values / 1e8
            fig.add_trace(
                go.Bar(x=years, y=vals, marker_color=color, showlegend=False,
                       text=[f"{v:,.0f}억" for v in vals], textposition="outside"),
                row=row, col=col,
            )

        _bar("Total Revenue",    1, 1, "#1e4d7b", "억원")
        _bar("Operating Income", 1, 2, "#2e7d32", "억원")

        if not profit_margins.empty:
            m_years = _years_from_df(profit_margins.to_frame().T)
            fig.add_trace(
                go.Scatter(
                    x=m_years, y=profit_margins.values,
                    mode="lines+markers",
                    line=dict(color="#6a1b9a", width=3),
                    marker=dict(size=10),
                    showlegend=False,
                ),
                row=2, col=1,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

        val_labels, val_vals, val_colors = [], [], []
        for label, key, color in [
            ("PER", "pe_ratio", "#1565c0"),
            ("PBR", "pb_ratio", "#00695c"),
            ("ROE(%)", "roe",   "#6a1b9a"),
        ]:
            v = metrics.get(key)
            if v and v > 0:
                val_labels.append(label)
                val_vals.append(v * 100 if key == "roe" else v)
                val_colors.append(color)

        if val_labels:
            fig.add_trace(
                go.Bar(x=val_labels, y=val_vals, marker_color=val_colors,
                       showlegend=False,
                       text=[f"{v:.2f}" for v in val_vals], textposition="outside"),
                row=2, col=2,
            )

        fig.update_yaxes(title_text="억원", row=1, col=1)
        fig.update_yaxes(title_text="억원", row=1, col=2)
        fig.update_yaxes(title_text="%",    row=2, col=1)
        fig.update_layout(height=560, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 원본 재무 수치 (억원 단위)"):
            disp = financials.copy()
            disp.columns = years
            disp = (disp / 1e8).round(1)
            label_map = {
                "Total Revenue":    "매출액",
                "Operating Income": "영업이익",
                "Net Income":       "당기순이익",
                "Gross Profit":     "매출총이익",
            }
            disp.index = [label_map.get(r, r) for r in disp.index]
            st.dataframe(disp, use_container_width=True)

    elif image_data:
        st.subheader("📸 이미지 추출 데이터")
        years = image_data.get("years", [])
        unit = image_data.get("unit", "")
        data_map = {
            "매출액": image_data.get("revenue", []),
            "영업이익": image_data.get("operating_income", []),
            "당기순이익": image_data.get("net_income", []),
        }
        chart_data = {k: v for k, v in data_map.items() if v}
        if chart_data and years:
            df_img = pd.DataFrame(chart_data, index=years).T
            fig2 = go.Figure()
            for i, col in enumerate(df_img.columns):
                fig2.add_trace(
                    go.Bar(name=col, x=df_img.index, y=df_img[col],
                           marker_color=["#1e4d7b", "#2e7d32", "#c62828"][i % 3])
                )
            fig2.update_layout(barmode="group", title=f"재무 추이 ({unit})", height=380)
            st.plotly_chart(fig2, use_container_width=True)
        st.json(image_data)
    else:
        st.info("재무 데이터를 불러올 수 없습니다.")

    # Price history
    if not price_history.empty:
        st.subheader("📈 주가 추이 (3년)")
        fig_p = go.Figure()
        fig_p.add_trace(
            go.Scatter(
                x=price_history.index, y=price_history["Close"],
                mode="lines", name="주가",
                line=dict(color="#1e4d7b", width=2),
                fill="tozeroy", fillcolor="rgba(30,77,123,0.08)",
            )
        )
        h52 = metrics.get("fifty_two_week_high")
        l52 = metrics.get("fifty_two_week_low")
        if h52:
            fig_p.add_hline(y=h52, line_dash="dash", line_color="#2e7d32",
                            annotation_text=f"52주 최고 {h52:,.1f}")
        if l52:
            fig_p.add_hline(y=l52, line_dash="dash", line_color="#c62828",
                            annotation_text=f"52주 최저 {l52:,.1f}")
        fig_p.update_layout(height=360, hovermode="x unified", xaxis_title="날짜")
        st.plotly_chart(fig_p, use_container_width=True)


# ──────────────────────────────────────────────
# Tab 2 – AI plain-language summary
# ──────────────────────────────────────────────

def render_tab_ai_summary(
    plain_summary: str,
    metrics: dict,
    has_api_key: bool,
) -> None:
    st.markdown('<span class="step-badge">2단계 · AI 해석</span>', unsafe_allow_html=True)
    st.subheader("💬 재무 분석 요약")
    st.caption("퀀트 챔피언의 시각으로 쉽게 풀어드립니다")

    if plain_summary:
        st.markdown(plain_summary)
    elif not has_api_key:
        st.info("💡 사이드바에 GitHub PAT를 입력하면 AI 분석 요약을 받을 수 있습니다.")

    if metrics:
        st.subheader("📊 핵심 지표")
        cols = st.columns(4)
        items = [
            ("PER",   fmt_ratio(metrics.get("pe_ratio")),      "주가수익비율"),
            ("PBR",   fmt_ratio(metrics.get("pb_ratio")),      "주가순자산비율"),
            ("ROE",   fmt_pct(metrics.get("roe")),             "자기자본이익률"),
            ("순이익률", fmt_pct(metrics.get("profit_margin")), "Net Profit Margin"),
        ]
        for col, (label, val, desc) in zip(cols, items):
            with col:
                st.metric(f"{label}", val, help=desc)

        cols2 = st.columns(4)
        items2 = [
            ("부채비율",   fmt_ratio(metrics.get("debt_to_equity")), "D/E Ratio"),
            ("유동비율",   fmt_ratio(metrics.get("current_ratio")),  "Current Ratio"),
            ("베타",      fmt_ratio(metrics.get("beta"), 2),        "시장 민감도"),
            ("배당수익률", fmt_pct(metrics.get("dividend_yield")),   "Dividend Yield"),
        ]
        for col, (label, val, desc) in zip(cols2, items2):
            with col:
                st.metric(f"{label}", val, help=desc)


# ──────────────────────────────────────────────
# Tab 3 – Undervaluation
# ──────────────────────────────────────────────

def render_tab_undervaluation(
    quant_result: dict,
    undervaluation: dict,
    peer_df: pd.DataFrame,
    metrics: dict,
    company_name: str,
    has_api_key: bool,
) -> None:
    st.markdown('<span class="step-badge">3단계 · 저평가 분석</span>', unsafe_allow_html=True)
    st.subheader("🔍 저평가 여부 & 위험 유형 판별")

    col_l, col_r = st.columns([1, 1])

    # Left: quant score breakdown
    with col_l:
        st.markdown("#### 퀀트 점수 세부 내역 (0–5점)")
        max_map = {"per": 1.5, "pbr": 1.5, "profitability": 1.0, "financial_health": 1.0}
        label_map = {
            "per": "PER 밸류에이션",
            "pbr": "PBR 밸류에이션",
            "profitability": "수익성",
            "financial_health": "재무건전성",
        }
        for cat, score in quant_result.get("breakdown", {}).items():
            max_s = max_map.get(cat, 1.0)
            st.markdown(f"**{label_map.get(cat, cat)}** — `{score:.2f} / {max_s:.1f}`")
            st.progress(score / max_s)
            exp = quant_result.get("explanations", {}).get(cat, "")
            if exp:
                st.caption(f"↪ {exp}")
            st.markdown("")

        grade = quant_result.get("grade", "")
        total = quant_result.get("total_score", 0)
        color = score_color(total, 5.0)
        st.markdown(
            f"**퀀트 등급:** <span style='color:{color}; font-size:1.1rem'>{grade}</span>",
            unsafe_allow_html=True,
        )

    # Right: peer comparison
    with col_r:
        if peer_df is not None and not peer_df.empty:
            st.markdown("#### 동종 업계 PER / PBR 비교")

            def _peer_chart(metric_col: str, title: str, ref_line: float | None = None) -> None:
                comp = [{"Company": f"▶ {company_name}", metric_col: metrics.get(metric_col), "is_target": True}]
                for _, row in peer_df.iterrows():
                    comp.append({"Company": row["name"], metric_col: row.get(metric_col), "is_target": False})
                df_c = pd.DataFrame(comp).dropna(subset=[metric_col])
                df_c = df_c[df_c[metric_col] > 0]
                if df_c.empty:
                    return
                colors = ["#c62828" if r["is_target"] else "#78909c" for _, r in df_c.iterrows()]
                fig = go.Figure(
                    go.Bar(
                        x=df_c["Company"], y=df_c[metric_col],
                        marker_color=colors,
                        text=[f"{v:.2f}배" for v in df_c[metric_col]],
                        textposition="outside",
                    )
                )
                peers_only = df_c[~df_c["is_target"]][metric_col]
                if not peers_only.empty:
                    med = peers_only.median()
                    fig.add_hline(y=med, line_dash="dash", line_color="orange",
                                  annotation_text=f"중앙값 {med:.2f}")
                if ref_line is not None:
                    fig.add_hline(y=ref_line, line_dash="dot", line_color="red",
                                  annotation_text=f"기준 {ref_line}")
                fig.update_layout(title=title, height=280, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            _peer_chart("pe_ratio", "PER 비교")
            _peer_chart("pb_ratio", "PBR 비교", ref_line=1.0)

    # AI undervaluation
    if undervaluation and has_api_key:
        st.markdown("---")
        st.subheader("🤖 AI 저평가 종합 판단")

        is_under = undervaluation.get("is_undervalued", False)
        level = undervaluation.get("undervaluation_level", "")
        risk_type = undervaluation.get("risk_type", "")

        col_a, col_b = st.columns(2)
        with col_a:
            if is_under:
                st.success(f"✅ {level}")
            else:
                st.warning(f"⚠️ {level}")
            st.info(f"📊 위험 유형: **{risk_type}**")
            exp = undervaluation.get("risk_explanation", "")
            if exp:
                st.caption(exp)

        with col_b:
            st.markdown("**PER 분석**")
            st.write(undervaluation.get("per_assessment", ""))
            st.markdown("**PBR 분석**")
            st.write(undervaluation.get("pbr_assessment", ""))

        st.markdown(
            f'<div class="card">{undervaluation.get("summary", "")}</div>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# Tab 4 – Investment score + DCA
# ──────────────────────────────────────────────

def render_tab_investment_score(
    quant_result: dict,
    investment_data: dict,
    dca_results: dict,
    monthly_amount: float,
    dca_years: int,
    company_name: str,
    currency: str,
    has_api_key: bool,
) -> None:
    st.markdown('<span class="step-badge">4단계 · 투자 점수 & 전략</span>', unsafe_allow_html=True)
    st.subheader("🏆 투자 매력도 점수")

    quant_total = quant_result.get("total_score", 0.0)
    ai_total = float(investment_data.get("ai_score", 0)) if investment_data else 0.0
    total_score = round(quant_total + ai_total, 1)
    color = score_color(total_score, 10.0)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 퀀트 점수")
        st.markdown(
            f'<div class="score-big" style="color:#1e4d7b">{quant_total:.1f}'
            f'<span style="font-size:1.4rem; color:#666"> / 5</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("데이터 정량 분석")
    with c2:
        st.markdown("#### AI 점수")
        ai_disp = f"{ai_total:.1f}" if investment_data else "N/A"
        st.markdown(
            f'<div class="score-big" style="color:#2e7d32">{ai_disp}'
            f'<span style="font-size:1.4rem; color:#666"> / 5</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("AI 정성 분석")
    with c3:
        st.markdown("#### 종합 투자 매력도")
        st.markdown(
            f'<div class="score-big" style="color:{color}">{total_score}'
            f'<span style="font-size:1.4rem; color:#666"> / 10</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(quant_result.get("grade", ""))

    # Gauge chart
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=total_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "투자 매력도 (10점 만점)", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 3],  "color": "#ffcdd2"},
                    {"range": [3, 5],  "color": "#fff9c4"},
                    {"range": [5, 7],  "color": "#c8e6c9"},
                    {"range": [7, 10], "color": "#a5d6a7"},
                ],
                "threshold": {
                    "line": {"color": "#333", "width": 3},
                    "thickness": 0.75,
                    "value": total_score,
                },
            },
        )
    )
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Investment thesis & DCA recommendation
    if investment_data and has_api_key:
        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 💡 투자 논거")
            thesis = investment_data.get("investment_thesis", "")
            if thesis:
                st.markdown(f'<div class="card">{thesis}</div>', unsafe_allow_html=True)

            st.markdown("#### ⚠️ 주요 위험 요인")
            for r in investment_data.get("key_risks", []):
                st.markdown(f"• {r}")

            st.markdown("#### 🎯 매수 신호")
            for t in investment_data.get("buy_triggers", []):
                st.markdown(f"✅ {t}")

            horizon = investment_data.get("investment_horizon", "")
            if horizon:
                st.info(f"⏱ 추천 투자 기간: **{horizon}**")

        with col_right:
            dca_rec = investment_data.get("dca_recommendation", {})
            if dca_rec:
                st.markdown("#### 💰 적립식 투자 비중 추천")
                stock_pct = dca_rec.get("individual_stock_pct", 30)
                index_pct = dca_rec.get("index_fund_pct", 70)

                fig_pie = go.Figure(
                    go.Pie(
                        labels=[f"{company_name} ({stock_pct}%)", f"S&P 500 지수 ({index_pct}%)"],
                        values=[stock_pct, index_pct],
                        marker_colors=["#1e4d7b", "#90a4ae"],
                        hole=0.45,
                        textinfo="label+percent",
                    )
                )
                fig_pie.update_layout(height=280, showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)

                rationale = dca_rec.get("rationale", "")
                strategy = dca_rec.get("dca_strategy", "")
                if rationale:
                    st.caption(f"**이유:** {rationale}")
                if strategy:
                    st.caption(f"**전략:** {strategy}")

    # DCA simulation
    st.markdown("---")
    st.subheader("📈 DCA 시뮬레이션 비교")
    st.caption(
        f"월 {monthly_amount:,.0f}원 적립 · {dca_years}년 · 수익률 기준 비교 (통화 무관)"
    )

    if dca_results:
        total_months = dca_years * 12
        total_invested = monthly_amount * total_months

        st.markdown(f"**총 납입 원금:** {total_invested:,.0f}원 ({monthly_amount:,.0f}원 × {total_months}개월)")
        st.markdown("---")

        col_d1, col_d2 = st.columns(2)

        def _dca_metric(col, label: str, result: dict, currency_label: str = "") -> None:
            with col:
                ret = result["return_pct"]
                invested = result["total_invested"]
                final = result["final_value_units"]
                profit = final - invested

                color_d = "normal" if ret >= 0 else "inverse"
                st.metric(
                    label,
                    f"{ret:+.2f}%",
                    f"{'▲' if ret >= 0 else '▼'} {abs(profit):,.0f}{currency_label} 수익",
                    delta_color=color_d,
                )
                st.caption(f"납입 원금: {invested:,.0f}{currency_label}  →  최종 평가액: {final:,.0f}{currency_label}")

        if "stock" in dca_results:
            _dca_metric(col_d1, f"📊 {company_name}", dca_results["stock"], currency_label=f" {currency}")
        if "sp500" in dca_results:
            _dca_metric(col_d2, "📊 S&P 500 (SPY)", dca_results["sp500"], currency_label=" USD")

        # Normalised price chart
        if "stock" in dca_results and "sp500" in dca_results:
            s_prices = dca_results["stock"]["prices"]
            i_prices = dca_results["sp500"]["prices"]
            s_norm = s_prices / s_prices.iloc[0] * 100
            i_norm = i_prices / i_prices.iloc[0] * 100

            fig_dca = go.Figure()
            fig_dca.add_trace(
                go.Scatter(x=s_norm.index, y=s_norm.values,
                           name=company_name, line=dict(color="#1e4d7b", width=2))
            )
            fig_dca.add_trace(
                go.Scatter(x=i_norm.index, y=i_norm.values,
                           name="S&P 500", line=dict(color="#c62828", width=2, dash="dash"))
            )
            fig_dca.add_hline(y=100, line_dash="dot", line_color="gray",
                              annotation_text="기준 (100)")
            fig_dca.update_layout(
                title=f"주가 성과 비교 (기준: {dca_years}년 전 = 100)",
                xaxis_title="날짜", yaxis_title="성과 지수",
                height=380, hovermode="x unified",
            )
            st.plotly_chart(fig_dca, use_container_width=True)
    else:
        st.info("종목 코드 입력 시 DCA 시뮬레이션이 계산됩니다.")


# ──────────────────────────────────────────────
# Results renderer  (live 분석 & 캐시 로드 공용)
# ──────────────────────────────────────────────

def _render_results(
    data: dict,
    api_key: str,
    monthly_amount: float,
    dca_years: int,
    cache_info: dict | None = None,
) -> None:
    company_name   = data["company_name"]
    sector         = data["sector"]
    industry       = data["industry"]
    currency       = data["currency"]
    metrics        = data.get("metrics", {})
    financials     = data.get("financials", pd.DataFrame())
    price_history  = data.get("price_history", pd.DataFrame())
    profit_margins = data.get("profit_margins", pd.Series(dtype=float))
    peer_df        = data.get("peer_df", pd.DataFrame())
    dca_results    = data.get("dca_results", {})
    plain_summary   = data.get("plain_summary", "")
    undervaluation  = data.get("undervaluation", {})
    investment_data = data.get("investment_data", {})
    quant_result    = data.get("quant_result", {})
    image_data      = data.get("image_data")

    if cache_info:
        next_r = cache_info["next_refresh"]
        quarter = cache_info["quarter"]
        cached_at = cache_info["cached_at"]
        st.info(
            f"📦 캐시된 분석 결과 · **{cached_at}** 저장 · "
            f"다음 분기 갱신 예정: **{next_r}** ({quarter})  "
            f"　*(사이드바 🔄 체크 후 재분석하면 최신 데이터로 업데이트됩니다)*"
        )

    st.markdown(
        f"""
        <div class="hero">
          <h1>📋 {company_name} 분석 결과</h1>
          <p>{sector} · {industry} · 통화: {currency}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        price = metrics.get("current_price")
        mc    = metrics.get("market_cap")
        per   = metrics.get("pe_ratio")
        pbr   = metrics.get("pb_ratio")
        with c1:
            st.metric("현재 주가", f"{price:,.2f} {currency}" if price else "N/A")
        with c2:
            st.metric("시가총액", fmt_usd(mc) if currency == "USD" else fmt_number(mc, currency))
        with c3:
            st.metric("PER", fmt_ratio(per))
        with c4:
            st.metric("PBR", fmt_ratio(pbr))

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 재무 데이터", "💬 AI 해석", "🔍 저평가 분석", "🏆 투자 점수 & 전략"]
    )
    with tab1:
        render_tab_financials(
            None, financials, metrics, profit_margins, price_history,
            company_name, image_data,
        )
    with tab2:
        render_tab_ai_summary(plain_summary, metrics, bool(api_key))
    with tab3:
        render_tab_undervaluation(
            quant_result, undervaluation, peer_df, metrics, company_name, bool(api_key)
        )
    with tab4:
        render_tab_investment_score(
            quant_result, investment_data, dca_results,
            monthly_amount, dca_years, company_name, currency, bool(api_key),
        )


# ──────────────────────────────────────────────
# Main orchestration
# ──────────────────────────────────────────────

def run_analysis(cfg: dict) -> None:
    api_key = cfg["api_key"]
    ticker_input = cfg["ticker_input"]
    uploaded_image = cfg["uploaded_image"]
    monthly_amount = cfg["monthly_amount"]
    dca_years = cfg["dca_years"]

    agent = AnalystAgent(api_key, provider=cfg.get("provider", "openai")) if api_key else None
    scorer = QuantScorer()

    # ── 캐시 확인 ─────────────────────────────
    force_refresh = cfg.get("force_refresh", False)
    if ticker_input and not force_refresh:
        resolved = FinancialDataFetcher._resolve_ticker(ticker_input)
        if cache_manager.is_valid(resolved):
            cached = cache_manager.load(resolved)
            if cached:
                _render_results(
                    data=cached["data"],
                    api_key=api_key,
                    monthly_amount=monthly_amount,
                    dca_years=dca_years,
                    cache_info={
                        "cached_at":    cached["cached_at"],
                        "quarter":      cached["quarter"],
                        "next_refresh": cached["next_refresh"],
                    },
                )
                return

    status = st.empty()
    progress = st.progress(0)

    # ── Collect data ──────────────────────────
    status.text("📡 데이터 수집 중…")
    progress.progress(10)

    fetcher: FinancialDataFetcher | None = None
    image_data: dict | None = None
    company_name = "분석 대상"

    if uploaded_image and agent:
        status.text("🔍 이미지에서 재무 데이터 추출 중 (AI Vision)…")
        img_bytes = uploaded_image.read()
        try:
            image_data = agent.analyze_financial_image(img_bytes)
            company_name = image_data.get("company_name", "알 수 없음")
            st.success(f"✅ 이미지 분석 완료 — {company_name}")
        except Exception as e:
            st.error(f"이미지 분석 실패: {e}")
            return

    elif ticker_input:
        try:
            fetcher = FinancialDataFetcher(ticker_input)
            company_name = fetcher.get_company_name()
            if company_name == fetcher.ticker:
                st.warning(f"⚠️ '{fetcher.ticker}' 데이터가 제한적입니다. Yahoo Finance 요청 한도일 수 있으니 잠시 후 재시도해주세요.")
            status.text(f"📊 {company_name} 데이터 수집 중…")
        except Exception as e:
            st.error(f"티커 조회 실패 ({ticker_input}): {e}\n\n💡 **해결 방법:** `pip install --upgrade yfinance` 실행 후 재시도")
            return
    else:
        st.warning("API 키가 필요합니다 (이미지 모드).")
        return

    # ── Fetch all financial data ───────────────
    progress.progress(30)
    financials = fetcher.get_financials_3yr() if fetcher else pd.DataFrame()
    metrics = fetcher.get_key_metrics() if fetcher else {}
    price_history = fetcher.get_price_history() if fetcher else pd.DataFrame()
    profit_margins = fetcher.calculate_net_profit_margin() if fetcher else pd.Series(dtype=float)
    sector = fetcher.get_sector() if fetcher else ""
    industry = fetcher.get_industry() if fetcher else ""
    currency = fetcher.get_currency() if fetcher else "KRW"

    progress.progress(45)
    status.text("🏭 동종 업계 데이터 수집 중…")
    peer_df = fetcher.get_peer_metrics() if fetcher else pd.DataFrame()
    peer_averages = scorer.calculate_peer_averages(peer_df)

    progress.progress(55)
    status.text("💰 DCA 시뮬레이션 계산 중…")
    dca_results = fetcher.calculate_dca_comparison(monthly_amount, dca_years) if fetcher else {}

    # ── AI analysis ───────────────────────────
    plain_summary = ""
    undervaluation: dict = {}
    investment_data: dict = {}

    if agent:
        progress.progress(65)
        status.text("🤖 AI 재무 해석 중…")
        fin_summary = {
            "company": company_name,
            "sector": sector,
            "industry": industry,
            "metrics": {k: v for k, v in metrics.items() if v is not None},
            "profit_margin_trend": profit_margins.to_dict() if not profit_margins.empty else {},
        } if fetcher else (image_data or {})
        fin_summary = _sanitize_json(fin_summary)

        try:
            plain_summary = agent.summarize_financials(fin_summary)
        except Exception as e:
            plain_summary = f"⚠️ AI 요약 실패: {e}"

        if fetcher:
            progress.progress(75)
            status.text("📈 저평가 분석 중…")
            try:
                undervaluation = agent.analyze_undervaluation(
                    metrics,
                    _sanitize_json({
                        "peer_averages": peer_averages,
                        "peers": peer_df.to_dict() if not peer_df.empty else {},
                    }),
                )
            except Exception as e:
                undervaluation = {"summary": f"분석 실패: {e}", "is_undervalued": False}

            progress.progress(87)
            status.text("🏆 투자 점수 산정 중…")
            quant_result = scorer.calculate_score(metrics, peer_averages)
            try:
                investment_data = agent.generate_investment_score(
                    {
                        "company": company_name,
                        "sector": sector,
                        "metrics": {k: v for k, v in metrics.items() if v is not None},
                        "quant_score": quant_result["total_score"],
                        "undervaluation": undervaluation,
                    },
                    quant_result["total_score"],
                )
            except Exception as e:
                investment_data = {"ai_score": 0, "score_reasoning": f"실패: {e}"}
        else:
            quant_result = scorer.calculate_score({}, {})
    else:
        quant_result = scorer.calculate_score(metrics, peer_averages) if fetcher else scorer.calculate_score({}, {})

    progress.progress(100)
    status.empty()
    progress.empty()

    # ── 캐시 저장 ─────────────────────────────
    if fetcher:
        try:
            cache_manager.save(fetcher.ticker, {
                "company_name":   company_name,
                "sector":         sector,
                "industry":       industry,
                "currency":       currency,
                "metrics":        metrics,
                "financials":     financials,
                "price_history":  price_history,
                "profit_margins": profit_margins,
                "peer_df":        peer_df,
                "dca_results":    dca_results,
                "plain_summary":  plain_summary,
                "undervaluation": undervaluation,
                "investment_data": investment_data,
                "quant_result":   quant_result,
                "image_data":     None,
            })
        except Exception:
            pass  # 캐시 저장 실패는 비치명적

    # ── 결과 렌더링 ───────────────────────────
    _render_results(
        data={
            "company_name":   company_name,
            "sector":         sector,
            "industry":       industry,
            "currency":       currency,
            "metrics":        metrics,
            "financials":     financials,
            "price_history":  price_history,
            "profit_margins": profit_margins,
            "peer_df":        peer_df,
            "dca_results":    dca_results,
            "plain_summary":  plain_summary,
            "undervaluation": undervaluation,
            "investment_data": investment_data,
            "quant_result":   quant_result,
            "image_data":     image_data,
        },
        api_key=api_key,
        monthly_amount=monthly_amount,
        dca_years=dca_years,
    )


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main() -> None:
    cfg = build_sidebar()

    if cfg["analyze"]:
        run_analysis(cfg)
    else:
        show_welcome()


if __name__ == "__main__":
    main()
