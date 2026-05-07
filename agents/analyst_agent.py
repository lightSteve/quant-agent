"""
AI-powered financial analyst agent.
Supports OpenAI API and GitHub Models API (GitHub Copilot 구독자 무료 사용 가능).

GitHub Models 사용법:
  1. https://github.com/settings/tokens 에서 PAT 생성 (권한 불필요, 기본 설정)
  2. provider="github", api_key=<PAT> 로 초기화
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, Literal

from openai import OpenAI

GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"

SYSTEM_PROMPT = (
    "너는 세계 퀀트 대회 우승자의 투자 철학을 가진 전문 금융 애널리스트야. "
    "복잡한 재무 수치를 일반 투자자가 이해하기 쉬운 언어로 번역하고, "
    "데이터에 기반해 냉철하게 저평가된 기회를 포착하는 역할을 수행해. "
    "항상 데이터를 근거로 분석하고, 투기적 추측보다는 정량적 근거를 우선시해. "
    "분석 결과는 반드시 한국어로 제공해."
)


class AnalystAgent:
    """GPT-4o powered financial analyst.

    provider="openai"  → OpenAI API 직접 사용
    provider="github"  → GitHub Models API (Copilot 구독자 무료)
    """

    def __init__(
        self,
        api_key: str,
        provider: Literal["openai", "github"] = "openai",
    ) -> None:
        if provider == "github":
            self.client = OpenAI(
                api_key=api_key,
                base_url=GITHUB_MODELS_BASE_URL,
            )
            self.model = "gpt-4o"
        else:
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4o"

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _chat(
        self,
        user_prompt: str,
        *,
        temperature: float = 0.6,
        max_tokens: int = 1200,
        json_mode: bool = False,
    ) -> str:
        kwargs: Dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            err_str = str(e)
            if "401" in err_str and "models" in err_str:
                raise PermissionError(
                    "GitHub PAT에 Models 권한이 없습니다.\n\n"
                    "해결 방법:\n"
                    "1. https://github.com/settings/tokens/new 접속\n"
                    "2. Fine-grained token 선택\n"
                    "3. Account permissions → Models → Read-only 설정\n"
                    "4. 생성된 토큰을 사이드바에 입력"
                ) from e
            raise

    # ------------------------------------------------------------------
    # Step 1 – Image OCR
    # ------------------------------------------------------------------

    def analyze_financial_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract financial figures from an uploaded image via Vision API.
        Returns structured JSON with revenue, operating income, net income, etc.
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        extraction_prompt = (
            "이 재무제표 이미지에서 다음 데이터를 추출해 JSON으로 반환해줘.\n\n"
            "반환 형식:\n"
            "{\n"
            '  "company_name": "회사명",\n'
            '  "years": ["연도1", "연도2", "연도3"],\n'
            '  "revenue": [값1, 값2, 값3],\n'
            '  "operating_income": [값1, 값2, 값3],\n'
            '  "net_income": [값1, 값2, 값3],\n'
            '  "debt_ratio": 값 또는 null,\n'
            '  "eps": [값1, 값2, 값3] 또는 null,\n'
            '  "unit": "단위 (억원, 백만원, 백만달러 등)",\n'
            '  "notes": "특이사항이나 주의사항"\n'
            "}\n\n"
            "숫자는 콤마 없이 순수 숫자로 입력하고, 데이터가 없으면 null을 사용해."
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": extraction_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content or "{}")

    # ------------------------------------------------------------------
    # Step 2 – Plain language summary
    # ------------------------------------------------------------------

    def summarize_financials(self, financial_data: Dict[str, Any]) -> str:
        """
        Translate financial data into plain Korean for non-expert investors.
        Answers three core questions:
          1. Is the company earning more than before?
          2. Is debt properly managed?
          3. What is driving recent earnings growth?
        """
        data_str = json.dumps(financial_data, ensure_ascii=False, indent=2)
        prompt = (
            f"다음 재무 데이터를 분석해주세요:\n\n{data_str}\n\n"
            "위 데이터를 바탕으로 초보 투자자도 이해할 수 있게 아래 3가지 질문에 각각 2~3문장으로 답해주세요. "
            "숫자보다 쉬운 표현을 사용하세요.\n\n"
            "**질문 1.** 돈을 예전보다 잘 벌고 있는가? (매출과 이익 추세)\n"
            "**질문 2.** 빚은 적절하게 관리되고 있는가? (부채비율과 유동성)\n"
            "**질문 3.** 최근 실적 성장의 핵심 원인은 무엇인가?\n"
        )
        return self._chat(prompt, temperature=0.7, max_tokens=900)

    # ------------------------------------------------------------------
    # Step 3 – Undervaluation analysis
    # ------------------------------------------------------------------

    def analyze_undervaluation(
        self,
        metrics: Dict[str, Any],
        peer_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Determine if the stock is undervalued vs peers.
        Classifies risk as systematic vs unsystematic.
        """
        prompt = (
            "다음 종목의 밸류에이션을 분석하고, 저평가 여부를 판단해줘.\n\n"
            f"현재 종목 지표:\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n\n"
            f"동종 업계 비교:\n{json.dumps(peer_context, ensure_ascii=False, indent=2)}\n\n"
            "다음 JSON 형식으로 답해줘:\n"
            "{\n"
            '  "per_assessment": "PER 분석 결과",\n'
            '  "pbr_assessment": "PBR 분석 결과",\n'
            '  "risk_type": "체계적 위험 / 비체계적 위험 / 혼합",\n'
            '  "risk_explanation": "위험 유형 및 근거",\n'
            '  "is_undervalued": true 또는 false,\n'
            '  "undervaluation_level": "심각한 저평가 / 저평가 / 적정가치 / 고평가",\n'
            '  "summary": "퀀트 챔피언 시각에서 본 종합 판단 (3문장)"\n'
            "}"
        )
        raw = self._chat(prompt, temperature=0.3, max_tokens=1200, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"summary": raw, "is_undervalued": False, "undervaluation_level": "분석 불가"}

    # ------------------------------------------------------------------
    # Step 4 – Investment score + DCA strategy
    # ------------------------------------------------------------------

    def generate_investment_score(
        self,
        all_data: Dict[str, Any],
        quant_score: float,
    ) -> Dict[str, Any]:
        """
        Produce an AI qualitative score (0–5) and DCA strategy recommendation.
        """
        prompt = (
            "아래 종합 데이터와 퀀트 점수를 바탕으로 AI 정성 투자 점수와 전략을 제시해줘.\n\n"
            f"종합 데이터:\n{json.dumps(all_data, ensure_ascii=False, indent=2)}\n\n"
            f"퀀트 점수 (0-5점): {quant_score}\n\n"
            "다음 JSON 형식으로 답해줘:\n"
            "{\n"
            '  "ai_score": 0~5 사이 숫자,\n'
            '  "score_reasoning": "점수 부여 근거 (2문장)",\n'
            '  "investment_thesis": "투자 핵심 논거 (3문장)",\n'
            '  "key_risks": ["위험 요인 1", "위험 요인 2", "위험 요인 3"],\n'
            '  "dca_recommendation": {\n'
            '    "individual_stock_pct": 개별주식 비중(0~100),\n'
            '    "index_fund_pct": 지수펀드 비중,\n'
            '    "rationale": "비중 추천 이유",\n'
            '    "dca_strategy": "적립식 투자 전략 (2문장)"\n'
            "  },\n"
            '  "investment_horizon": "추천 투자 기간",\n'
            '  "buy_triggers": ["매수 신호 1", "매수 신호 2"]\n'
            "}"
        )
        raw = self._chat(prompt, temperature=0.5, max_tokens=1500, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "ai_score": 0,
                "score_reasoning": "점수 산정 실패",
                "investment_thesis": "",
                "key_risks": [],
                "dca_recommendation": {},
                "investment_horizon": "",
                "buy_triggers": [],
            }

    # ------------------------------------------------------------------
    # Step 5 – Multi-stock comparison & recommendation
    # ------------------------------------------------------------------

    def compare_and_recommend(self, summaries: list) -> dict:
        """
        Compare multiple stocks and pick the best one given current market conditions.
        summaries: list of dicts with ticker, company_name, sector, quant_score, ai_score,
                   total_score, per, pbr, profit_margin, undervaluation_level, grade.
        """
        prompt = (
            "다음 여러 종목을 비교 분석하고, 현재 시장 상황에서 가장 투자 매력도가 높은 종목 1개를 추천해줘.\n\n"
            f"종목 비교 데이터:\n{json.dumps(summaries, ensure_ascii=False, indent=2)}\n\n"
            "현재 금리 환경, 섹터 모멘텀, 밸류에이션 매력도, 성장성, 재무건전성을 종합적으로 고려해서 판단해줘.\n\n"
            "반드시 다음 JSON 형식으로 답해줘:\n"
            "{\n"
            '  "recommended_ticker": "추천 티커",\n'
            '  "recommended_company": "추천 회사명",\n'
            '  "reasoning": "추천 핵심 이유 (3~4문장, 구체적 수치 근거 포함)",\n'
            '  "market_context": "현재 시장 상황 분석 (2문장)",\n'
            '  "rank_list": [\n'
            '    {"rank": 1, "ticker": "티커", "company": "회사명", "reason": "이 순위인 이유 (1문장)"},\n'
            '    ...\n'
            "  ],\n"
            '  "caution": "투자 시 공통 주의사항 (1~2문장)"\n'
            "}"
        )
        raw = self._chat(prompt, temperature=0.4, max_tokens=1800, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "recommended_ticker": summaries[0]["ticker"] if summaries else "",
                "recommended_company": summaries[0]["company_name"] if summaries else "",
                "reasoning": raw,
                "market_context": "",
                "rank_list": [],
                "caution": "",
            }
