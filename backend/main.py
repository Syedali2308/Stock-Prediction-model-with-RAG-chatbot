"""
main.py  –  FastAPI server exposing /api/chat (RAG + Ollama pipeline).
"""

import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

from engine import predict_next_day_price

app = FastAPI(
    title="Stock RAG Dashboard API",
    version="1.0.0",
    description="Live stock forecasting + RAG investment chatbot powered by Gemma 3.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"


# ── request / response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    ticker:     str
    user_query: str


class ChatResponse(BaseModel):
    ticker:                    str
    current_price:             float | None
    predicted_price:           float | None
    pct_change:                float | None
    direction:                 str
    sentiment_score:           float | None
    historical_context_string: str
    data_source:               str
    llm_response:              str
    error:                     str | None


# ── prompt orchestration ──────────────────────────────────────────────────────

_HISTORICAL_PATTERNS = (
    r"\bprice\b", r"\bclose\b", r"\bclosing\b", r"\bhistorical\b",
    r"\byesterday\b", r"\bparso\b", r"\bdin\b", r"\bdays?\s+ago\b",
    r"\blast\s+\d+\s", r"\bpichle\b", r"\bpehle\b", r"\bdate\b",
    r"\bkitna\s+tha\b", r"\bkya\s+tha\b",
)

_INVESTMENT_PATTERNS = (
    r"\bbuy\b", r"\bsell\b", r"\binvest\b", r"\bkhareed", r"\bavoid\b",
    r"\bshould\s+i\b", r"\bkyu\b", r"\bkya\s+kar", r"\brecommend\b",
    r"\bpros\b", r"\bcons\b", r"\brationale\b", r"\bconviction\b",
    r"\bhold\b", r"\bexit\b", r"\bna\s+khareed",
)


def _detect_query_intent(query: str) -> str:
    q = query.lower().strip()
    if any(re.search(p, q) for p in _HISTORICAL_PATTERNS):
        return "historical"
    if any(re.search(p, q) for p in _INVESTMENT_PATTERNS):
        return "investment"
    return "general"


def _extract_history_days(query: str, default: int = 3) -> int:
    q = query.lower()
    m = re.search(r"(?:last|pichle|pehle)\s+(\d+)\s*(?:din|day|days)?", q)
    if m:
        return max(1, min(int(m.group(1)), 5))
    if "yesterday" in q or "kal" in q:
        return 1
    if "parso" in q:
        return 2
    return default


def _format_historical_table(ticker: str, days: int, rows: list[dict]) -> str:
    """Deterministic markdown table — used when Gemma ignores format rules."""
    window = rows[-days:] if rows else []
    lines = [
        f"Here is the price data for {ticker} over the last {days} days:",
        "",
        "| Date | Close Price |",
        "| :--- | :--- |",
    ]
    for row in window:
        lines.append(f"| {row['date']} | ₹{row['close']:,.2f} |")
    if len(window) < days:
        lines.append(f"\nNote: only {len(window)} days of historical data available.")
    return "\n".join(lines)


def _format_investment_matrix(
    ticker: str,
    current,
    predicted,
    pct_change,
    direction,
    sentiment,
    history: str,
) -> str:
    """Deterministic investment matrix — used when Gemma ignores format rules."""
    bull = "bullish" in direction.lower() or (pct_change is not None and pct_change >= 0)
    sent_label = "positive" if (sentiment or 0) >= 0 else "negative"

    buy_points = [
        f"Model projects next-day close at ₹{predicted} vs today's ₹{current} ({direction}).",
        f"News sentiment score is {sentiment:+.4f} ({sent_label}), supporting near-term momentum.",
        f"Recent closes in RECENT TRADING HISTORY show actionable levels for {ticker}.",
    ]
    avoid_points = [
        f"Forecast implies {pct_change}% move — volatility may exceed risk tolerance.",
        f"Sentiment at {sentiment:+.4f} can reverse quickly if headline flow deteriorates.",
        "Single-day LSTM projections are probabilistic, not guaranteed execution prices.",
    ]
    summary = (
        f"{ticker} screens as {'constructive' if bull else 'cautious'} on model + sentiment fusion today. "
        f"Size positions to recent closes ({history.split(',')[-1].strip() if history else 'N/A'}) "
        "and reassess after the next session."
    )

    return (
        "**🟢 Why You Should Buy:**\n"
        f"- {buy_points[0]}\n"
        f"- {buy_points[1]}\n"
        f"- {buy_points[2]}\n\n"
        "**🔴 Why You Should Avoid:**\n"
        f"- {avoid_points[0]}\n"
        f"- {avoid_points[1]}\n"
        f"- {avoid_points[2]}\n\n"
        "**💡 Final Recommendation Summary:**\n"
        f"{summary}"
    )


def _enforce_structured_response(
    intent: str,
    llm_text: str,
    ticker: str,
    days: int,
    rows: list[dict],
    current,
    predicted,
    pct_change,
    direction,
    sentiment,
    history: str,
) -> str:
    if intent == "historical":
        if "| Date |" not in llm_text or "Close Price" not in llm_text:
            return _format_historical_table(ticker, days, rows)
        return llm_text

    if intent == "investment":
        required = ("🟢 Why You Should Buy", "🔴 Why You Should Avoid", "💡 Final Recommendation Summary")
        if not all(marker in llm_text for marker in required):
            return _format_investment_matrix(
                ticker, current, predicted, pct_change, direction, sentiment, history
            )
        return llm_text

    return llm_text


def _build_system_prompt(
    ticker: str,
    current,
    predicted,
    pct_change,
    direction,
    sentiment,
    history: str,
    data_source: str,
    user_query: str,
) -> str:
    intent = _detect_query_intent(user_query)
    days_window = _extract_history_days(user_query)
    source_note = (
        "Live yfinance data."
        if data_source == "live"
        else "Synthetic fail-safe matrix (network/yfinance blocked — values are internally consistent for formatting)."
    )

    base = f"""You are a seasoned equity research analyst and investment strategist.
You provide data-backed, precise investment commentary in professional financial prose.

DATA SOURCE: {data_source.upper()} — {source_note}

LIVE TECHNICAL DATA FOR {ticker}:
  • Current Close Price  : ₹{current}
  • Predicted Next-Day   : ₹{predicted}
  • Expected % Change    : {pct_change}%
  • Market Direction     : {direction}
  • News Sentiment Score : {sentiment} (range -1 to +1; positive = bullish sentiment)

RECENT TRADING HISTORY (LAST 5 DAYS):
  {history}

CRITICAL RULE: Even if DATA SOURCE is SYNTHETIC or network-constrained, you MUST still obey
the exact output format for the detected intent. Never reply with unstructured plain paragraphs
when a structured format is required below.

DETECTED USER INTENT: {intent.upper()}
"""

    if intent == "historical":
        return base + f"""
MANDATORY RESPONSE FORMAT — HISTORICAL PRICE ANALYTICS (NO EXCEPTIONS):

Your entire reply MUST be ONLY:

Here is the price data for {ticker} over the last {days_window} days:

| Date | Close Price |
| :--- | :--- |

(Populate {days_window} rows using RECENT TRADING HISTORY — newest last.)

Rules:
- First line must match exactly: Here is the price data for {ticker} over the last {days_window} days:
- Use the pipe-table header shown above with | :--- | :--- | alignment row.
- Pull Date and Close Price ONLY from RECENT TRADING HISTORY.
- No paragraphs, no buy/sell advice, no extra sections.
"""

    if intent == "investment":
        return base + """
MANDATORY RESPONSE FORMAT — INVESTMENT REASONING (NO EXCEPTIONS):

Your entire reply MUST contain ONLY these three sections:

**🟢 Why You Should Buy:**
- Point 1 (one sentence, data-anchored)
- Point 2 (one sentence, data-anchored)
- Point 3 (one sentence, data-anchored)

**🔴 Why You Should Avoid:**
- Point 1 (one sentence, risk-anchored)
- Point 2 (one sentence, risk-anchored)
- Point 3 (one sentence, risk-anchored)

**💡 Final Recommendation Summary:**
Exactly two sentences of tactical takeaway for the investor/brand manager.

Rules:
- Exactly 3 bullets under each of the first two headers.
- Exactly 2 sentences under the final header.
- No plain-paragraph dumps, no extra headers, no disclaimers.
"""

    return base + """
GENERAL RESPONSE RULES:
- Answer in ≤200 words, grounded in the supplied data.
- If the question is about past prices → use the historical table format.
- If the question is about buy/sell rationale → use the three-section investment matrix.
- Never output unstructured essay text when a structured format applies.
"""


# ── endpoint ──────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = predict_next_day_price(request.ticker)

    current     = result["current_price"]
    predicted   = result["predicted_price"]
    sentiment   = result["sentiment_score"]
    history     = result.get("historical_context_string", "") or "No historical data available"
    rows        = result.get("historical_rows", [])
    data_source = result.get("data_source", "live")
    err         = result["error"]

    pct_change = None
    direction  = "NEUTRAL"

    if current and predicted:
        pct_change = round((predicted - current) / current * 100, 2)
        direction  = "BULLISH 📈" if predicted > current else "BEARISH 📉"

    user_message = request.user_query.strip()
    intent       = _detect_query_intent(user_message)
    days_window  = _extract_history_days(user_message)

    system_prompt = _build_system_prompt(
        ticker=result["ticker"],
        current=current,
        predicted=predicted,
        pct_change=pct_change,
        direction=direction,
        sentiment=sentiment,
        history=history,
        data_source=data_source,
        user_query=user_message,
    )
    full_prompt = f"{system_prompt}\n\nUser Question: {user_message}"

    llm_text = "LLM unavailable – ensure Ollama is running with `ollama serve`."
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            ollama_resp = await client.post(
                OLLAMA_URL,
                json={
                    "model":  OLLAMA_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                },
            )
            ollama_resp.raise_for_status()
            payload  = ollama_resp.json()
            llm_text = payload.get("response", "").strip()
    except httpx.ConnectError:
        llm_text = (
            "⚠️ Cannot reach Ollama. Start it with: `ollama serve` "
            "and pull the model with `ollama pull gemma3:1b`."
        )
    except Exception as exc:
        llm_text = f"⚠️ LLM error: {exc}"

    llm_text = _enforce_structured_response(
        intent=intent,
        llm_text=llm_text,
        ticker=result["ticker"],
        days=days_window,
        rows=rows,
        current=current,
        predicted=predicted,
        pct_change=pct_change,
        direction=direction,
        sentiment=sentiment or 0.0,
        history=history,
    )

    return ChatResponse(
        ticker=result["ticker"],
        current_price=current,
        predicted_price=predicted,
        pct_change=pct_change,
        direction=direction,
        sentiment_score=sentiment,
        historical_context_string=history,
        data_source=data_source,
        llm_response=llm_text,
        error=err,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
