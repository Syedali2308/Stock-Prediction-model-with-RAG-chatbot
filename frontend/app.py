"""
app.py  –  Streamlit dashboard: live forecasting + RAG investment chatbot.
Run with:  cd frontend && streamlit run app.py
"""

import os

import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock RAG Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

NSE_TICKERS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "WIPRO", "AXISBANK", "SBIN", "BAJFINANCE", "MARUTI",
    "TATAMOTORS", "TATASTEEL", "HCLTECH", "LTIM", "SUNPHARMA",
    "ADANIENT", "ADANIPORTS", "ONGC", "COALINDIA", "NTPC",
    "POWERGRID", "ULTRACEMCO", "JSWSTEEL", "TITAN", "NESTLEIND",
    "ASIANPAINT", "HINDUNILVR", "BRITANNIA", "DABUR", "MARICO",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Custom CSS  –  Minimalist HFT Terminal (Red / Black / White)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;600;700&display=swap');

:root {
    --bg:       #050505;
    --panel:    #0a0a0a;
    --border:   #E60026;
    --accent:   #E60026;
    --white:    #FFFFFF;
    --muted:    #888888;
    --green:    #00FF66;
    --red:      #FF3344;
    --card-bg:  #0d0d0d;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--white) !important;
    font-family: 'Inter', sans-serif !important;
}

.stApp { background: var(--bg) !important; }

[data-testid="stSidebar"] {
    background: var(--bg) !important;
    border-right: none !important;
    width: 0 !important;
    min-width: 0 !important;
}
[data-testid="stSidebar"] > div { display: none !important; }

.stSelectbox div[data-baseweb="select"] > div,
.stTextInput input, .stTextArea textarea {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--white) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 2px !important;
}

.stButton > button {
    background: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    color: var(--white) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    border-radius: 2px !important;
    text-transform: uppercase !important;
    transition: all .15s ease !important;
}
.stButton > button:hover {
    background: var(--white) !important;
    color: var(--bg) !important;
    border-color: var(--white) !important;
}

.terminal-header {
    border-bottom: 2px solid var(--accent);
    padding-bottom: 12px;
    margin-bottom: 8px;
}
.terminal-title {
    font-family: 'Inter', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: var(--white);
    letter-spacing: 3px;
    margin: 0;
}
.terminal-sub {
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 4px;
}

.section-heading {
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: var(--white);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0 0 6px 0;
}
.section-caption {
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 1px;
    margin: 0 0 12px 0;
}

.selector-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

.combobox-wrap {
    border: 1px solid #333;
    border-radius: 2px;
    padding: 10px 12px;
    background: #080808;
}

.metric-card {
    background: var(--card-bg);
    border: 1px solid #222;
    border-top: 3px solid var(--accent);
    border-radius: 2px;
    padding: 18px 16px;
    text-align: center;
    min-height: 100px;
}
.metric-label {
    font-size: 10px;
    letter-spacing: 2.5px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 24px;
    font-weight: 600;
}
.metric-value.up   { color: var(--green) !important; }
.metric-value.down { color: var(--red)   !important; }
.metric-value.neutral { color: var(--white) !important; }

.history-strip {
    background: var(--panel);
    border-left: 3px solid var(--accent);
    padding: 12px 16px;
    margin: 12px 0 16px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    line-height: 1.7;
}
.history-strip b { color: var(--white); letter-spacing: 1px; }

.chat-user {
    background: #111;
    border-left: 3px solid var(--accent);
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 0 4px 4px 0;
    font-size: 14px;
}
.chat-bot {
    background: #0a0a0a;
    border-left: 3px solid var(--white);
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 0 4px 4px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    line-height: 1.65;
    color: #FFFFFF !important;
}
.chat-label-user { color: var(--accent); font-weight: 700; font-size: 10px; letter-spacing: 2px; }
.chat-label-bot  { color: var(--white); font-weight: 700; font-size: 10px; letter-spacing: 2px; }

/* RAG chat — force bright white markdown on dark theme */
.rag-chat-block,
.rag-chat-block [data-testid="stMarkdownContainer"],
.rag-chat-block [data-testid="stMarkdownContainer"] p,
.rag-chat-block [data-testid="stMarkdownContainer"] li,
.rag-chat-block [data-testid="stMarkdownContainer"] ul,
.rag-chat-block [data-testid="stMarkdownContainer"] ol,
.rag-chat-block [data-testid="stMarkdownContainer"] td,
.rag-chat-block [data-testid="stMarkdownContainer"] th,
.rag-chat-block [data-testid="stMarkdownContainer"] strong,
.rag-chat-block [data-testid="stMarkdownContainer"] em,
.rag-chat-block [data-testid="stMarkdownContainer"] span,
.rag-chat-block [data-testid="stMarkdownContainer"] h1,
.rag-chat-block [data-testid="stMarkdownContainer"] h2,
.rag-chat-block [data-testid="stMarkdownContainer"] h3,
.rag-chat-block [data-testid="stMarkdownContainer"] h4,
.chat-response-block,
.chat-response-block p,
.chat-response-block li,
.chat-response-block td,
.chat-response-block th,
.chat-response-block strong,
.chat-response-block code,
.chat-response-block pre,
.chat-response-block span {
    color: #FFFFFF !important;
}

.chat-response-block > div[data-testid="stMarkdownContainer"],
.chat-response-block > div[data-testid="stMarkdownContainer"] * {
    color: #FFFFFF !important;
}

.chat-response-block > div[data-testid="stMarkdownContainer"] table,
.chat-response-block > div[data-testid="stMarkdownContainer"] th,
.chat-response-block > div[data-testid="stMarkdownContainer"] td {
    color: #FFFFFF !important;
}

.chat-input-area {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 12px;
    margin-top: 12px;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: var(--white) !important;
    letter-spacing: 2px !important;
}

hr { border-color: #222 !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
.stAlert { background: var(--panel) !important; border: 1px solid #333 !important; }

#MainMenu, footer, header { visibility: hidden; }

/* ─────────────────────────────────────────────────────────────────────
   GLOBAL TEXT-VISIBILITY FIX
   Forces bright white (#FFFFFF) on all Streamlit-rendered markdown /
   output text app-wide for contrast against the dark theme. Explicit
   semantic colors (up/down signal, sentiment bar) are excluded so
   they keep conveying meaning; everything else defaults to white.
   ───────────────────────────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] ol,
[data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em,
[data-testid="stMarkdownContainer"] code,
[data-testid="stMarkdownContainer"] span:not(.chat-label-user),
[data-testid="stMarkdownContainer"] div:not(.metric-value):not(.metric-label):not(.history-strip):not(.chat-label-user),
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] *,
[data-testid="stText"],
[data-testid="stAlertContentInfo"],
[data-testid="stAlertContentWarning"],
[data-testid="stAlertContentError"],
[data-testid="stAlertContentSuccess"],
.stAlert p, .stAlert div, .stAlert span,
.stMarkdown p, .stMarkdown li, .stMarkdown span {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)


def resolve_ticker_combobox(typed: str) -> str:
    """
    Single combobox logic:
    - Empty input  → searchable selectbox over full NSE list
    - Typed text   → filter list; if no list match, treat typed value as custom ticker
    """
    typed = typed.strip().upper()

    if not typed:
        return st.selectbox(
            "Ticker options",
            NSE_TICKERS,
            index=0,
            label_visibility="collapsed",
            key="ticker_combobox_select",
        )

    matches = [t for t in NSE_TICKERS if typed in t]
    if len(matches) > 1:
        return st.selectbox(
            "Ticker options",
            matches,
            index=0,
            label_visibility="collapsed",
            key="ticker_combobox_filtered",
        )
    if len(matches) == 1:
        st.caption(f"Matched: **{matches[0]}**")
        return matches[0]

    st.caption(f"Custom ticker: **{typed}**")
    return typed


# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='terminal-header'>"
    "<p class='terminal-title'>STOCK RAG TERMINAL</p>"
    "<p class='terminal-sub'>Live Forecast · Historical RAG · NSE</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Ticker combobox + actions
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<p class='section-heading'>TICKER SELECTION</p>"
    "<p class='section-caption'>Type to filter popular NSE symbols, pick from matches, "
    "or enter any custom ticker code in the same field.</p>",
    unsafe_allow_html=True,
)
st.markdown("<div class='selector-panel'>", unsafe_allow_html=True)

row1, row2, row3 = st.columns([4, 1.2, 1])

with row1:
    st.markdown("<div class='combobox-wrap'>", unsafe_allow_html=True)
    typed_ticker = st.text_input(
        "Ticker combobox",
        placeholder="Search or type ticker — e.g. RELIANCE, TCS, PAYTM",
        label_visibility="collapsed",
        key="ticker_combobox_input",
    )
    active_ticker = resolve_ticker_combobox(typed_ticker)
    st.markdown("</div>", unsafe_allow_html=True)

with row2:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    forecast_btn = st.button("RUN FORECAST", use_container_width=True)

with row3:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    clear_btn = st.button("CLEAR", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

if clear_btn:
    st.session_state.chat_history = []
    st.session_state.last_result = None
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  API helper
# ─────────────────────────────────────────────────────────────────────────────
def call_chat_api(ticker: str, query: str) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}/api/chat",
            json={"ticker": ticker, "user_query": query},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot reach FastAPI backend at {API_BASE}. Make sure the backend is running and the API_BASE setting points to the correct host."
        )
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


if forecast_btn:
    with st.spinner(f"Running inference for **{active_ticker}**…"):
        result = call_chat_api(
            active_ticker,
            f"Give me a professional overview of {active_ticker} based on today's forecast.",
        )
    if result:
        st.session_state.last_result = result
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result.get("llm_response", ""),
            "ticker": result.get("ticker", active_ticker),
        })


# ─────────────────────────────────────────────────────────────────────────────
#  Forecast output
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### **LIVE FORECAST OUTPUT**")

res = st.session_state.last_result

if res:
    current    = res.get("current_price")
    predicted  = res.get("predicted_price")
    pct        = res.get("pct_change")
    direction  = res.get("direction", "NEUTRAL")
    sentiment  = res.get("sentiment_score", 0.0)
    history    = res.get("historical_context_string", "")
    data_source  = res.get("data_source", "live")
    err          = res.get("error")
    ticker_lbl   = res.get("ticker", active_ticker)

    if err:
        st.warning(err)
    elif data_source == "synthetic":
        st.info("Simulated market data active (yfinance blocked). LSTM + VADER pipeline still operational.")

    perf_cls = "up" if (pct is not None and pct >= 0) else "down"

    def card(col, label, value, force_neutral=False):
        cls = "neutral" if force_neutral else perf_cls
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value {cls}'>{value}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### **KEY METRICS**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        card(c1, "TICKER", ticker_lbl, force_neutral=True)
    with c2:
        card(c2, "TODAY CLOSE", f"₹{current}" if current else "—")
    with c3:
        card(c3, "TOMORROW PRED", f"₹{predicted}" if predicted else "—")
    pct_str = (f"{'+' if pct >= 0 else ''}{pct}%" if pct is not None else "—")
    with c4:
        card(c4, "EXP % CHANGE", pct_str)
    with c5:
        card(c5, "SIGNAL", direction)

    if history:
        st.markdown("#### **RECENT TRADING HISTORY (LAST 5 DAYS)**")
        st.markdown(
            f"<div class='history-strip'>{history}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### **NEWS SENTIMENT (VADER FUSION)**")
    sent_pct = int((sentiment + 1) / 2 * 100)
    sent_col = "#00FF66" if sentiment >= 0 else "#FF3344"
    st.markdown(
        f"<div style='margin-top:4px;'>"
        f"<b style='color:{sent_col};font-family:IBM Plex Mono,monospace;'>{sentiment:+.4f}</b>"
        f"<div style='height:4px;background:#111;border-radius:2px;margin-top:6px;max-width:400px;'>"
        f"<div style='width:{sent_pct}%;height:4px;background:{sent_col};border-radius:2px;'></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<p style='color:#888;font-size:13px;'>"
        "<b>Select a ticker above and click RUN FORECAST to begin.</b></p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  Chat
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### **INVESTMENT CHAT (RAG + GEMMA 3)**")
st.markdown(
    "<p class='section-caption'><b>Historical price queries</b> (e.g. "
    "<i>pichle 3 din ka price</i>) and <b>investment rationale</b> (e.g. "
    "<i>kyu khareedna chahiye</i>) use structured response formats.</p>",
    unsafe_allow_html=True,
)

st.markdown('<div class="rag-chat-block">', unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='chat-user'>"
            f"<div class='chat-label-user'>YOU</div>{msg['content']}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        label = msg.get("ticker", active_ticker)
        st.markdown(
            f"<div class='chat-label-bot'>GEMMA 3 · {label}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="chat-response-block">', unsafe_allow_html=True)
        st.markdown(msg["content"])
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='chat-input-area'>", unsafe_allow_html=True)
st.markdown("**ASK A QUESTION**")
col_inp, col_send = st.columns([5, 1])
with col_inp:
    user_input = st.text_input(
        "Query",
        placeholder=f"e.g. {active_ticker} pichle 3 din ka price / kyu khareedna chahiye?",
        label_visibility="collapsed",
        key="chat_input",
    )
with col_send:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    send_btn = st.button("SEND", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if send_btn and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Analysing…"):
        result = call_chat_api(active_ticker, user_input)
    if result:
        st.session_state.last_result = result
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result.get("llm_response", "No response."),
            "ticker": result.get("ticker", active_ticker),
        })
    st.rerun()