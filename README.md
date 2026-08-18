# 📊 Stock RAG Dashboard

**Live Multi-Stock Price Forecasting & RAG Investment Chatbot**

A hybrid, sentiment-aware stock forecasting system for NSE-listed equities that fuses a **Bidirectional LSTM** time-series model with **real-time news sentiment (VADER)**, and wraps the forecast in a **locally-hosted RAG investment chatbot** (Gemma 3 via Ollama) — so predictions come with grounded, explainable reasoning instead of just a number.

---

## 🎥 Demo

<!--
Add your demo video/GIF here. A few common options:

1. Upload a GIF/short clip directly to the repo and embed it:
   ![Demo](assets/demo.gif)

2. Link to a hosted video (YouTube/Drive/Loom) with a thumbnail:
   [![Watch the demo](assets/thumbnail.png)](https://your-video-link-here)

3. If hosting on GitHub, drag-and-drop the video file directly into this
   README while editing on github.com — GitHub will auto-generate the
   embed markup for you.
-->



https://github.com/user-attachments/assets/c07cd6dd-12c8-4453-b21c-b02480a74eab



*A quick walkthrough of the dashboard: entering a ticker, viewing the live forecast + sentiment signal, and chatting with the RAG investment assistant in both English and Hinglish.*

---

## 📖 About the Project

Classical stock forecasting approaches — moving averages, ARIMA, or a plain price-only LSTM — learn exclusively from historical price sequences. They stay blind to the sentiment-driven shifts that news and headlines routinely trigger in real markets, which means their forecasts often lag exactly when it matters most.

**Stock RAG Dashboard** addresses this by fusing two independent signals into one model:

- **Price action** — Close price plus 7-day and 21-day rolling moving averages, capturing short- and medium-term trend.
- **News sentiment** — a live VADER compound sentiment score computed from the stock's most recent headlines.

These four features are fed into a **Bidirectional LSTM** trained to predict the next trading day's closing price. On top of that forecast sits a **Retrieval-Augmented Generation (RAG) chatbot**: instead of retrieving from a vector database, it assembles the latest live numbers (current price, predicted price, % change, sentiment, recent history) directly into a structured prompt for a locally-run **Gemma 3:1b** model via **Ollama**. A deterministic format-enforcement layer guarantees the chatbot's answer is always correctly structured — a historical price table or a Buy/Avoid/Summary investment matrix — even if the small local LLM ignores instructions.

The result is a fully local, zero-cloud-cost pipeline: no paid market-data API, no paid sentiment API, and no paid LLM API — everything from data acquisition to the final chat response runs on free tools and a locally-hosted model.

### ✨ Key Features

- **Multi-stock, on-demand forecasting** — enter any NSE ticker and get a live next-day price prediction.
- **Sentiment-augmented model** — a Bidirectional LSTM trained on price *and* real-time news sentiment, not price alone.
- **Triple-redundant data pipeline** — `yfinance` → `Ticker().history()` → Stooq fallback, so a single data-source outage never breaks a forecast.
- **Local RAG investment chatbot** — ask natural-language questions ("should I buy this?", "pichle 3 din ka price?") and get grounded, data-backed answers with zero hallucinated facts.
- **Bilingual query understanding** — the chatbot's intent detection recognizes both English and Hinglish phrasing.
- **Self-correcting output formatting** — a Python fallback guarantees a well-formatted response even if the LLM drifts off-format.
- **Fully local LLM runtime** — powered by Ollama + Gemma 3:1b, with no API keys and no per-query cost.
- **Interactive Streamlit dashboard** — live KPI cards (current price, predicted price, % change, directional signal) plus an embedded chat UI.

### 🏗️ How It Works (High Level)

```
User (Streamlit) → FastAPI /api/chat → engine.py
                                          ├─ fetch OHLCV data (triple fallback)
                                          ├─ engineer features (Close, MA-7, MA-21)
                                          ├─ score live news sentiment (VADER)
                                          ├─ scale + window (RobustScaler, 60-day lookback)
                                          ├─ run Bidirectional LSTM inference
                                          └─ inverse-transform → predicted price
                        → build structured prompt with live numbers
                        → Ollama (Gemma 3:1b) generates a response
                        → format-enforcement verifies/repairs the response
                        → JSON returned to Streamlit and rendered
```

---

# 🗂 Project Structure

```
Stock_RAG_Dashboard/
├── backend/
│   ├── main.py                       ← FastAPI server
│   ├── engine.py                     ← ML inference core
│   ├── fix_model.py                  ← regenerate Keras model
│   └── universal_stock_model.keras   ← trained LSTM model
├── frontend/
│   └── app.py                        ← Streamlit dashboard
├── README.md
├── requirements.txt
└── .gitignore
```

| Component | File | Responsibility |
|---|---|---|
| Presentation | `frontend/app.py` (Streamlit) | Ticker input, KPI cards, historical chart, chatbot UI |
| API | `backend/main.py` (FastAPI) | `/api/chat` endpoint, prompt construction, Ollama invocation, response-format enforcement |
| Data & Inference | `backend/engine.py` | Market-data download, feature engineering, LSTM inference, VADER sentiment scoring |
| Model | `backend/universal_stock_model.keras` | Pre-trained Bidirectional LSTM used for next-day price prediction |
| LLM Runtime | Ollama + Gemma 3:1b | Local, zero-cost text generation for the RAG chatbot |

---

## ⚙️ Setup

### 1. Create & activate a virtual environment (project root)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Model file
Ensure `backend/universal_stock_model.keras` exists. Regenerate if needed:
```bash
cd backend
python fix_model.py
```

### 4. Pull & run Ollama with Gemma 3
```bash
# Install Ollama from https://ollama.com then:
ollama pull gemma3:1b
ollama serve          # keeps running in the background
```

---

## 🚀 Running the App

**Terminal 1 – FastAPI backend**
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

**Terminal 2 – Streamlit frontend**
```bash
cd frontend
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

Optional: override the API URL from the frontend with `API_BASE=http://127.0.0.1:8000`.

---

## 🔌 API Reference

| Method | Endpoint     | Payload                              | Description          |
|--------|--------------|--------------------------------------|----------------------|
| POST   | /api/chat    | `{"ticker":"RELIANCE","user_query":"…"}` | RAG chat + forecast |
| GET    | /health      | —                                    | Health check         |

**Example request:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"ticker": "HDFCBANK", "user_query": "should I buy this stock?"}'
```

**Example response (truncated):**
```json
{
  "ticker": "HDFCBANK.NS",
  "current_price": 1544.64,
  "predicted_price": 1596.63,
  "pct_change": 3.37,
  "direction": "BULLISH 📈",
  "sentiment_score": 0.21,
  "data_source": "live",
  "llm_response": "🟢 Why You Should Buy: ...",
  "error": null
}
```

---

## 🧠 Model Details

- **Architecture**: Bidirectional LSTM(32, return_sequences=True) → Dropout(0.1) → LSTM(16) → Dense(1)
- **Input**: 60-day lookback window across 4 features — `[Close, MA-7, MA-21, VADER Sentiment]`, shaped as `(1, 60, 4)`
- **Scaling**: `RobustScaler` (median/IQR-based), chosen for resilience to outlier price days
- **Loss / Optimiser**: Huber loss (outlier-robust) with the Adam optimiser
- **Output**: Scaled next-day (T+1) closing price, inverse-transformed back into ₹

## 📈 Evaluation Snapshot

Evaluated on five NSE large-caps (RELIANCE, HDFCBANK, TCS, INFY, WIPRO) using historical data with a held-out test split:

- **~24.9%** average RMSE reduction versus a price-only baseline LSTM
- **60.9%–64.1%** directional accuracy across all five stocks
- **100%** factual accuracy on historical price queries in the RAG chatbot (deterministic table formatting)
- **0%** hallucinated news events or fabricated narratives detected in chatbot evaluation

---

## 📦 Tech Stack

- **FastAPI** – async REST backend, handles the non-blocking call to the local LLM
- **TensorFlow / Keras** – Bidirectional LSTM model for price forecasting
- **yfinance** – live NSE market data, with `Ticker().history()` and Stooq as sequential fallbacks
- **VADER (NLTK)** – zero-key, real-time news sentiment scoring
- **Ollama + Gemma 3:1b** – local LLM runtime powering the RAG investment chatbot
- **Streamlit** – interactive, single-file dashboard for live ticker analysis

---

## ⚠️ Limitations

- Forecasts only the next trading day (T+1) — no multi-day horizon yet
- VADER is a general-purpose sentiment lexicon, not finance-tuned
- News source is limited to `yfinance`'s bundled headlines
- Backtested performance does not account for transaction costs or slippage

## 🔭 Future Scope

- Multi-day (5-day, 10-day) forecasting horizons
- Multi-publisher news aggregation for richer sentiment
- Optional upgrade to a finance-tuned sentiment model (e.g. FinBERT)
- Containerised cloud deployment for always-on access
- Portfolio-level view aggregating signals across a full watchlist

---

## 📄 License

Add your license here (e.g. MIT).
