
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import json

# =====================================================
# 1. APPLICATION CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="StockIQ | AI Stock Analyst",
    page_icon="📈",
    layout="wide"
)

GEMINI_MODEL = "gemini-3.1-flash-lite"

stocks = {
    "Reliance Industries": {
        "symbol": "RELIANCE.NS",
        "news_query": '"Reliance Industries"'
    },
    "TCS": {
        "symbol": "TCS.NS",
        "news_query": '"Tata Consultancy Services"'
    },
    "Infosys": {
        "symbol": "INFY.NS",
        "news_query": '"Infosys"'
    },
    "HDFC Bank": {
        "symbol": "HDFCBANK.NS",
        "news_query": '"HDFC Bank"'
    },
    "ICICI Bank": {
        "symbol": "ICICIBANK.NS",
        "news_query": '"ICICI Bank"'
    }
}

# =====================================================
# 2. API KEY CONFIGURATION
# =====================================================

def get_secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
NEWS_API_KEY = get_secret("NEWS_API_KEY")


# =====================================================
# 3. STOCK DATA RETRIEVAL
# =====================================================

@st.cache_data(ttl=300)
def fetch_stock_data(symbol, period):

    data = yf.Ticker(symbol).history(
        period=period,
        interval="1d",
        auto_adjust=True
    )

    if data.empty:
        raise ValueError("No stock data received.")

    data = data.copy()

    data = data.dropna(subset=["Close"])

    # Moving averages
    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()

    data["EMA_20"] = data["Close"].ewm(
        span=20,
        adjust=False
    ).mean()

    # RSI calculation
    delta = data["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (100 / (1 + rs))

    data.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "RSI"
    ] = 100

    data.loc[
        (avg_loss == 0) & (avg_gain == 0),
        "RSI"
    ] = 50

    # MACD calculation
    ema12 = data["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = data["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = ema12 - ema26

    data["Signal"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    data["MACD_Histogram"] = (
        data["MACD"] - data["Signal"]
    )

    return data


# =====================================================
# 4. NEWS RETRIEVAL
# =====================================================

@st.cache_data(ttl=900)
def fetch_company_news(company, api_key):

    if not api_key:
        raise ValueError("NEWS_API_KEY is missing.")

    news_url = "https://newsapi.org/v2/everything"

    params = {
        "q": company,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": api_key
    }

    response = requests.get(
        news_url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    if result.get("status") != "ok":
        raise ValueError(
            result.get("message", "News API error")
        )

    articles = []

    for article in result.get("articles", []):

        articles.append({
            "headline": article.get("title") or "No title",
            "description": article.get("description") or "",
            "source": article.get("source", {}).get("name", "Unknown"),
            "published": article.get("publishedAt", ""),
            "url": article.get("url", "")
        })

    return articles


# =====================================================
# 5. GEMINI AI INTEGRATION
# =====================================================

def ask_gemini(prompt):

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing.")

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{GEMINI_MODEL}:generateContent"
    )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1800
        }
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    candidates = result.get("candidates", [])

    if not candidates:
        raise ValueError(
            "Gemini did not return a response. "
            "Check API limits or safety restrictions."
        )

    parts = candidates[0].get(
        "content", {}
    ).get("parts", [])

    answer = "\n".join(
        part["text"]
        for part in parts
        if "text" in part
    )

    if not answer:
        raise ValueError("Gemini returned an empty response.")

    return answer


# =====================================================
# 6. APPLICATION HEADER
# =====================================================

st.title("📈 StockIQ")

st.subheader(
    "AI-Powered Stock Market Intelligence"
)

st.write(
    "Research stocks using technical indicators, "
    "financial news and artificial intelligence."
)

st.divider()


# =====================================================
# 7. STOCK SELECTION
# =====================================================

col1, col2 = st.columns(2)

with col1:

    selected_stock = st.selectbox(
        "Select Company",
        list(stocks.keys())
    )

with col2:

    period = st.selectbox(
        "Historical Data Period",
        ["3mo", "6mo", "1y", "2y", "5y"],
        index=2
    )

symbol = stocks[selected_stock]["symbol"]

if st.button("🔍 Analyse Stock", type="primary"):

    st.session_state["stock_analysis_requested"] = True

    st.session_state.pop("sentiment_report", None)
    st.session_state.pop("ai_stock_report", None)

# =====================================================
# 8. LOAD STOCK DATA
# =====================================================

if st.session_state.get("stock_analysis_requested", False):

    try:

        with st.spinner("Retrieving stock information..."):

            data = fetch_stock_data(symbol, period)

        st.session_state["stock_data"] = data
        st.session_state["current_symbol"] = symbol

        latest = data.iloc[-1]

        st.success(
            f"Stock analysis loaded: {selected_stock}"
        )

        # =================================================
        # 9. FINANCIAL METRICS
        # =================================================

        st.header("📊 Market Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Latest Closing Price",
            f"₹{latest['Close']:,.2f}"
        )

        c2.metric(
            "RSI",
            f"{latest['RSI']:.2f}"
        )

        c3.metric(
            "MACD",
            f"{latest['MACD']:.2f}"
        )

        c4.metric(
            "Trading Volume",
            f"{latest['Volume']:,.0f}"
        )

        # =================================================
        # 10. PRICE CHART
        # =================================================

        st.header("📈 Historical Price Analysis")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["Close"],
            name="Closing Price",
            line=dict(width=2)
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["SMA_20"],
            name="SMA 20"
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["SMA_50"],
            name="SMA 50"
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["EMA_20"],
            name="EMA 20"
        ))

        fig.update_layout(
            height=500,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Price (INR)",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =================================================
        # 11. RSI CHART
        # =================================================

        st.header("Relative Strength Index")

        rsi_fig = go.Figure()

        rsi_fig.add_trace(go.Scatter(
            x=data.index,
            y=data["RSI"],
            name="RSI"
        ))

        rsi_fig.add_hline(
            y=70,
            line_dash="dash",
            annotation_text="Overbought"
        )

        rsi_fig.add_hline(
            y=30,
            line_dash="dash",
            annotation_text="Oversold"
        )

        rsi_fig.update_layout(
            height=350,
            template="plotly_white",
            yaxis_range=[0, 100]
        )

        st.plotly_chart(
            rsi_fig,
            use_container_width=True
        )

        # =================================================
        # 12. MACD CHART
        # =================================================

        st.header("MACD Analysis")

        macd_fig = go.Figure()

        macd_fig.add_trace(go.Scatter(
            x=data.index,
            y=data["MACD"],
            name="MACD"
        ))

        macd_fig.add_trace(go.Scatter(
            x=data.index,
            y=data["Signal"],
            name="Signal"
        ))

        macd_fig.add_trace(go.Bar(
            x=data.index,
            y=data["MACD_Histogram"],
            name="Histogram"
        ))

        macd_fig.update_layout(
            height=350,
            template="plotly_white"
        )

        st.plotly_chart(
            macd_fig,
            use_container_width=True
        )

        # =================================================
        # 13. RECENT TRADING DATA
        # =================================================

        st.header("Recent Trading Data")

        st.dataframe(
            data.tail(10),
            use_container_width=True
        )

        # =================================================
        # 14. FINANCIAL NEWS INTELLIGENCE
        # =================================================

        st.divider()

        st.header("📰 Financial News Intelligence")

        news_company = stocks[selected_stock]["news_query"]

        if st.button("Fetch Latest Company News"):

            try:

                with st.spinner("Fetching company news..."):

                    articles = fetch_company_news(
                        news_company,
                        NEWS_API_KEY
                    )

                st.session_state["company_news"] = articles
                st.session_state["news_symbol"] = symbol

                st.session_state.pop("sentiment_report", None)
                st.session_state.pop("ai_stock_report", None)

            except Exception as e:

                st.error(f"News retrieval failed: {e}")

        articles = st.session_state.get("company_news", [])

        if st.session_state.get("news_symbol") == symbol:

            if articles:

                st.success(
                    f"{len(articles)} news articles retrieved."
                )

                for article in articles:

                    st.subheader(article["headline"])

                    st.caption(
                        f"{article['source']} | "
                        f"{article['published']}"
                    )

                    st.write(article["description"])

                    if article["url"]:
                        st.markdown(
                            f"[Read full article]({article['url']})"
                        )

                    st.divider()

            else:

                st.info("Click Fetch Latest Company News to retrieve articles.")

        # =================================================
        # 15. GEMINI AI NEWS SENTIMENT
        # =================================================

        st.header("🤖 AI News Sentiment Analyst")

        if (
            articles
            and st.session_state.get("news_symbol") == symbol
        ):

            if st.button("Analyse News Sentiment"):

                try:

                    prompt = f"""
                    You are StockIQ, a financial research analyst.

                    Company: {selected_stock}
                    Stock symbol: {symbol}

                    Analyse these news articles:

                    {json.dumps(articles, indent=2)}

                    For EACH article provide:

                    1. Headline
                    2. Sentiment: Bullish, Bearish or Neutral
                    3. Reason for classification
                    4. Potential business impact

                    Then provide:

                    5. Overall news sentiment
                    6. Major company risks
                    7. Possible short-term implications
                    8. Conflicting or duplicate news
                    9. Uncertainty and limitations

                    IMPORTANT:
                    - Use only supplied news.
                    - Do not invent facts.
                    - News sentiment is not a guaranteed stock prediction.
                    - Clearly distinguish facts from interpretations.
                    """

                    with st.spinner(
                        "Gemini is analysing financial news..."
                    ):

                        report = ask_gemini(prompt)

                    st.session_state["sentiment_report"] = report

                except Exception as e:

                    st.error(
                        f"AI sentiment analysis failed: {e}"
                    )

            if st.session_state.get("sentiment_report"):

                st.subheader("News Sentiment Report")

                st.markdown(
                    st.session_state["sentiment_report"]
                )

        else:

            st.info(
                "Fetch company news first to enable sentiment analysis."
            )

        # =================================================
        # 16. GEMINI AI STOCK ANALYSIS
        # =================================================

        st.divider()

        st.header("🧠 StockIQ AI Research Analyst")

        if st.button("Generate Complete AI Stock Analysis"):

            try:

                latest = data.iloc[-1]

                stock_info = {
                    "company": selected_stock,
                    "symbol": symbol,
                    "latest_close": round(float(latest["Close"]), 2),
                    "RSI": round(float(latest["RSI"]), 2),
                    "MACD": round(float(latest["MACD"]), 4),
                    "Signal": round(float(latest["Signal"]), 4),
                    "SMA_20": round(float(latest["SMA_20"]), 2),
                    "SMA_50": round(float(latest["SMA_50"]), 2),
                    "EMA_20": round(float(latest["EMA_20"]), 2),
                    "Volume": int(latest["Volume"])
                }

                current_news = (
                    articles
                    if st.session_state.get("news_symbol") == symbol
                    else []
                )

                prompt = f"""
                You are StockIQ, an AI financial research analyst.

                COMPANY:
                {selected_stock}

                TECHNICAL DATA:
                {json.dumps(stock_info, indent=2)}

                RECENT NEWS:
                {json.dumps(current_news, indent=2)}

                Prepare a structured research report.

                SECTION 1: Overall Market Trend

                SECTION 2: Technical Indicator Analysis
                Explain RSI, MACD, SMA and EMA.

                SECTION 3: News Sentiment
                Discuss positive, negative and neutral news.
                If no news is supplied, clearly state this.

                SECTION 4: Bullish Scenario
                Explain conditions that could support an upward move.

                SECTION 5: Bearish Scenario
                Explain conditions that could support a downward move.

                SECTION 6: Neutral Scenario
                Explain possible sideways market conditions.

                SECTION 7: Major Risks

                SECTION 8: Uncertainty and Limitations

                RULES:
                - Use only supplied data.
                - Never invent prices or indicators.
                - Do not fabricate probabilities.
                - Do not guarantee future returns.
                - Separate observed facts from interpretations.
                - Explain conflicting technical indicators.
                - This is research, not personalized investment advice.
                """

                with st.spinner(
                    "StockIQ AI is preparing your research report..."
                ):

                    report = ask_gemini(prompt)

                st.session_state["ai_stock_report"] = report

            except Exception as e:

                st.error(f"AI stock analysis failed: {e}")

        if st.session_state.get("ai_stock_report"):

            st.subheader("Complete AI Stock Research Report")

            st.markdown(
                st.session_state["ai_stock_report"]
            )

        # =================================================
        # 17. DISCLAIMER
        # =================================================

        st.divider()

        st.caption(
            "StockIQ is an educational financial research application. "
            "Market data may be delayed or incomplete. "
            "AI-generated analysis is not guaranteed to be accurate "
            "and should not be treated as investment advice."
        )

    except Exception as e:

        st.error(f"Stock analysis failed: {e}")

else:

    st.info(
        "Select a company and historical period, "
        "then click Analyse Stock to begin."
    )

# =====================================================
# 18. FOOTER
# =====================================================

st.divider()

st.caption(
    "StockIQ | AI-Powered Stock Market Intelligence | "
    "Educational Research Platform"
)
