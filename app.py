
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import json
from sklearn.ensemble import RandomForestRegressor

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
# 1A. LOAD NSE EQUITY STOCKS
# =====================================================

def load_nse_stocks():

    try:
        df = pd.read_csv(
            "NSE_CM_security_20112026.csv.gz",
            dtype=str,
            encoding="utf-8-sig"
        )

        required_columns = [
            "TckrSymb",
            "SctySrs",
            "FinInstrmNm",
            "ISIN",
            "DelFlg"
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns in NSE file: {missing}"
            )

        # Keep ordinary equity and trade-to-trade equity
        df["SctySrs"] = df["SctySrs"].str.strip().str.upper()

        df = df[
            df["SctySrs"].isin(["EQ", "BE"])
        ].copy()

        # Exclude deleted securities
        df = df[
            df["DelFlg"].str.strip().str.upper() != "Y"
        ]

        # Exclude test and invalid records
        df = df[
            df["TckrSymb"].str.strip().ne("")
            & ~df["TckrSymb"].str.contains(
                "TEST", case=False, na=False
            )
            & df["ISIN"].str.match(
                r"^INE[A-Z0-9]{9}$", na=False
            )
        ]

        # Prefer EQ when a symbol has multiple series
        df["SeriesPriority"] = df["SctySrs"].map(
            {"EQ": 0, "BE": 1}
        )

        df = df.sort_values("SeriesPriority")
        df = df.drop_duplicates(
            subset=["TckrSymb"],
            keep="first"
        )

        nse_stocks = {}

        for _, row in df.iterrows():

            company = row["FinInstrmNm"].strip()
            symbol = row["TckrSymb"].strip()

            if not company or not symbol:
                continue

            nse_stocks[company] = {
                "symbol": f"{symbol}.NS",
                "news_query": f'"{company}"'
            }

        return nse_stocks

    except Exception as e:
        st.warning(
            f"Could not load NSE stock list: {e}"
        )
        return {}


# Merge NSE stocks with the original five stocks
nse_stocks = load_nse_stocks()

stocks.update(nse_stocks)



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
# 4. MACHINE LEARNING PRICE FORECASTING
# =====================================================

def forecast_stock_prices(close_prices, forecast_days=20):

    prices = pd.Series(close_prices).dropna().astype(float).tolist()

    if len(prices) < 60:
        raise ValueError(
            "At least 60 trading days of data are required."
        )

    def create_features(values):

        return [
            values[-1],
            values[-2],
            values[-3],
            values[-5],
            values[-10],
            values[-20],
            np.mean(values[-5:]),
            np.mean(values[-10:]),
            np.mean(values[-20:]),
            np.std(values[-5:]),
            np.std(values[-20:])
        ]

    X = []
    y = []

    for i in range(20, len(prices) - 1):

        X.append(create_features(prices[:i + 1]))
        y.append(prices[i + 1])

    X = np.array(X)
    y = np.array(y)

    # Chronological train-test split
    split = int(len(X) * 0.80)

    if split < 20 or len(X) - split < 2:
        raise ValueError("Insufficient data for model evaluation.")

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # Random Forest model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Model performance
    predictions = model.predict(X_test)

    mae = np.mean(np.abs(y_test - predictions))

    baseline_mae = np.mean(
        np.abs(y_test - X_test[:, 0])
    )

    # Retrain using all available observations
    model.fit(X, y)

    # Iterative future price prediction
    history = prices.copy()
    future_prices = []

    for _ in range(forecast_days):

        features = np.array(
            create_features(history)
        ).reshape(1, -1)

        next_price = float(model.predict(features)[0])

        next_price = max(next_price, 0)

        future_prices.append(next_price)

        history.append(next_price)

    return {
        "future_prices": future_prices,
        "mae": float(mae),
        "baseline_mae": float(baseline_mae)
    }

def forecast_four_year_scenarios(symbol, current_price, simulations=10000):
    """Bootstrap historical monthly returns into 1-4 year scenario ranges.

    These are illustrative scenarios based on historical monthly returns,
    not reliable point forecasts or investment recommendations.
    """
    monthly = yf.Ticker(symbol).history(
        period="10y",
        interval="1mo",
        auto_adjust=True
    )

    if monthly.empty or "Close" not in monthly.columns:
        raise ValueError("Unable to retrieve monthly history for the 4-year projection.")

    closes = monthly["Close"].dropna().astype(float)
    monthly_returns = closes.pct_change().dropna()
    monthly_returns = monthly_returns.replace([np.inf, -np.inf], np.nan).dropna()

    if len(monthly_returns) < 36:
        raise ValueError(
            "At least 3 years of monthly history are needed for this scenario analysis."
        )

    # Bootstrap observed monthly returns to create many possible paths.
    rng = np.random.default_rng(42)
    sampled_returns = rng.choice(
        monthly_returns.to_numpy(),
        size=(simulations, 48),
        replace=True
    )

    # Avoid invalid negative prices if a source contains a bad return observation.
    sampled_returns = np.maximum(sampled_returns, -0.99)
    paths = float(current_price) * np.cumprod(1.0 + sampled_returns, axis=1)

    rows = []
    percentiles = {}
    for year in range(1, 5):
        values = paths[:, year * 12 - 1]
        p10, p50, p90 = np.percentile(values, [10, 50, 90])
        percentiles[year] = {"low": float(p10), "base": float(p50), "high": float(p90)}
        rows.append({
            "Horizon": f"Year {year}",
            "Pessimistic (10th percentile)": round(float(p10), 2),
            "Base case (median)": round(float(p50), 2),
            "Optimistic (90th percentile)": round(float(p90), 2)
        })

    historical_cagr = (closes.iloc[-1] / closes.iloc[0]) ** (12 / max(len(closes) - 1, 1)) - 1

    return {
        "table": pd.DataFrame(rows),
        "percentiles": percentiles,
        "historical_months": int(len(monthly_returns)),
        "historical_cagr": float(historical_cagr)
    }


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
        "Search / Select Company",
        options=sorted(stocks.keys()),
        index=0
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

        
        # ==========================================
        # 15. FUTURE STOCK PRICE PREDICTION
        # ==========================================

        st.divider()

        st.header("🔮 Future Stock Price Prediction")

        st.write(
            "Explore estimated future stock prices "
            "using machine learning."
        )

        if st.button("Generate Future Price Forecast"):

            try:
                with st.spinner("Generating forecast..."):

                    forecast = forecast_stock_prices(
                        data["Close"].dropna().values
                    )

                    st.session_state["stock_forecast"] = {
                        "symbol": symbol,
                        "period": period,
                        "result": forecast
                    }

            except Exception as e:
                st.error(f"Forecasting error: {e}")

        saved_forecast = st.session_state.get("stock_forecast")

        if (
            saved_forecast
            and saved_forecast["symbol"] == symbol
            and saved_forecast["period"] == period
        ):

            result = saved_forecast["result"]

            current_price = float(data["Close"].iloc[-1])

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "5-Day Forecast",
                f"₹{result['future_prices'][4]:,.2f}"
            )

            col2.metric(
                "10-Day Forecast",
                f"₹{result['future_prices'][9]:,.2f}"
            )

            col3.metric(
                "20-Day Forecast",
                f"₹{result['future_prices'][19]:,.2f}"
            )

            st.subheader("Model Performance")

            c1, c2 = st.columns(2)

            c1.metric("Random Forest MAE", f"₹{result['mae']:,.2f}")

            c2.metric(
                "Baseline MAE",
                f"₹{result['baseline_mae']:,.2f}"
            )

            st.subheader("Future Price Projection")

            future_dates = pd.bdate_range(
                start=pd.Timestamp(data.index[-1]).normalize()
                     + pd.offsets.BDay(1),
                periods=20
            )

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=data.index[-30:],
                y=data["Close"].tail(30),
                name="Historical Price"
            ))

            fig.add_trace(go.Scatter(
                x=future_dates,
                y=result["future_prices"],
                name="Predicted Price",
                mode="lines+markers",
                line=dict(dash="dash")
            ))

            fig.update_layout(
                height=450,
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Price (INR)"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Predictions are experimental estimates, "
                "not guaranteed future prices or investment advice."
            )

        # =================================================
        # 15. FOUR-YEAR PRICE SCENARIO ANALYSIS
        # =================================================

        st.divider()
        st.header("📅 Four-Year Stock Price Outlook")
        st.write(
            "Explore a range of possible prices over the next four years, "
            "based on resampling the stock's historical monthly returns."
        )
        st.caption(
            "This is a historical-return scenario simulation, not a machine-learning "
            "certainty or a guaranteed price target. Actual prices may fall outside these ranges."
        )

        if st.button("Generate 4-Year Price Scenarios"):
            try:
                with st.spinner("Simulating 4-year scenarios using up to 10 years of monthly history..."):
                    long_term_result = forecast_four_year_scenarios(
                        symbol,
                        float(data["Close"].dropna().iloc[-1])
                    )
                    st.session_state["four_year_forecast"] = {
                        "symbol": symbol,
                        "result": long_term_result
                    }
            except Exception as e:
                st.error(f"4-year scenario analysis failed: {e}")

        saved_long_term = st.session_state.get("four_year_forecast")
        if saved_long_term and saved_long_term.get("symbol") == symbol:
            long_result = saved_long_term["result"]
            st.subheader("Estimated Price Ranges")
            st.dataframe(long_result["table"], use_container_width=True, hide_index=True)

            st.metric(
                "Historical annualised growth over retrieved monthly history",
                f"{long_result['historical_cagr'] * 100:.2f}%"
            )
            st.caption(
                f"Simulation resampled {long_result['historical_months']} historical monthly returns "
                "across 10,000 possible paths. The 10th, 50th and 90th percentiles are scenario ranges, "
                "not confidence guarantees. Past performance does not predict future results."
            )

            projection_years = [0, 1, 2, 3, 4]
            current_for_chart = float(data["Close"].dropna().iloc[-1])
            low_line = [current_for_chart] + [long_result["percentiles"][y]["low"] for y in range(1, 5)]
            base_line = [current_for_chart] + [long_result["percentiles"][y]["base"] for y in range(1, 5)]
            high_line = [current_for_chart] + [long_result["percentiles"][y]["high"] for y in range(1, 5)]
            labels = ["Today", "1 year", "2 years", "3 years", "4 years"]

            scenario_fig = go.Figure()
            scenario_fig.add_trace(go.Scatter(
                x=labels, y=high_line, name="Optimistic (90th percentile)",
                mode="lines+markers", line=dict(dash="dot")
            ))
            scenario_fig.add_trace(go.Scatter(
                x=labels, y=low_line, name="Pessimistic (10th percentile)",
                mode="lines+markers", fill="tonexty", line=dict(dash="dot")
            ))
            scenario_fig.add_trace(go.Scatter(
                x=labels, y=base_line, name="Base case (median)",
                mode="lines+markers", line=dict(width=3)
            ))
            scenario_fig.update_layout(
                height=480,
                template="plotly_white",
                xaxis_title="Forecast horizon",
                yaxis_title="Illustrative price (INR)",
                hovermode="x unified"
            )
            st.plotly_chart(scenario_fig, use_container_width=True)

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
