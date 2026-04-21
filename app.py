import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from pathlib import Path
from io import StringIO
import requests
from yahooquery import Ticker as YahooQueryTicker


st.set_page_config(page_title="Idiosyncratic Volatility Analyzer", layout="wide")

gif_path = Path(__file__).with_name("finance.gif")

st.markdown("### Group: Bob Cat")

st.title("Idiosyncratic Volatility Analyzer")
st.write(
    "This app estimates a stock's idiosyncratic volatility using a CAPM-style "
    "regression against the market benchmark SPY."
)

if gif_path.exists():
    st.image(str(gif_path), caption="Finance-themed intro", use_container_width=True)

st.sidebar.header("Inputs")
with st.sidebar.form("analysis_form"):
    ticker = st.text_input("Stock ticker", value="AAPL").upper().strip()
    start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-02"))
    end_date = st.date_input("End date", value=pd.to_datetime("2025-12-31"))
    rolling_window = st.slider("Rolling window (days)", 20, 120, 30)
    submitted = st.form_submit_button("Submit")

if not submitted:
    st.info("Choose the inputs in the sidebar and click Submit to load the analysis.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()

market_ticker = "SPY"
FALLBACK_COMPANY_PROFILES = {
    "AAPL": {
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "sector": "Technology",
        "industry": "Software - Infrastructure",
    },
    "TSLA": {
        "name": "Tesla, Inc.",
        "sector": "Consumer Cyclical",
        "industry": "Auto Manufacturers",
    },
    "AMZN": {
        "name": "Amazon.com, Inc.",
        "sector": "Consumer Cyclical",
        "industry": "Internet Retail",
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
    },
    "META": {
        "name": "Meta Platforms, Inc.",
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
    },
    "NVDA": {
        "name": "NVIDIA Corporation",
        "sector": "Technology",
        "industry": "Semiconductors",
    },
}


def extract_close(downloaded_df):
    if downloaded_df.empty:
        return None

    if isinstance(downloaded_df.columns, pd.MultiIndex):
        close_data = downloaded_df["Close"]
        if isinstance(close_data, pd.DataFrame):
            return close_data.iloc[:, 0]
        return close_data

    return downloaded_df["Close"]


@st.cache_data
def get_sp500_reference():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        response.raise_for_status()
        table = pd.read_html(StringIO(response.text))[0]
    except Exception:
        return {}

    reference = {}
    for _, row in table.iterrows():
        symbol = str(row["Symbol"]).replace(".", "-")
        reference[symbol] = {
            "name": row.get("Security"),
            "sector": row.get("GICS Sector"),
            "industry": row.get("GICS Sub-Industry"),
        }
    return reference


@st.cache_data
def get_company_profile(stock_ticker):
    sp500_reference = get_sp500_reference()
    reference_profile = sp500_reference.get(stock_ticker, {})

    try:
        ticker_data = YahooQueryTicker(stock_ticker, asynchronous=False, validate=True, progress=False)
        price_info = ticker_data.price.get(stock_ticker, {})
        asset_profile = ticker_data.asset_profile.get(stock_ticker, {})
    except Exception:
        return reference_profile or FALLBACK_COMPANY_PROFILES.get(
            stock_ticker,
            {
                "name": stock_ticker,
                "sector": None,
                "industry": None,
            },
        )

    if not isinstance(price_info, dict):
        price_info = {}
    if not isinstance(asset_profile, dict):
        asset_profile = {}

    if not price_info and not asset_profile:
        return reference_profile or FALLBACK_COMPANY_PROFILES.get(
            stock_ticker,
            {
                "name": stock_ticker,
                "sector": None,
                "industry": None,
            },
        )

    profile = {
        "name": price_info.get("longName") or price_info.get("shortName") or stock_ticker,
        "sector": asset_profile.get("sector"),
        "industry": asset_profile.get("industry"),
    }
    fallback = reference_profile or FALLBACK_COMPANY_PROFILES.get(stock_ticker, {})
    if not profile["name"] or profile["name"] == stock_ticker:
        profile["name"] = fallback.get("name", stock_ticker)
    if not profile["sector"]:
        profile["sector"] = fallback.get("sector")
    if not profile["industry"]:
        profile["industry"] = fallback.get("industry")
    return profile


@st.cache_data
def load_data(stock_ticker, market_symbol, start, end):
    stock = yf.download(stock_ticker, start=start, end=end, auto_adjust=True, progress=False)
    market = yf.download(market_symbol, start=start, end=end, auto_adjust=True, progress=False)

    if stock.empty or market.empty:
        return None

    stock_close = extract_close(stock)
    market_close = extract_close(market)

    data = pd.DataFrame({
        "stock_close": stock_close,
        "market_close": market_close,
    }).dropna()

    if data.empty:
        return None

    data["stock_return"] = data["stock_close"].pct_change()
    data["market_return"] = data["market_close"].pct_change()
    data = data.dropna()

    return data


company_profile = get_company_profile(ticker)
data = load_data(ticker, market_ticker, start_date, end_date)

if data is None or data.empty:
    st.error("No data was downloaded. Please check the ticker and date range.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Company:** {company_profile['name']}")
if company_profile.get("sector"):
    st.sidebar.markdown(f"**Sector:** {company_profile['sector']}")
if company_profile.get("industry"):
    st.sidebar.markdown(f"**Industry:** {company_profile['industry']}")

st.markdown(f"**Selected Company:** {company_profile['name']} (`{ticker}`)")

profile_parts = []
if company_profile.get("sector"):
    profile_parts.append(f"**Sector:** {company_profile['sector']}")
if company_profile.get("industry"):
    profile_parts.append(f"**Industry:** {company_profile['industry']}")
if profile_parts:
    st.markdown(" | ".join(profile_parts))

actual_start = data.index.min().date()
actual_end = data.index.max().date()
if actual_start > start_date:
    st.info(
        f"Market data did not start exactly on {start_date}. The analysis begins on the next available trading day: {actual_start}."
    )
if actual_end < end_date:
    st.info(
        f"Market data did not extend exactly to {end_date}. The analysis ends on the most recent available trading day: {actual_end}."
    )

X = sm.add_constant(data["market_return"])
y = data["stock_return"]
model = sm.OLS(y, X).fit()

data["predicted_return"] = model.predict(X)
data["residual"] = data["stock_return"] - data["predicted_return"]
data["rolling_iv"] = data["residual"].rolling(rolling_window).std() * np.sqrt(252)

alpha_daily = model.params["const"]
beta = model.params["market_return"]
r_squared = model.rsquared
idio_vol_daily = data["residual"].std()
idio_vol_annual = idio_vol_daily * np.sqrt(252)
stock_vol_annual = data["stock_return"].std() * np.sqrt(252)
avg_daily_return = data["stock_return"].mean()
annualized_return = avg_daily_return * 252

st.subheader("Summary Metrics")
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

col1.metric("Beta", f"{beta:.3f}")
col2.metric("Alpha (daily)", f"{alpha_daily:.5f}")
col3.metric("R-squared", f"{r_squared:.3f}")
col4.metric("Annualized Return", f"{annualized_return:.2%}")
col5.metric("Annualized Total Volatility", f"{stock_vol_annual:.2%}")
col6.metric("Annualized Idiosyncratic Volatility", f"{idio_vol_annual:.2%}")

st.subheader("Economic Interpretation")
if r_squared >= 0.5:
    st.info(
        "A relatively large part of this stock's movements is explained by the market. "
        "Firm-specific risk is present, but market-wide forces are also important."
    )
else:
    st.info(
        "A relatively small part of this stock's movements is explained by the market. "
        "Firm-specific factors appear to play a larger role."
    )

if idio_vol_annual >= stock_vol_annual * 0.5:
    st.warning(
        "Idiosyncratic volatility is high relative to total volatility, suggesting that "
        "company-specific news may be an important driver of returns."
    )
else:
    st.success(
        "Idiosyncratic volatility is moderate relative to total volatility, suggesting that "
        "market movements explain a meaningful share of the stock's risk."
    )

st.subheader("Adjusted Closing Prices")
price_fig = go.Figure()
price_fig.add_trace(
    go.Scatter(x=data.index, y=data["stock_close"], mode="lines", name=ticker)
)
price_fig.add_trace(
    go.Scatter(x=data.index, y=data["market_close"], mode="lines", name=market_ticker)
)
price_fig.update_layout(xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(price_fig, use_container_width=True)

st.subheader("Stock vs Market Daily Returns")
scatter_fig = go.Figure()
scatter_fig.add_trace(
    go.Scatter(
        x=data["market_return"],
        y=data["stock_return"],
        mode="markers",
        name="Daily Returns",
    )
)
scatter_fig.update_layout(
    title=f"{ticker} Return vs Market Return",
    xaxis_title="Market Return (SPY)",
    yaxis_title=f"{ticker} Return",
)
st.plotly_chart(scatter_fig, use_container_width=True)

st.subheader("Rolling Idiosyncratic Volatility")
rolling_fig = go.Figure()
rolling_fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["rolling_iv"],
        mode="lines",
        name="Rolling Idiosyncratic Volatility",
    )
)
rolling_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Annualized Idiosyncratic Volatility",
)
st.plotly_chart(rolling_fig, use_container_width=True)

st.subheader("Regression Table")
results_df = pd.DataFrame(
    {
        "Metric": [
            "Alpha (daily)",
            "Beta",
            "R-squared",
            "Daily Idiosyncratic Volatility",
        ],
        "Value": [alpha_daily, beta, r_squared, idio_vol_daily],
    }
)
st.dataframe(results_df, use_container_width=True)

st.caption(
    "Method: CAPM-style regression of stock daily returns on SPY daily returns. "
    "Idiosyncratic volatility is measured as the standard deviation of regression residuals."
)
