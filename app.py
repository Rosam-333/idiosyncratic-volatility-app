"""
Bob Cat Risk Lab — Core / original scope (separate entry point).

Run:  streamlit run app_classic.py

CAPM-style rolling idiosyncratic volatility, compare, S&P screener, and Yahoo Finance
headlines only (no Google News, RSS wire harvest, regional papers, or central-bank layer).

For the full extended app:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from bobcat_lab_core import (
    STOCK_MARKET_BENCHMARK,
    CRYPTO_OPTIONS,
    build_prediction_signal,
    build_sp500_screener,
    build_relevance_terms,
    classify_catalysts,
    classify_risk_profile,
    compute_analysis,
    compute_ivol_percentile,
    fetch_yahoo_headlines,
    format_market_caption,
    format_profile_line,
    get_company_profile,
    get_market_snapshot,
    get_stock_catalog,
    initialize_app_state,
    load_asset_vs_benchmark,
    build_takeaway,
    render_date_messages,
)

st.set_page_config(page_title="Bob Cat Risk Lab (Core)", layout="wide")

st.markdown("### Group: Bob Cat")
st.title("Bob Cat Risk Lab — Core (original scope)")
st.caption(
    "**Separate app** from the extended build. This version uses **Yahoo Finance headlines only**. "
    "Run `streamlit run app.py` for multi-source news, policy/central-bank context, and prediction verification."
)


@st.cache_data(ttl=600, show_spinner=False)
def get_yahoo_only_news_context(asset_ticker, asset_name, sector=None, industry=None, country=None):
    relevance_terms = build_relevance_terms(asset_name, asset_ticker, sector, industry)
    articles = fetch_yahoo_headlines(asset_ticker, max_items=8)
    notes = ["Core app: **Yahoo Finance** headlines only."]
    if not articles:
        notes.append("No Yahoo Finance headlines returned for this ticker.")
    return {
        "articles": articles,
        "coverage_notes": notes,
        "source_counts": {"Yahoo Finance": len(articles)} if articles else {},
        "relevance_terms": relevance_terms,
        "country": country,
    }


def render_compare_dashboard(name_a, ticker_a, profile_a, frame_a, metrics_a, name_b, ticker_b, profile_b, frame_b, metrics_b, benchmark_label):
    st.subheader("Side-by-Side Comparison")
    left, right = st.columns(2)
    snapshot_a = get_market_snapshot(ticker_a)
    snapshot_b = get_market_snapshot(ticker_b)
    with left:
        st.markdown(f"### {name_a} (`{ticker_a}`)")
        line_a = format_profile_line(profile_a)
        if line_a:
            st.markdown(line_a)
        if snapshot_a:
            st.caption(format_market_caption(snapshot_a))
    with right:
        st.markdown(f"### {name_b} (`{ticker_b}`)")
        line_b = format_profile_line(profile_b)
        if line_b:
            st.markdown(line_b)
        if snapshot_b:
            st.caption(format_market_caption(snapshot_b))

    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "Beta",
                "Alpha (daily)",
                "R-squared",
                "Annualized Return",
                "Annualized Total Volatility",
                "Annualized Idiosyncratic Volatility",
                "Current Rolling IVOL",
            ],
            name_a: [
                metrics_a["beta"],
                metrics_a["alpha_daily"],
                metrics_a["r_squared"],
                metrics_a["annualized_return"],
                metrics_a["total_vol_annual"],
                metrics_a["idio_vol_annual"],
                metrics_a["rolling_iv_current"],
            ],
            name_b: [
                metrics_b["beta"],
                metrics_b["alpha_daily"],
                metrics_b["r_squared"],
                metrics_b["annualized_return"],
                metrics_b["total_vol_annual"],
                metrics_b["idio_vol_annual"],
                metrics_b["rolling_iv_current"],
            ],
        }
    )
    st.dataframe(comparison_df, use_container_width=True)

    normalized_a = frame_a["asset_close"] / frame_a["asset_close"].iloc[0]
    normalized_b = frame_b["asset_close"] / frame_b["asset_close"].iloc[0]
    benchmark_norm_a = frame_a["benchmark_close"] / frame_a["benchmark_close"].iloc[0]

    tabs = st.tabs(["Normalized Performance", "Rolling IVOL Comparison", "Quick Interpretation"])
    with tabs[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normalized_a.index, y=normalized_a, mode="lines", name=ticker_a))
        fig.add_trace(go.Scatter(x=normalized_b.index, y=normalized_b, mode="lines", name=ticker_b))
        fig.add_trace(go.Scatter(x=benchmark_norm_a.index, y=benchmark_norm_a, mode="lines", name=benchmark_label))
        fig.update_layout(xaxis_title="Date", yaxis_title="Normalized Value (Start = 1)")
        st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frame_a.index, y=frame_a["rolling_iv"], mode="lines", name=ticker_a))
        fig.add_trace(go.Scatter(x=frame_b.index, y=frame_b["rolling_iv"], mode="lines", name=ticker_b))
        fig.update_layout(xaxis_title="Date", yaxis_title="Annualized Idiosyncratic Volatility")
        st.plotly_chart(fig, use_container_width=True)
    with tabs[2]:
        profile_label_a = classify_risk_profile(metrics_a["r_squared"], metrics_a["idio_vol_annual"], metrics_a["total_vol_annual"])
        profile_label_b = classify_risk_profile(metrics_b["r_squared"], metrics_b["idio_vol_annual"], metrics_b["total_vol_annual"])
        st.info(
            f"{name_a} is classified as **{profile_label_a}**, while {name_b} is classified as **{profile_label_b}**."
        )
        if metrics_a["idio_vol_annual"] > metrics_b["idio_vol_annual"]:
            st.warning(f"{name_a} has the higher annualized idiosyncratic volatility in this sample.")
        else:
            st.warning(f"{name_b} has the higher annualized idiosyncratic volatility in this sample.")


def render_core_dashboard(asset_name, asset_ticker, profile, benchmark_label, frame, metrics, prediction):
    ivol_percentile = compute_ivol_percentile(frame["rolling_iv"])
    risk_profile = classify_risk_profile(metrics["r_squared"], metrics["idio_vol_annual"], metrics["total_vol_annual"])
    takeaway = build_takeaway(asset_name, risk_profile, metrics, benchmark_label)
    market_snapshot = get_market_snapshot(asset_ticker)
    idio_share = metrics["idio_vol_annual"] / metrics["total_vol_annual"] if metrics["total_vol_annual"] > 0 else np.nan

    st.markdown(f"**Selected Asset:** {asset_name} (`{asset_ticker}`)")
    profile_line = format_profile_line(profile)
    if profile_line:
        st.markdown(profile_line)

    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col1.metric("Beta", f"{metrics['beta']:.3f}")
    col2.metric("Alpha (daily)", f"{metrics['alpha_daily']:.5f}")
    col3.metric("R-squared", f"{metrics['r_squared']:.3f}")
    col4.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
    col5.metric("Annualized Total Volatility", f"{metrics['total_vol_annual']:.2%}")
    col6.metric("Annualized Idiosyncratic Volatility", f"{metrics['idio_vol_annual']:.2%}")

    if market_snapshot:
        st.subheader("Current Market Snapshot")
        snap1, snap2, snap3 = st.columns(3)
        price = market_snapshot.get("price")
        change_pct = market_snapshot.get("change_pct")
        market_time = market_snapshot.get("market_time")
        snap1.metric("Latest Market Price", f"{price:.2f}" if price is not None else "N/A")
        snap2.metric("Latest Change", f"{change_pct:.2%}" if change_pct is not None else "N/A")
        snap3.metric("Market Timestamp", str(market_time) if market_time is not None else "N/A")

    st.subheader("Historical Risk Context")
    ctx1, ctx2, ctx3, ctx4 = st.columns(4)
    ctx1.metric("Current Rolling IVOL", f"{metrics['rolling_iv_current']:.2%}" if pd.notna(metrics["rolling_iv_current"]) else "N/A")
    ctx2.metric("Average Rolling IVOL", f"{metrics['rolling_iv_average']:.2%}" if pd.notna(metrics["rolling_iv_average"]) else "N/A")
    ctx3.metric("IVOL Percentile", f"{ivol_percentile:.0f}th" if pd.notna(ivol_percentile) else "N/A")
    ctx4.metric("Firm-Specific Share of Risk", f"{idio_share:.0%}" if pd.notna(idio_share) else "N/A")

    st.subheader("Risk Interpretation")
    st.info(f"**Risk Profile:** {risk_profile}")
    st.success(f"**Investment Takeaway:** {takeaway}")
    st.subheader("Educational Outlook")
    st.warning(f"**Preferred Action:** {prediction['action']} ({prediction['confidence']} confidence)")
    st.write(prediction["summary"])
    st.caption(prediction["risk_text"])

    if prediction["grouped_articles"]:
        with st.expander("Yahoo Finance headlines used for context"):
            for catalyst, items in prediction["grouped_articles"].items():
                st.markdown(f"**{catalyst.upper()}**")
                for item in items[:5]:
                    if item.get("url"):
                        st.markdown(f"- [{item['title']}]({item['url']})")
                    else:
                        st.markdown(f"- {item['title']}")
                    st.caption(item.get("source") or item.get("source_type", "Source"))
    elif prediction["coverage_notes"]:
        st.info(" | ".join(prediction["coverage_notes"]))

    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Price Chart", "Return Scatter", "Rolling IVOL"])
    with chart_tab1:
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=frame.index, y=frame["asset_close"], mode="lines", name=asset_ticker))
        price_fig.add_trace(go.Scatter(x=frame.index, y=frame["benchmark_close"], mode="lines", name=benchmark_label))
        price_fig.update_layout(xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(price_fig, use_container_width=True)
    with chart_tab2:
        scatter_fig = go.Figure()
        scatter_fig.add_trace(
            go.Scatter(
                x=frame["benchmark_return"],
                y=frame["asset_return"],
                mode="markers",
                name="Daily Returns",
            )
        )
        scatter_fig.update_layout(
            title=f"{asset_ticker} Return vs {benchmark_label} Return",
            xaxis_title=f"{benchmark_label} Return",
            yaxis_title=f"{asset_ticker} Return",
        )
        st.plotly_chart(scatter_fig, use_container_width=True)
    with chart_tab3:
        rolling_fig = go.Figure()
        rolling_fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["rolling_iv"],
                mode="lines",
                name="Rolling Idiosyncratic Volatility",
            )
        )
        rolling_fig.update_layout(xaxis_title="Date", yaxis_title="Annualized Idiosyncratic Volatility")
        st.plotly_chart(rolling_fig, use_container_width=True)

    st.subheader("Model Summary")
    results_df = pd.DataFrame(
        {
            "Metric": [
                "Alpha (daily)",
                "Beta",
                "R-squared",
                "Daily Idiosyncratic Volatility",
                "Current Rolling IVOL",
                "IVOL Percentile",
            ],
            "Value": [
                metrics["alpha_daily"],
                metrics["beta"],
                metrics["r_squared"],
                metrics["idio_vol_daily"],
                metrics["rolling_iv_current"],
                ivol_percentile / 100 if pd.notna(ivol_percentile) else np.nan,
            ],
        }
    )
    st.dataframe(results_df, use_container_width=True)


stock_catalog = get_stock_catalog()
stock_labels = stock_catalog["label"].tolist()
stock_label_to_ticker = dict(zip(stock_catalog["label"], stock_catalog["ticker"]))
default_stock_label = next((label for label in stock_labels if label.endswith("(AAPL)")), stock_labels[0] if stock_labels else "")

initialize_app_state()

st.sidebar.header("Controls")
asset_universe = st.sidebar.radio("Asset Universe", ["S&P 500 Stocks", "Cryptocurrencies"])

if asset_universe == "S&P 500 Stocks":
    analysis_mode = st.sidebar.radio(
        "Stock Mode",
        ["Single Company", "Compare Two Companies", "Full S&P 500 Screener"],
    )
    if analysis_mode == "Single Company":
        selected_label = st.sidebar.selectbox("Search company name", stock_labels, index=stock_labels.index(default_stock_label))
    elif analysis_mode == "Compare Two Companies":
        selected_label_a = st.sidebar.selectbox("Company 1", stock_labels, index=stock_labels.index(default_stock_label))
        default_b = next((label for label in stock_labels if label.endswith("(MSFT)")), stock_labels[1] if len(stock_labels) > 1 else stock_labels[0])
        selected_label_b = st.sidebar.selectbox("Company 2", stock_labels, index=stock_labels.index(default_b))
    else:
        st.sidebar.caption("Screens the current S&P 500 list by idiosyncratic volatility.")
else:
    analysis_mode = st.sidebar.radio("Crypto Mode", ["Single Crypto", "Compare Two Cryptos"])
    crypto_labels = list(CRYPTO_OPTIONS.keys())
    if analysis_mode == "Single Crypto":
        selected_crypto_label = st.sidebar.selectbox("Choose cryptocurrency", crypto_labels, index=0)
        benchmark_crypto_options = [label for label in crypto_labels if label != selected_crypto_label]
        selected_crypto_benchmark = st.sidebar.selectbox("Crypto benchmark", benchmark_crypto_options, index=0)
    else:
        selected_crypto_a = st.sidebar.selectbox("Crypto 1", crypto_labels, index=0)
        remaining = [label for label in crypto_labels if label != selected_crypto_a]
        selected_crypto_b = st.sidebar.selectbox("Crypto 2", remaining, index=0)
        benchmark_options = [label for label in crypto_labels if label not in {selected_crypto_a, selected_crypto_b}]
        selected_crypto_benchmark = st.sidebar.selectbox("Comparison benchmark", benchmark_options, index=0 if benchmark_options else 0)

with st.sidebar.form("classic_analysis_form"):
    start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-02"))
    end_date = st.date_input("End date", value=pd.Timestamp.today().date())
    rolling_window = st.slider("Rolling window (days)", 20, 120, 30)
    submitted = st.form_submit_button("Run Analysis")

if submitted:
    request_state = {
        "asset_universe": asset_universe,
        "analysis_mode": analysis_mode,
        "start_date": start_date,
        "end_date": end_date,
        "rolling_window": rolling_window,
    }
    if asset_universe == "S&P 500 Stocks":
        if analysis_mode == "Single Company":
            request_state["selected_label"] = selected_label
        elif analysis_mode == "Compare Two Companies":
            request_state["selected_label_a"] = selected_label_a
            request_state["selected_label_b"] = selected_label_b
    else:
        if analysis_mode == "Single Crypto":
            request_state["selected_crypto_label"] = selected_crypto_label
            request_state["selected_crypto_benchmark"] = selected_crypto_benchmark
        else:
            request_state["selected_crypto_a"] = selected_crypto_a
            request_state["selected_crypto_b"] = selected_crypto_b
            request_state["selected_crypto_benchmark"] = selected_crypto_benchmark
    st.session_state["analysis_request"] = request_state

analysis_request = st.session_state.get("analysis_request")
if not analysis_request:
    st.info("Use the sidebar, then click **Run Analysis**.")
    st.stop()

start_date = analysis_request["start_date"]
end_date = analysis_request["end_date"]
rolling_window = analysis_request["rolling_window"]
asset_universe = analysis_request["asset_universe"]
analysis_mode = analysis_request["analysis_mode"]

if start_date >= end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()

if asset_universe == "S&P 500 Stocks":
    if analysis_mode == "Single Company":
        ticker = stock_label_to_ticker[analysis_request["selected_label"]]
        profile = get_company_profile(ticker)
        frame = load_asset_vs_benchmark(ticker, STOCK_MARKET_BENCHMARK, start_date, end_date)
        if frame is None or frame.empty:
            st.error("No stock data was downloaded. Please try another date range.")
            st.stop()
        _, enriched, metrics = compute_analysis(frame, rolling_window)
        risk_profile = classify_risk_profile(metrics["r_squared"], metrics["idio_vol_annual"], metrics["total_vol_annual"])
        market_snapshot = get_market_snapshot(ticker)
        with st.spinner("Loading Yahoo Finance headlines..."):
            news_context = get_yahoo_only_news_context(
                ticker, profile["name"], profile.get("sector"), profile.get("industry"), profile.get("country")
            )
        classified_context = classify_catalysts(news_context, sector=profile.get("sector"), industry=profile.get("industry"))
        prediction = build_prediction_signal(
            profile["name"],
            metrics,
            risk_profile,
            market_snapshot,
            news_context,
            classified_context,
            news_coverage_label="headlines from Yahoo Finance (core app)",
        )
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Company:** {profile['name']}")
        render_date_messages(metrics, start_date, end_date)
        render_core_dashboard(profile["name"], ticker, profile, STOCK_MARKET_BENCHMARK, enriched, metrics, prediction)

    elif analysis_mode == "Compare Two Companies":
        ticker_a = stock_label_to_ticker[analysis_request["selected_label_a"]]
        ticker_b = stock_label_to_ticker[analysis_request["selected_label_b"]]
        if ticker_a == ticker_b:
            st.error("Please choose two different companies.")
            st.stop()
        profile_a = get_company_profile(ticker_a)
        profile_b = get_company_profile(ticker_b)
        frame_a = load_asset_vs_benchmark(ticker_a, STOCK_MARKET_BENCHMARK, start_date, end_date)
        frame_b = load_asset_vs_benchmark(ticker_b, STOCK_MARKET_BENCHMARK, start_date, end_date)
        if frame_a is None or frame_a.empty or frame_b is None or frame_b.empty:
            st.error("One of the companies could not be loaded.")
            st.stop()
        _, enriched_a, metrics_a = compute_analysis(frame_a, rolling_window)
        _, enriched_b, metrics_b = compute_analysis(frame_b, rolling_window)
        render_date_messages(metrics_a, start_date, end_date)
        render_compare_dashboard(
            profile_a["name"], ticker_a, profile_a, enriched_a, metrics_a,
            profile_b["name"], ticker_b, profile_b, enriched_b, metrics_b,
            STOCK_MARKET_BENCHMARK,
        )
    else:
        st.subheader("Full S&P 500 Screener")
        with st.spinner("Running screener..."):
            screener_df = build_sp500_screener(start_date, end_date, rolling_window)
        if screener_df.empty:
            st.error("Screener could not be generated for this range.")
            st.stop()
        st.dataframe(screener_df, use_container_width=True)
        st.download_button(
            "Download CSV",
            screener_df.to_csv(index=False).encode("utf-8"),
            file_name="sp500_ivol_screener_core.csv",
            mime="text/csv",
        )

else:
    if analysis_mode == "Single Crypto":
        crypto_ticker = CRYPTO_OPTIONS[analysis_request["selected_crypto_label"]]
        benchmark_ticker = CRYPTO_OPTIONS[analysis_request["selected_crypto_benchmark"]]
        frame = load_asset_vs_benchmark(crypto_ticker, benchmark_ticker, start_date, end_date)
        if frame is None or frame.empty:
            st.error("Crypto data could not be loaded.")
            st.stop()
        _, enriched, metrics = compute_analysis(frame, rolling_window)
        profile = {
            "name": analysis_request["selected_crypto_label"].replace(f" ({crypto_ticker})", ""),
            "sector": "Digital Asset",
            "industry": "Cryptocurrency",
        }
        risk_profile = classify_risk_profile(metrics["r_squared"], metrics["idio_vol_annual"], metrics["total_vol_annual"])
        market_snapshot = get_market_snapshot(crypto_ticker)
        with st.spinner("Loading Yahoo Finance headlines..."):
            news_context = get_yahoo_only_news_context(
                crypto_ticker, profile["name"], profile.get("sector"), profile.get("industry"), None
            )
        classified_context = classify_catalysts(news_context, sector=profile.get("sector"), industry=profile.get("industry"))
        prediction = build_prediction_signal(
            profile["name"],
            metrics,
            risk_profile,
            market_snapshot,
            news_context,
            classified_context,
            news_coverage_label="headlines from Yahoo Finance (core app)",
        )
        render_date_messages(metrics, start_date, end_date)
        render_core_dashboard(analysis_request["selected_crypto_label"], crypto_ticker, profile, benchmark_ticker, enriched, metrics, prediction)
    else:
        crypto_ticker_a = CRYPTO_OPTIONS[analysis_request["selected_crypto_a"]]
        crypto_ticker_b = CRYPTO_OPTIONS[analysis_request["selected_crypto_b"]]
        benchmark_ticker = CRYPTO_OPTIONS[analysis_request["selected_crypto_benchmark"]]
        frame_a = load_asset_vs_benchmark(crypto_ticker_a, benchmark_ticker, start_date, end_date)
        frame_b = load_asset_vs_benchmark(crypto_ticker_b, benchmark_ticker, start_date, end_date)
        if frame_a is None or frame_a.empty or frame_b is None or frame_b.empty:
            st.error("One of the cryptos could not be loaded.")
            st.stop()
        _, enriched_a, metrics_a = compute_analysis(frame_a, rolling_window)
        _, enriched_b, metrics_b = compute_analysis(frame_b, rolling_window)
        profile_a = {"name": analysis_request["selected_crypto_a"].replace(f" ({crypto_ticker_a})", ""), "sector": "Digital Asset", "industry": "Cryptocurrency"}
        profile_b = {"name": analysis_request["selected_crypto_b"].replace(f" ({crypto_ticker_b})", ""), "sector": "Digital Asset", "industry": "Cryptocurrency"}
        render_date_messages(metrics_a, start_date, end_date)
        render_compare_dashboard(
            analysis_request["selected_crypto_a"], crypto_ticker_a, profile_a, enriched_a, metrics_a,
            analysis_request["selected_crypto_b"], crypto_ticker_b, profile_b, enriched_b, metrics_b,
            benchmark_ticker,
        )

st.caption("Core app: CAPM residual IVOL methodology matches the extended `app.py`; news inputs differ.")
