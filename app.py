"""
Bob Cat Risk Lab — Core (default entry point).

Self-contained for Streamlit Cloud: shared engine is inlined below (same code as
`bobcat_lab_core.py`). Locally, keep editing `bobcat_lab_core.py` and re-run:
  python _merge_app_core.py

Run: streamlit run app.py
Extended: streamlit run app_extended.py
"""

# --- Shared engine (bobcat_lab_core) ---

import re
from io import StringIO

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import streamlit as st
import yfinance as yf
from yahooquery import Ticker as YahooQueryTicker

TRADING_DAYS = 252
STOCK_MARKET_BENCHMARK = "SPY"
CRYPTO_OPTIONS = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Solana (SOL-USD)": "SOL-USD",
    "XRP (XRP-USD)": "XRP-USD",
    "Dogecoin (DOGE-USD)": "DOGE-USD",
    "Cardano (ADA-USD)": "ADA-USD",
    "Binance Coin (BNB-USD)": "BNB-USD",
}

FALLBACK_COMPANY_PROFILES = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics"},
    "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "industry": "Software - Infrastructure"},
    "TSLA": {"name": "Tesla, Inc.", "sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
    "AMZN": {"name": "Amazon.com, Inc.", "sector": "Consumer Cyclical", "industry": "Internet Retail"},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Communication Services", "industry": "Interactive Media & Services"},
    "META": {"name": "Meta Platforms, Inc.", "sector": "Communication Services", "industry": "Interactive Media & Services"},
    "NVDA": {"name": "NVIDIA Corporation", "sector": "Technology", "industry": "Semiconductors"},
    "WMT": {"name": "Walmart Inc.", "sector": "Consumer Staples", "industry": "Consumer Staples Merchandise Retail"},
    "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Financials", "industry": "Diversified Banks"},
    "KO": {"name": "The Coca-Cola Company", "sector": "Consumer Staples", "industry": "Soft Drinks & Non-alcoholic Beverages"},
    "XOM": {"name": "Exxon Mobil Corporation", "sector": "Energy", "industry": "Integrated Oil & Gas"},
}

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}

SECTOR_EXPOSURES = {
    "Technology": ["AI demand", "chip export rules", "cloud spending", "antitrust"],
    "Communication Services": ["advertising demand", "platform regulation", "AI competition", "consumer sentiment"],
    "Consumer Cyclical": ["consumer demand", "tariffs", "supply chain", "interest rates"],
    "Consumer Staples": ["pricing power", "input costs", "consumer demand", "inflation"],
    "Energy": ["oil supply", "OPEC", "sanctions", "refining margins"],
    "Financials": ["interest rates", "credit quality", "capital rules", "bank regulation"],
    "Health Care": ["drug approvals", "FDA regulation", "reimbursement", "clinical trials"],
    "Industrials": ["trade policy", "infrastructure spending", "defense demand", "supply chain"],
    "Materials": ["commodity prices", "China demand", "trade tariffs", "energy costs"],
    "Real Estate": ["interest rates", "occupancy", "credit markets", "commercial property"],
    "Utilities": ["rate cases", "grid regulation", "fuel costs", "weather demand"],
    "Digital Asset": ["crypto regulation", "spot ETF flows", "risk appetite", "rates"],
}
CATALYST_PATTERNS = {
    "earnings": [
        "earnings", "revenue", "guidance", "forecast", "profit", "margin", "results",
        "beat", "miss", "sales", "quarter", "dividend", "buyback",
    ],
    "m&a": [
        "merger", "acquisition", "acquire", "deal", "takeover", "buyout", "stake",
        "spin-off", "spinoff", "divest", "joint venture",
    ],
    "policy": [
        "regulation", "regulatory", "policy", "tariff", "tax", "sanction", "subsidy",
        "antitrust", "compliance", "export rule", "export control", "approval", "ban",
    ],
    "macro": [
        "inflation", "interest rate", "fed", "central bank", "gdp", "recession", "jobs",
        "consumer confidence", "oil", "currency", "bond yield", "treasury", "macro",
    ],
    "legal": [
        "lawsuit", "probe", "investigation", "litigation", "settlement", "fraud", "court",
        "fine", "penalty", "charge", "class action",
    ],
    "management": [
        "ceo", "cfo", "chairman", "board", "executive", "appoint", "resign", "step down",
        "leadership", "director",
    ],
    "product": [
        "launch", "release", "product", "platform", "model", "service", "partnership",
        "contract", "customer", "rollout", "approval", "innovation",
    ],
    "sector": [
        "demand", "pricing", "competition", "inventory", "capacity", "supply chain",
        "shipment", "market share", "industry", "sector",
    ],
}
POSITIVE_SIGNAL_WORDS = [
    "beat", "strong", "growth", "surge", "record", "upgrade", "gain", "profit",
    "expand", "bullish", "approval", "launch", "outperform", "rebound", "demand",
]
NEGATIVE_SIGNAL_WORDS = [
    "miss", "drop", "weak", "cut", "lawsuit", "downgrade", "loss", "fall", "probe",
    "bearish", "recall", "slowdown", "tariff", "sanction", "investigation",
]
DEFAULT_ARTICLE_BIAS = {
    "legal": -1.0,
    "policy": -0.2,
    "macro": 0.0,
    "earnings": 0.0,
    "m&a": 0.2,
    "management": 0.0,
    "product": 0.4,
    "sector": 0.0,
}
SOURCE_WEIGHTS = {
    "google": 1.0,
    "rss": 1.05,
    "gdelt": 1.0,
    "yahoo": 0.9,
    "google_outlet": 1.05,
    "google_central_bank": 1.1,
}


def extract_close(downloaded_df, asset_ticker=None):
    if downloaded_df.empty:
        return None

    if isinstance(downloaded_df.columns, pd.MultiIndex):
        close_data = downloaded_df["Close"]
        if isinstance(close_data, pd.DataFrame):
            if asset_ticker and asset_ticker in close_data.columns:
                return close_data[asset_ticker]
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
        rows = []
        for ticker, profile in FALLBACK_COMPANY_PROFILES.items():
            rows.append(
                {
                    "ticker": ticker,
                    "name": profile["name"],
                    "sector": profile["sector"],
                    "industry": profile["industry"],
                }
            )
        return pd.DataFrame(rows)

    rows = []
    for _, row in table.iterrows():
        rows.append(
            {
                "ticker": str(row["Symbol"]).replace(".", "-"),
                "name": row.get("Security"),
                "sector": row.get("GICS Sector"),
                "industry": row.get("GICS Sub-Industry"),
            }
        )
    reference_df = pd.DataFrame(rows)
    for ticker, profile in FALLBACK_COMPANY_PROFILES.items():
        if ticker not in set(reference_df["ticker"]):
            reference_df.loc[len(reference_df)] = {
                "ticker": ticker,
                "name": profile["name"],
                "sector": profile["sector"],
                "industry": profile["industry"],
            }
    return reference_df.drop_duplicates(subset=["ticker"]).sort_values("name").reset_index(drop=True)


@st.cache_data
def get_stock_catalog():
    reference_df = get_sp500_reference().copy()
    reference_df["label"] = reference_df["name"] + " (" + reference_df["ticker"] + ")"
    return reference_df.sort_values("label").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_company_profile(stock_ticker):
    reference_df = get_sp500_reference()
    match = reference_df.loc[reference_df["ticker"] == stock_ticker]
    reference_profile = match.iloc[0].to_dict() if not match.empty else {}

    try:
        ticker_data = YahooQueryTicker(stock_ticker, asynchronous=False, validate=True, progress=False)
        price_info = ticker_data.price.get(stock_ticker, {})
        asset_profile = ticker_data.asset_profile.get(stock_ticker, {})
    except Exception:
        return {
            "name": reference_profile.get("name") or FALLBACK_COMPANY_PROFILES.get(stock_ticker, {}).get("name") or stock_ticker,
            "sector": reference_profile.get("sector") or FALLBACK_COMPANY_PROFILES.get(stock_ticker, {}).get("sector"),
            "industry": reference_profile.get("industry") or FALLBACK_COMPANY_PROFILES.get(stock_ticker, {}).get("industry"),
            "country": reference_profile.get("country") or FALLBACK_COMPANY_PROFILES.get(stock_ticker, {}).get("country"),
        }

    if not isinstance(price_info, dict):
        price_info = {}
    if not isinstance(asset_profile, dict):
        asset_profile = {}

    fallback = reference_profile or FALLBACK_COMPANY_PROFILES.get(stock_ticker, {})
    return {
        "name": price_info.get("longName") or price_info.get("shortName") or fallback.get("name") or stock_ticker,
        "sector": asset_profile.get("sector") or fallback.get("sector"),
        "industry": asset_profile.get("industry") or fallback.get("industry"),
        "country": asset_profile.get("country") or fallback.get("country"),
    }


def initialize_app_state():
    st.session_state.setdefault("analysis_request", None)
    st.session_state.setdefault("analysis_view", "dashboard")


def clean_search_tokens(*values):
    tokens = []
    for value in values:
        if not value:
            continue
        if isinstance(value, (list, tuple, set)):
            tokens.extend(clean_search_tokens(*value))
            continue
        pieces = re.split(r"[^A-Za-z0-9]+", str(value))
        tokens.extend(piece.lower() for piece in pieces if len(piece) >= 3)
    return list(dict.fromkeys(tokens))


def build_relevance_terms(asset_name, asset_ticker, sector=None, industry=None):
    exposures = SECTOR_EXPOSURES.get(sector or "", [])
    company_terms = clean_search_tokens(asset_name, asset_ticker, sector, industry)
    exposure_terms = clean_search_tokens(exposures)
    relevance_terms = list(dict.fromkeys(company_terms + exposure_terms))
    return relevance_terms[:20]


@st.cache_data
def load_asset_vs_benchmark(asset_ticker, benchmark_ticker, start, end):
    raw = yf.download(
        [asset_ticker, benchmark_ticker],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        return None

    asset_close = extract_close(raw, asset_ticker)
    benchmark_close = extract_close(raw, benchmark_ticker)
    if asset_close is None or benchmark_close is None:
        return None

    frame = pd.DataFrame(
        {
            "asset_close": asset_close,
            "benchmark_close": benchmark_close,
        }
    ).dropna()

    if frame.empty:
        return None

    frame["asset_return"] = frame["asset_close"].pct_change()
    frame["benchmark_return"] = frame["benchmark_close"].pct_change()
    return frame.dropna()


def compute_analysis(frame, rolling_window):
    x = sm.add_constant(frame["benchmark_return"])
    y = frame["asset_return"]
    model = sm.OLS(y, x).fit()

    enriched = frame.copy()
    enriched["predicted_return"] = model.predict(x)
    enriched["residual"] = enriched["asset_return"] - enriched["predicted_return"]
    enriched["rolling_iv"] = enriched["residual"].rolling(rolling_window).std() * np.sqrt(TRADING_DAYS)

    metrics = {
        "alpha_daily": model.params["const"],
        "beta": model.params["benchmark_return"],
        "r_squared": model.rsquared,
        "idio_vol_daily": enriched["residual"].std(),
        "idio_vol_annual": enriched["residual"].std() * np.sqrt(TRADING_DAYS),
        "total_vol_annual": enriched["asset_return"].std() * np.sqrt(TRADING_DAYS),
        "annualized_return": enriched["asset_return"].mean() * TRADING_DAYS,
        "rolling_iv_current": enriched["rolling_iv"].dropna().iloc[-1] if not enriched["rolling_iv"].dropna().empty else np.nan,
        "rolling_iv_average": enriched["rolling_iv"].mean(),
        "actual_start": enriched.index.min().date(),
        "actual_end": enriched.index.max().date(),
    }
    return model, enriched, metrics


def compute_ivol_percentile(series):
    clean = series.dropna()
    if clean.empty:
        return np.nan
    current_value = clean.iloc[-1]
    return 100 * (clean <= current_value).mean()


def classify_risk_profile(r_squared, idio_vol_annual, total_vol_annual):
    if total_vol_annual <= 0:
        return "Insufficient Data"
    idio_share = idio_vol_annual / total_vol_annual
    if r_squared >= 0.7 and idio_share <= 0.45:
        return "Mostly Market-Driven"
    if r_squared <= 0.35 or idio_share >= 0.65:
        return "Mostly Firm-Specific"
    return "Mixed Risk Profile"


def build_takeaway(asset_name, risk_profile, metrics, benchmark_label):
    idio_share = metrics["idio_vol_annual"] / metrics["total_vol_annual"] if metrics["total_vol_annual"] > 0 else np.nan
    if risk_profile == "Mostly Market-Driven":
        return (
            f"{asset_name} appears to be primarily driven by broad market conditions relative to {benchmark_label}. "
            f"Most of its return variation is explained by the benchmark, and the firm-specific component looks relatively contained."
        )
    if risk_profile == "Mostly Firm-Specific":
        return (
            f"{asset_name} appears to be strongly influenced by asset-specific events. "
            f"The benchmark explains a smaller share of its movements, so security-level news and shocks likely matter more."
        )
    return (
        f"{asset_name} shows a balanced risk profile. The benchmark is still important, but the idiosyncratic component "
        f"remains meaningful at about {idio_share:.0%} of total annualized risk."
    )


@st.cache_data(ttl=90, show_spinner=False)
def get_market_snapshot(asset_ticker):
    try:
        ticker_data = YahooQueryTicker(asset_ticker, asynchronous=False, validate=True, progress=False)
        price_info = ticker_data.price.get(asset_ticker, {})
    except Exception:
        return {}

    if not isinstance(price_info, dict):
        return {}

    return {
        "price": price_info.get("regularMarketPrice"),
        "change_pct": price_info.get("regularMarketChangePercent"),
        "currency": price_info.get("currency"),
        "market_time": price_info.get("regularMarketTime"),
    }


def fetch_yahoo_headlines(asset_ticker, max_items=5):
    try:
        news_items = yf.Ticker(asset_ticker).news or []
    except Exception:
        return []

    cleaned = []
    for item in news_items[: max_items * 2]:
        content = item.get("content", {})
        title = content.get("title")
        if not title:
            continue
        cleaned.append(
            {
                "title": title,
                "summary": content.get("summary"),
                "source": content.get("provider", {}).get("displayName") or "Yahoo Finance",
                "published": content.get("pubDate"),
                "url": content.get("canonicalUrl", {}).get("url") or content.get("clickThroughUrl", {}).get("url"),
                "source_type": "yahoo",
                "query_category": "company",
            }
        )
        if len(cleaned) >= max_items:
            break
    return cleaned


def article_text(item):
    return " ".join(filter(None, [item.get("title"), item.get("summary"), item.get("source"), item.get("query_category")])).lower()


def deduplicate_articles(items, max_items=18):
    seen = set()
    cleaned = []
    for item in items:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        unique_key = ((item.get("url") or "").strip().lower(), title.lower())
        if unique_key in seen:
            continue
        seen.add(unique_key)
        cleaned.append(item)
        if len(cleaned) >= max_items:
            break
    return cleaned


def classify_article_catalysts(item, sector=None, industry=None):
    text = article_text(item)
    tags = []
    for label, keywords in CATALYST_PATTERNS.items():
        if any(keyword in text for keyword in keywords):
            tags.append(label)

    sector_terms = clean_search_tokens(sector, industry, SECTOR_EXPOSURES.get(sector or "", []))
    if sector_terms and any(term in text for term in sector_terms):
        tags.append("sector")

    if not tags:
        query_category = item.get("query_category")
        if query_category in {"policy", "policy_macro", "policy_cb"}:
            tags.append("policy")
        elif query_category == "macro":
            tags.append("macro")
        else:
            tags.append("sector")
    return list(dict.fromkeys(tags))


def score_article_impact(item, catalysts):
    text = article_text(item)
    positive_hits = sum(keyword in text for keyword in POSITIVE_SIGNAL_WORDS)
    negative_hits = sum(keyword in text for keyword in NEGATIVE_SIGNAL_WORDS)
    score = positive_hits - negative_hits

    if score == 0:
        score = sum(DEFAULT_ARTICLE_BIAS.get(catalyst, 0.0) for catalyst in catalysts)

    if "management" in catalysts and "resign" in text:
        score -= 1
    if "management" in catalysts and ("appoint" in text or "hire" in text):
        score += 1
    if "policy" in catalysts and ("approval" in text or "subsidy" in text):
        score += 1
    if "policy" in catalysts and ("ban" in text or "sanction" in text or "tariff" in text):
        score -= 1
    return float(np.clip(score, -3, 3))


def classify_catalysts(news_context, sector=None, industry=None):
    scorecard = {
        label: {"net_score": 0.0, "positive": [], "negative": [], "neutral": [], "count": 0}
        for label in CATALYST_PATTERNS
    }
    classified_articles = []

    for item in news_context.get("articles", []):
        catalysts = classify_article_catalysts(item, sector=sector, industry=industry)
        impact = score_article_impact(item, catalysts)
        source_weight = SOURCE_WEIGHTS.get(item.get("source_type"), 1.0)
        weighted_impact = impact * source_weight
        enriched_item = item.copy()
        enriched_item["catalysts"] = catalysts
        enriched_item["impact_score"] = weighted_impact
        classified_articles.append(enriched_item)

        bucket_key = "positive" if weighted_impact > 0.25 else "negative" if weighted_impact < -0.25 else "neutral"
        for catalyst in catalysts:
            bucket = scorecard.setdefault(catalyst, {"net_score": 0.0, "positive": [], "negative": [], "neutral": [], "count": 0})
            bucket["net_score"] += weighted_impact
            bucket["count"] += 1
            bucket[bucket_key].append(enriched_item)

    return {"articles": classified_articles, "scorecard": scorecard}


def build_quant_signal_components(metrics, risk_profile, market_snapshot):
    components = []
    annualized_return = metrics["annualized_return"]
    if annualized_return >= 0.18:
        components.append(("Return trend remains strong over the selected sample.", 2.0))
    elif annualized_return > 0:
        components.append(("Returns are positive, but not decisively strong.", 1.0))
    elif annualized_return <= -0.18:
        components.append(("Returns have been weak over the selected sample.", -2.0))
    else:
        components.append(("Returns have been slightly negative.", -1.0))

    if pd.notna(metrics["rolling_iv_current"]) and pd.notna(metrics["rolling_iv_average"]) and metrics["rolling_iv_average"] > 0:
        iv_ratio = metrics["rolling_iv_current"] / metrics["rolling_iv_average"]
        if iv_ratio <= 0.9:
            components.append(("Current firm-specific volatility is running below its own average.", 1.5))
        elif iv_ratio >= 1.1:
            components.append(("Current firm-specific volatility is elevated versus its own history.", -1.5))

    if risk_profile == "Mostly Market-Driven":
        components.append(("Risk is more benchmark-driven than firm-specific, which lowers single-name shock dependence.", 1.0))
    elif risk_profile == "Mostly Firm-Specific":
        components.append(("Risk is dominated by firm-specific factors, which raises execution and news sensitivity.", -1.0))

    change_pct = market_snapshot.get("change_pct") if market_snapshot else None
    if isinstance(change_pct, (int, float, np.floating)):
        if change_pct >= 0.02:
            components.append(("The latest market snapshot is supportive.", 0.5))
        elif change_pct <= -0.02:
            components.append(("The latest market snapshot is weak.", -0.5))

    return components


def summarize_catalyst_reasons(scorecard, polarity="positive", max_items=3):
    direction = 1 if polarity == "positive" else -1
    rows = []
    for catalyst, details in scorecard.items():
        score = details.get("net_score", 0.0)
        if direction * score <= 0.35:
            continue
        examples = details["positive"] if polarity == "positive" else details["negative"]
        if not examples:
            continue
        top_example = sorted(examples, key=lambda item: abs(item.get("impact_score", 0.0)), reverse=True)[0]
        headline = top_example.get("title") or catalyst.title()
        rows.append((abs(score), f"{catalyst.upper()} context skews {polarity}, led by: {headline}"))
    rows.sort(reverse=True)
    return [text for _, text in rows[:max_items]]


def group_articles_by_catalyst(classified_context):
    grouped = {label: [] for label in CATALYST_PATTERNS}
    for item in classified_context.get("articles", []):
        for catalyst in item.get("catalysts", []):
            grouped.setdefault(catalyst, []).append(item)
    return {key: value for key, value in grouped.items() if value}


def build_prediction_signal(
    asset_name,
    metrics,
    risk_profile,
    market_snapshot,
    news_context,
    classified_context,
    *,
    news_coverage_label="multi-source public news coverage",
):
    quant_components = build_quant_signal_components(metrics, risk_profile, market_snapshot)
    quant_score = sum(score for _, score in quant_components)
    news_score = sum(item.get("impact_score", 0.0) for item in classified_context.get("articles", []))
    total_score = quant_score + (news_score / 2.5)

    if total_score >= 4:
        action = "Buy"
    elif total_score >= 1:
        action = "Hold"
    elif total_score <= -4:
        action = "Avoid"
    else:
        action = "Sell"

    evidence_count = len(classified_context.get("articles", []))
    if abs(total_score) >= 4 and evidence_count >= 6:
        confidence = "High"
    elif abs(total_score) >= 2 and evidence_count >= 3:
        confidence = "Medium"
    else:
        confidence = "Low"

    positive_quant = [text for text, score in quant_components if score > 0][:3]
    negative_quant = [text for text, score in quant_components if score < 0][:3]
    positive_catalysts = summarize_catalyst_reasons(classified_context["scorecard"], "positive")
    negative_catalysts = summarize_catalyst_reasons(classified_context["scorecard"], "negative")

    why_buy = (positive_quant + positive_catalysts)[:5]
    why_sell = (negative_quant + negative_catalysts)[:5]
    if not why_buy:
        why_buy = ["The quantitative data is not strongly bullish, so upside support is limited."]
    if not why_sell:
        why_sell = ["Recent news flow is not strongly negative, but conviction is still limited."]

    risk_text = (
        f"Downside risk remains tied to annualized total volatility near {metrics['total_vol_annual']:.2%} "
        f"and idiosyncratic volatility near {metrics['idio_vol_annual']:.2%}. "
        f"{'Firm-specific headlines could move the asset sharply.' if risk_profile == 'Mostly Firm-Specific' else 'Broad market moves remain an important driver.'}"
    )
    upside_text = (
        f"Upside potential is linked to the recent return trend of {metrics['annualized_return']:.2%} and "
        f"{'supportive catalyst flow.' if positive_catalysts else 'a stabilization in volatility and benchmark conditions.'}"
    )

    summary = (
        f"{asset_name} currently leans toward **{action}** with **{confidence.lower()} confidence**. "
        f"The model blends return trend, rolling IVOL behavior, risk profile, and {news_coverage_label}."
    )

    return {
        "action": action,
        "confidence": confidence,
        "summary": summary,
        "why_buy": why_buy,
        "why_sell": why_sell,
        "risk_text": risk_text,
        "upside_text": upside_text,
        "quant_score": quant_score,
        "news_score": news_score,
        "total_score": total_score,
        "grouped_articles": group_articles_by_catalyst(classified_context),
        "coverage_notes": news_context.get("coverage_notes", []),
        "source_counts": news_context.get("source_counts", {}),
        "headquarters_country": news_context.get("country"),
    }


def format_market_caption(snapshot):
    if not snapshot:
        return "Live market snapshot unavailable."
    price = snapshot.get("price")
    change_pct = snapshot.get("change_pct")
    price_text = f"{price:.2f}" if isinstance(price, (int, float, np.floating)) else "N/A"
    change_text = f"{change_pct:.2%}" if isinstance(change_pct, (int, float, np.floating)) else "N/A"
    return f"Latest price: {price_text} | Latest change: {change_text}"


def format_profile_line(profile):
    parts = []
    if profile.get("sector"):
        parts.append(f"**Sector:** {profile['sector']}")
    if profile.get("industry"):
        parts.append(f"**Industry:** {profile['industry']}")
    if profile.get("country"):
        parts.append(f"**Headquarters (Yahoo):** {profile['country']}")
    return " | ".join(parts)


def render_date_messages(metrics, selected_start, selected_end):
    if metrics["actual_start"] > selected_start:
        st.info(
            f"Market data did not start exactly on {selected_start}. The analysis begins on the next available trading day: {metrics['actual_start']}."
        )
    if metrics["actual_end"] < selected_end:
        st.info(
            f"Market data did not extend exactly to {selected_end}. The analysis ends on the most recent available trading day: {metrics['actual_end']}."
        )


@st.cache_data
def download_close_batches(tickers, start, end, batch_size=80):
    close_frames = []
    unique_tickers = list(dict.fromkeys(tickers))
    for idx in range(0, len(unique_tickers), batch_size):
        batch = unique_tickers[idx : idx + batch_size]
        batch_data = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False, group_by="column", threads=True)
        if batch_data.empty:
            continue
        close = batch_data["Close"] if isinstance(batch_data.columns, pd.MultiIndex) else batch_data[["Close"]]
        if isinstance(close, pd.Series):
            close = close.to_frame(name=batch[0])
        elif not isinstance(close.columns, pd.Index):
            close = pd.DataFrame(close)
        close_frames.append(close)

    if not close_frames:
        return pd.DataFrame()

    close_df = pd.concat(close_frames, axis=1)
    close_df = close_df.loc[:, ~close_df.columns.duplicated()]
    return close_df.sort_index()


@st.cache_data
def build_sp500_screener(start, end, rolling_window):
    catalog = get_stock_catalog()
    all_tickers = catalog["ticker"].tolist()
    close_df = download_close_batches(all_tickers + [STOCK_MARKET_BENCHMARK], start, end)
    if close_df.empty or STOCK_MARKET_BENCHMARK not in close_df.columns:
        return pd.DataFrame()

    benchmark_close = close_df[STOCK_MARKET_BENCHMARK]
    rows = []
    for row in catalog.itertuples(index=False):
        if row.ticker not in close_df.columns:
            continue

        frame = pd.DataFrame(
            {
                "asset_close": close_df[row.ticker],
                "benchmark_close": benchmark_close,
            }
        ).dropna()
        if len(frame) < 80:
            continue
        frame["asset_return"] = frame["asset_close"].pct_change()
        frame["benchmark_return"] = frame["benchmark_close"].pct_change()
        frame = frame.dropna()
        if len(frame) < 30:
            continue

        _, enriched, metrics = compute_analysis(frame, rolling_window)
        percentile = compute_ivol_percentile(enriched["rolling_iv"])
        risk_profile = classify_risk_profile(metrics["r_squared"], metrics["idio_vol_annual"], metrics["total_vol_annual"])
        rows.append(
            {
                "Ticker": row.ticker,
                "Company": row.name,
                "Sector": row.sector,
                "Industry": row.industry,
                "Beta": metrics["beta"],
                "R-squared": metrics["r_squared"],
                "Annualized Return": metrics["annualized_return"],
                "Annualized Total Volatility": metrics["total_vol_annual"],
                "Annualized Idiosyncratic Volatility": metrics["idio_vol_annual"],
                "IVOL Percentile": percentile,
                "Risk Profile": risk_profile,
            }
        )

    screener_df = pd.DataFrame(rows)
    if screener_df.empty:
        return screener_df
    return screener_df.sort_values("Annualized Idiosyncratic Volatility", ascending=False).reset_index(drop=True)


# --- Streamlit UI (core) ---

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


st.set_page_config(page_title="Bob Cat Risk Lab (Core)", layout="wide")

st.markdown("### Group: Bob Cat")
st.title("Bob Cat Risk Lab — Core (original scope)")
st.caption(
    "**Core build** — **Yahoo Finance headlines only.** "
    "For multi-source news, policy/central-bank context, prediction details, and forward sense-check, run **`streamlit run app_extended.py`**."
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

st.caption("Core app: CAPM residual IVOL methodology matches **`app_extended.py`**; news inputs differ.")
