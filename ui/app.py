"""
NIFTY SENTINEL - Simplified, Ultra-Clean F&O Trading & Risk Terminal
Designed for maximum clarity, intuitive decision making, and ease of use.
"""

import os
import sys
from pathlib import Path
import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Core imports
from core.models import OptionType, TradeAction, StrategyLeg
from strategy_optimizer.payoff_calculator import PayoffCalculator
from core.option_evaluator import OptionEvaluator
from core.portfolio import (PortfolioManager, greeks_in_words, position_alert, DEFAULT_STOP_LOSS_PCT,
                            DEFAULT_TARGET_PCT)
from typing import Optional

from data_feed.live_feed import LiveDataFeed, previous_session_close
from risk_engine.expected_move import expected_move
from ui.chart_helpers import market_closed_rangebreaks
from ml_engine.evaluation import evaluate_models, evaluate_real, MODEL_LABELS, FEATURES as FEATURE_NAMES
from ml_engine.real_data import build_option_trades, load_history
from ml_engine.explain import explain_forest_prediction
from backtesting.threshold import threshold_backtest
from data_feed.market_source import SNAPSHOT_DIR, Source, VixQuote, get_nifty, get_vix, load_snapshot_meta
from trade_analyzer.parser import TradebookParser
from trade_analyzer.metrics import PerformanceMetrics
from trade_analyzer.leakage_detector import LeakageDetector
from trade_analyzer.counterfactual import CounterfactualSimulator
from risk_engine.monte_carlo import MonteCarloEngine
from risk_engine.command_center import build_risk_snapshot, compute_max_pain, DEFAULT_STOP_LOSS_POINTS
from core.expiry import upcoming_expiries
from risk_engine.containment import DrawdownContainmentEngine
from risk_engine.trade_budget import check_trade_budget
from ml_engine.classifier import get_predictor
from trade_analyzer.upload import (TradebookLoad, load_tradebook_bytes, SUPPORTED_BROKERS,
                                   REQUIRED_COLUMNS_HINT)


# Page Setup
st.set_page_config(
    page_title="Predictive Drawdown Containment System",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="collapsed",  # Keep collapsed by default for clean look
)

# Initialize Session State
# Positions are saved to a JSON file; SENTINEL_PORTFOLIO_PATH overrides it (tests use a temp file).
PORTFOLIO_PATH = Path(os.environ.get("SENTINEL_PORTFOLIO_PATH", PROJECT_ROOT / "data" / "portfolio.json"))
if "portfolio_manager" not in st.session_state:
    first_run = not PORTFOLIO_PATH.exists()
    pm = PortfolioManager(storage_path=PORTFOLIO_PATH)
    if first_run:
        pm.add_position(strike=24850.0, option_type=OptionType.CALL, action=TradeAction.BUY, quantity=50,
                        entry_price=110.0, symbol="NIFTY 24850 CE")
    st.session_state.portfolio_manager = pm

if "selected_strike" not in st.session_state:
    st.session_state.selected_strike = 24850

if "selected_type" not in st.session_state:
    st.session_state.selected_type = "CALL (CE)"

if "selected_action" not in st.session_state:
    st.session_state.selected_action = "BUY (Long)"

# Modern, Clean, Minimalist CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    /* ---- Charcoal and gold palette (option F) ----
       90%: charcoal #121212 (page) + warm white #EDEAE2 (text, hero cards); card #1E1D1B · line #2E2C29
       10%: gold #D4AF37 accent (logo dot, active tab, hover, Live badge, hero chart lines)
       meaning colours (not part of the 3): profit #5FD39A · loss #FF8A73 · caution #F0913A
       muted text #A39E93 · on-cream profit #2E6B1E / loss #B4442C */

    html, body, [class*="css"], .stApp {
        background-color: #121212 !important;
        color: #EDEAE2 !important;
        font-family: 'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-feature-settings: "tnum" 1;
    }
    [data-testid="stHeader"] { background: transparent !important; }

    h1, h2, h3, h4, h5, h6 {
        color: #EDEAE2 !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }

    .mono-num { font-feature-settings: "tnum" 1; }

    /* Cards: soft green surfaces, generous radius, no heavy shadows */
    .clean-card {
        background: #1E1D1B !important;
        border: 1px solid rgba(237, 234, 226, 0.08) !important;
        border-radius: 20px !important;
        padding: 20px 24px !important;
        margin-bottom: 16px !important;
    }
    /* Hero card: cream on green, the signature Velto move */
    .hero-card {
        background: #EDEAE2 !important;
        color: #121212 !important;
        border: none !important;
    }
    .hero-card .clean-metric-title { color: #5C5850 !important; }
    .hero-card .clean-metric-val { color: #121212 !important; }
    .hero-card .val-green { color: #2E6B1E !important; }
    .hero-card .val-red, .hero-card .val-coral { color: #B4442C !important; }
    .hero-card .hero-sub { color: #5C5850 !important; }

    .clean-metric-title {
        font-size: 0.78rem !important;
        color: #A39E93 !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        margin-bottom: 6px !important;
    }

    .nowrap-num { white-space: nowrap !important; }
    .pos-head { font-size: 0.78rem; font-weight: 700; color: #A39E93; white-space: nowrap;
                letter-spacing: 0.02em; }

    .clean-metric-val {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #EDEAE2 !important;
        letter-spacing: -0.02em !important;
    }

    .clean-metric-val.metric-sm { font-size: 1.45rem !important; }
    .val-green { color: #5FD39A !important; }
    .val-red { color: #FF8A73 !important; }
    .val-coral { color: #FF8A73 !important; }
    .val-cream { color: #EDEAE2 !important; }
    .val-amber { color: #F0913A !important; }

    /* Verdict banners */
    .verdict-box-optimal, .verdict-box-speculative, .verdict-box-hazardous {
        border-radius: 20px !important;
        padding: 18px 24px !important;
        margin-bottom: 18px !important;
    }
    .verdict-box-optimal { background: rgba(95, 211, 154, 0.10) !important; border: 1.5px solid #5FD39A !important; }
    .verdict-box-speculative { background: rgba(240, 145, 58, 0.10) !important; border: 1.5px solid #F0913A !important; }
    .verdict-box-hazardous { background: rgba(255, 138, 115, 0.10) !important; border: 1.5px solid #FF8A73 !important; }

    /* Pill badges */
    .pill-badge {
        display: inline-block !important;
        padding: 5px 12px !important;
        border-radius: 999px !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em !important;
    }
    .pill-optimal { background: #5FD39A !important; color: #121212 !important; }
    .pill-speculative { background: #F0913A !important; color: #121212 !important; }
    .pill-hazardous { background: #FF8A73 !important; color: #121212 !important; }

    /* Tabs as pills; the active one is lime */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        border-bottom: none !important;
        padding-bottom: 10px !important;
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E1D1B !important;
        border-radius: 999px !important;
        color: #A39E93 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 8px 18px !important;
        transition: background-color 0.2s ease, color 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #D4AF37 !important;
        color: #121212 !important;
    }

    /* Buttons: cream outline pills, lime on hover */
    .stButton > button, .stDownloadButton > button {
        background: transparent !important;
        color: #EDEAE2 !important;
        border: 1px solid rgba(237, 234, 226, 0.35) !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #D4AF37 !important;
        border-color: #D4AF37 !important;
        color: #121212 !important;
    }

    .stSelectbox > div > div, .stNumberInput input, .stTextInput input {
        background-color: #1E1D1B !important;
        color: #EDEAE2 !important;
        border-color: rgba(237, 234, 226, 0.14) !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# ---- Market data: live feed, or the saved snapshot in Demo mode ----
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False
demo_mode = st.session_state.demo_mode


@st.cache_data(ttl=300, show_spinner=False)
def _cached_vix(demo: bool) -> VixQuote:
    return get_vix(demo_mode=demo)


SOURCE_BADGE = {
    # Solid fills with deep-green text: readable on both the cream and the dark-green cards.
    Source.LIVE: ("● Live", "#121212", "#D4AF37"),
    Source.CACHED: ("● Cached", "#121212", "#F0913A"),
    Source.SIMULATED: ("● Simulated", "#121212", "#FF8A73"),
}


def source_badge(source: Source, as_of: Optional[str] = None) -> str:
    """Small pill saying where a number came from (LIVE / CACHED / SIMULATED)."""
    label, fg, bg = SOURCE_BADGE[source]
    tip = f"Saved {as_of}" if source == Source.CACHED and as_of else (
        "No feed and no snapshot: random demo data" if source == Source.SIMULATED else "Yahoo Finance")
    return (f"<span title='{tip}' style='background:{bg}; color:{fg}; padding:1px 8px; border-radius:999px; "
            f"font-size:0.7rem; font-weight:700; white-space:nowrap;'>{label}</span>")


nifty_quote = get_nifty("5m", "5d", demo_mode=demo_mode)
vix_quote = _cached_vix(demo_mode)
live_spot, spot_change_pct, nifty_candles_df = nifty_quote.spot, nifty_quote.change_pct, nifty_quote.candles
live_vix = vix_quote.value

# Top Header: Live Nifty Status & Simple Market Sentiment
h_col1, h_col2, h_col3 = st.columns([3, 1, 1])

with h_col1:
    st.markdown(f"""
    <div style="padding-top: 4px;">
        <div style="font-size: 2.1rem; font-weight: 800; letter-spacing: -0.04em; color: #EDEAE2; line-height: 1;">
            sentinel<span style="color: #D4AF37;">.</span>
        </div>
        <div style="font-size: 0.95rem; color: #A39E93; margin-top: 6px;">Predictive Drawdown Containment System · Nifty 50 F&amp;O</div>
    </div>
    """, unsafe_allow_html=True)
    st.toggle("Demo mode (offline)", key="demo_mode",
              help="Use the market snapshot saved on this computer instead of the internet. "
                   "Refresh it with: python scripts/save_market_snapshot.py")
    if demo_mode:
        _meta = load_snapshot_meta()
        st.caption(f"Showing market data saved {_meta['saved_at']}." if _meta
                   else "No snapshot saved yet, so prices are simulated. Run scripts/save_market_snapshot.py.")
    elif nifty_quote.source != Source.LIVE:
        st.caption("Live feed unavailable, so showing the saved snapshot." if nifty_quote.source == Source.CACHED
                   else "Live feed unavailable and no snapshot saved, so prices are simulated.")

with h_col2:
    trend_tag = "BULLISH" if spot_change_pct >= 0 else "BEARISH"
    trend_color = "#2E6B1E" if spot_change_pct >= 0 else "#B4442C"
    st.markdown(f"""
    <div style="background: #EDEAE2; border-radius: 18px; padding: 10px 16px;">
        <div style="font-size: 0.75rem; font-weight: 600; color: #5C5850;">Nifty 50 {source_badge(nifty_quote.source, nifty_quote.as_of)}</div>
        <div style="font-size: 1.35rem; font-weight: 800; color: #121212; letter-spacing: -0.02em; white-space: nowrap;">
            {live_spot:,.2f}
        </div>
        <div style="font-size: 0.85rem; font-weight: 700; color: {trend_color};">{spot_change_pct:+.2f}% today</div>
    </div>
    """, unsafe_allow_html=True)

with h_col3:
    st.markdown(f"""
    <div style="background: #1E1D1B; border-radius: 18px; padding: 10px 16px;">
        <div style="font-size: 0.75rem; font-weight: 600; color: #A39E93;">India VIX {source_badge(vix_quote.source, vix_quote.as_of)}</div>
        <div style="font-size: 1.35rem; font-weight: 800; color: #EDEAE2; letter-spacing: -0.02em; white-space: nowrap;">
            {live_vix:.2f}
        </div>
        <div style="font-size: 0.85rem; font-weight: 700; color: #A39E93;">{'Normal' if live_vix < 16 else 'High'} volatility</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ---- Tradebook source: the bundled sample, or the user's own upload ----
SAMPLE_TRADEBOOK = PROJECT_ROOT / "data" / "sample_zerodha_trades.csv"


@st.cache_data(show_spinner=False)
def _load_tradebook(data: bytes, name: str) -> TradebookLoad:
    return load_tradebook_bytes(data, name)


_upload = st.session_state.get("tradebook_upload")
_sample_load = _load_tradebook(SAMPLE_TRADEBOOK.read_bytes(), SAMPLE_TRADEBOOK.name)
_user_load = _load_tradebook(_upload.getvalue(), _upload.name) if _upload is not None else None
upload_failed = _user_load is not None and _user_load.error is not None
using_sample = _user_load is None or upload_failed
active_load = _sample_load if using_sample else _user_load
active_trades = active_load.trades if active_load.trades is not None else pd.DataFrame()
active_tradebook_name = "sample tradebook" if using_sample else active_load.source_name

_n_trades = len(active_trades)
_span = ""
if _n_trades:
    _days = pd.to_datetime(active_trades["entry_time"])
    _span = f" · {_days.min():%d %b %Y} to {_days.max():%d %b %Y}"
with st.expander(f"Tradebook: {active_tradebook_name} · {_n_trades} trades{_span}", expanded=_upload is not None):
    st.file_uploader(
        "Upload your broker tradebook to analyse your own trades",
        type=["csv", "xlsx", "xls"], key="tradebook_upload",
        help=f"Works with {SUPPORTED_BROKERS}. Needs {REQUIRED_COLUMNS_HINT}.",
    )
    st.caption(f"Works with {SUPPORTED_BROKERS}. The file is only used in this session and isn't saved. "
               "Remove the file to go back to the sample.")
    if upload_failed:
        st.error(f"{_user_load.error} Showing the sample tradebook instead.")
    elif not using_sample:
        st.success(f"Loaded {_n_trades} completed trades from {active_tradebook_name}. "
                   "The Risk Command Center and Mistakes Detected sections now use your trades.")


# Navigation: 6 tabs
tab1, tab_risk, tab2, tab3, tab4, tab5 = st.tabs([
    "Nifty 50 Price Chart",
    "Risk Command Center",
    "Option Safety & Chain",
    "Trade Tester",
    "Active Portfolio",
    "ML & Backtest Analysis"
])


# ==============================================================================
# TAB 1: NIFTY 50 LIVE CANDLESTICK CHART
# ==============================================================================
with tab1:
    st.markdown("""<style>
        .stPills { display: flex; justify-content: center; margin-top: 6px; margin-bottom: 12px; }
        .stPills button { font-weight: 600; font-size: 0.85rem; }
    </style>""", unsafe_allow_html=True)

    @st.fragment(run_every="2s")
    def render_nifty_candlestick_chart():
        if 'tf_choice' not in st.session_state:
            st.session_state.tf_choice = "1D"
        if 'chart_type_choice' not in st.session_state:
            st.session_state.chart_type_choice = "Candlestick"
            
        tf_map = {
            "1D": ("5m", "1d"), 
            "1W": ("15m", "5d"), 
            "1M": ("1d", "1mo"), 
            "3M": ("1d", "3mo"), 
            "6M": ("1d", "6mo"),
            "1Y": ("1d", "1y"),
            "5Y": ("1wk", "5y"),
            "All": ("1mo", "max")
        }
        curr_int, curr_per = tf_map.get(st.session_state.tf_choice, ("5m", "1d"))

        chart_quote = get_nifty(curr_int, curr_per, demo_mode=demo_mode)
        live_spot_val, spot_chg_pct, candles = chart_quote.spot, chart_quote.change_pct, chart_quote.candles
        
        # Calculate Baseline & metrics
        if not candles.empty:
            baseline = candles["Open"].iloc[0]
            current_close = candles["Close"].iloc[-1]
            high_val = candles["High"].max()
            low_val = candles["Low"].min()
            diff = current_close - baseline
            diff_pct = (diff / baseline) * 100
        else:
            baseline = live_spot_val
            current_close = live_spot_val
            high_val = live_spot_val
            low_val = live_spot_val
            diff = 0
            diff_pct = 0
            
        is_bullish = diff >= 0
        trend_accent = "#5FD39A" if is_bullish else "#FF8A73"
        sign = "+" if is_bullish else ""

        # Top Header Card matching professional trading terminals
        st.markdown(f"""
        <div style="background: linear-gradient(180deg, #1E1D1B 0%, #121212 100%); padding: 18px 24px 14px 24px; border-radius: 14px 14px 0 0; border: 1px solid rgba(237, 234, 226, 0.12); border-bottom: none;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.3rem; font-weight: 700; color: #EDEAE2; letter-spacing: 0.5px;">NIFTY 50 INDEX</span>
                        {source_badge(chart_quote.source, chart_quote.as_of)}
                    </div>
                    <div style="display: flex; align-items: baseline; gap: 14px; margin-top: 6px;">
                        <span style="font-size: 2.2rem; font-weight: 800; color: #EDEAE2; font-family: 'Manrope', sans-serif; font-variant-numeric: tabular-nums;">{current_close:,.2f}</span>
                        <span style="font-size: 1.05rem; font-weight: 700; color: {trend_accent};">
                            {sign}{diff:,.2f} ({sign}{diff_pct:.2f}%) <span style="font-size: 0.8rem; font-weight: 600; color: #A39E93;">over {st.session_state.tf_choice}</span>
                        </span>
                        <span style="color: #A39E93; font-size: 0.85rem; font-weight: 500;">Range: {low_val:,.1f} - {high_val:,.1f}</span>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 16px;">
                    <div style="text-align: right; border-right: 1px solid rgba(237, 234, 226, 0.12); padding-right: 16px;">
                        <div style="font-size: 0.72rem; color: #A39E93; text-transform: uppercase;">Timeframe</div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #B9B2A3;">{st.session_state.tf_choice} ({curr_int})</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.72rem; color: #A39E93; text-transform: uppercase;">Trend Bias</div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: {'#5FD39A' if is_bullish else '#FF8A73'};">{'Bullish' if is_bullish else 'Bearish'}</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()

        # Candlestick with lime/emerald up and yellow/red down gradient styling
        fig.add_trace(go.Candlestick(
            x=candles["time"],
            open=candles["Open"],
            high=candles["High"],
            low=candles["Low"],
            close=candles["Close"],
            increasing=dict(
                line=dict(color="#5FD39A", width=1.5),
                fillcolor="#5FD39A"
            ),
            decreasing=dict(
                line=dict(color="#FF8A73", width=1.5),
                fillcolor="#FF8A73"
            ),
            name="Nifty 50 OHLC",
            hoverlabel=dict(bgcolor="#1E1D1B", font=dict(color="#EDEAE2", family="Manrope, sans-serif"))
        ))

        # 20 EMA Overlay (Trend line)
        if "EMA20" in candles.columns:
            fig.add_trace(go.Scatter(
                x=candles["time"],
                y=candles["EMA20"],
                mode="lines",
                name="EMA 20",
                line=dict(color="#B9B2A3", width=1.5, dash="solid"),
                hoverinfo="y"
            ))

        # 50 EMA Overlay (Medium term trend)
        if "EMA50" in candles.columns:
            fig.add_trace(go.Scatter(
                x=candles["time"],
                y=candles["EMA50"],
                mode="lines",
                name="EMA 50",
                line=dict(color="#F0913A", width=1.3, dash="dot"),
                hoverinfo="y"
            ))

        # Baseline Reference Line
        if not candles.empty:
            fig.add_shape(
                type="line",
                x0=candles["time"].iloc[0],
                y0=baseline,
                x1=candles["time"].iloc[-1],
                y1=baseline,
                line=dict(color="#6B665D", width=1.2, dash="dot")
            )

        # Expected-move band: today's 68% range implied by India VIX, centred on yesterday's close.
        em = expected_move(previous_session_close(nifty_candles_df), live_vix)
        is_intraday = curr_int in ("5m", "15m") and not candles.empty
        if is_intraday:
            session_day = pd.to_datetime(candles["time"]).dt.normalize().iloc[-1]
            x0 = candles.loc[pd.to_datetime(candles["time"]).dt.normalize() == session_day, "time"].iloc[0]
            x1 = max(candles["time"].iloc[-1], session_day + pd.Timedelta(hours=15, minutes=30))
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=em.low, y1=em.high, xref="x", yref="y",
                          fillcolor="rgba(212, 175, 55, 0.10)", line=dict(width=0), layer="below")
            for level, label in ((em.high, "Expected high"), (em.low, "Expected low")):
                fig.add_shape(type="line", x0=x0, x1=x1, y0=level, y1=level, xref="x", yref="y",
                              line=dict(color="#D4AF37", width=1.2, dash="dash"))
                fig.add_annotation(x=x1, y=level, xref="x", yref="y", text=f"{label} {level:,.0f}",
                                   showarrow=False, xanchor="right", yanchor="bottom",
                                   font=dict(color="#D4AF37", size=11))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#121212",
            plot_bgcolor="#121212",
            height=400,
            margin=dict(l=10, r=55, t=10, b=10),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            xaxis=dict(
                showgrid=True,
                gridcolor="#1E1D1B",
                zeroline=False,
                showticklabels=True,
                tickfont=dict(color="#A39E93", size=11),
                # Hide nights, weekends and holidays so candles sit side by side.
                rangebreaks=market_closed_rangebreaks(candles["time"], intraday=curr_int in ("5m", "15m"),
                                                      daily=curr_int == "1d") if not candles.empty else [],
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#1E1D1B",
                zeroline=False,
                side="right",
                tickfont=dict(color="#A39E93", size=11)
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11, color="#A39E93")
            )
        )

        em_inside = em.contains(live_spot)
        em_status = ("Nifty is inside it" if em_inside
                     else f"Nifty is outside it: an unusually large move ({'above' if live_spot > em.high else 'below'})")
        em_where = "" if is_intraday else " Switch to 1D or 1W to see it on the chart."
        st.markdown(f"""
        <div style="border: 1px solid rgba(237, 234, 226, 0.12); border-top: none; background: #1E1D1B; padding: 10px 24px;
                    display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: baseline;">
            <span style="color: #D4AF37; font-weight: 700; font-size: 0.85rem;">Expected range today</span>
            <span style="color: #EDEAE2; font-weight: 800; font-size: 1.05rem; white-space: nowrap;">{em.low:,.0f} – {em.high:,.0f}</span>
            <span style="color: #A39E93; font-size: 0.85rem;">{em.probability_pct:.0f}% chance, ±{em.move_pct:.2f}% (±{em.move_points:,.0f} pts) from VIX {live_vix:.1f} around yesterday's close {em.reference_price:,.0f}. {em_status}.{em_where}</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='border: 1px solid rgba(237, 234, 226, 0.12); border-top: none; background: #121212; padding: 4px 6px;'>", unsafe_allow_html=True)
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        # Bottom Bar: Timeframe selector pills
        st.markdown("<div style='border: 1px solid rgba(237, 234, 226, 0.12); border-top: none; background: #1E1D1B; padding: 10px 16px 14px 16px; border-radius: 0 0 14px 14px;'>", unsafe_allow_html=True)
        tf_opts = ["1D", "1W", "1M", "3M", "6M", "1Y", "5Y", "All"]
        selected_tf = st.pills("Select Timeframe", tf_opts, default=st.session_state.tf_choice, label_visibility="collapsed")
        if selected_tf and selected_tf != st.session_state.tf_choice:
            st.session_state.tf_choice = selected_tf
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.caption("**Quick trend guide**: when Nifty trades **above the silver EMA 20 line**, momentum is bullish and calls carry higher odds; below it, puts are favoured. The **gold band** is the range India VIX implies for today: about two days in three, Nifty closes inside it.")

    render_nifty_candlestick_chart()

# ==============================================================================
# TAB: RISK COMMAND CENTER (regime, strategy, VaR, Monte Carlo, breaker, sizing)
# ==============================================================================
RISK_CAPITAL_INR = 200000.0
BREAKER_COLOURS = {"green": "#5FD39A", "amber": "#F0913A", "red": "#FF8A73"}


_VALUE_CLASS = {"#FF8A73": "val-red", "#5FD39A": "val-green", "#F0913A": "val-amber"}


def _risk_card(title: str, value: str, sub: str, value_colour: str = "#EDEAE2") -> str:
    return f"""
    <div class="clean-card" style="min-height: 150px;">
        <div class="clean-metric-title">{title}</div>
        <div class="clean-metric-val nowrap-num {_VALUE_CLASS.get(value_colour, '')}">{value}</div>
        <div style="font-size: 0.82rem; color: #A39E93;">{sub}</div>
    </div>"""


def _row(label: str, value: str) -> str:
    return (f"<div style='display:flex; justify-content:space-between; gap:12px;'>"
            f"<span style='color:#A39E93;'>{label}</span><b>{value}</b></div>")


with tab_risk:
    st.subheader("Risk Command Center")
    st.write("Every risk engine in one place: today's market regime, the strategy it calls for, "
             "how much you could lose, and whether you should be trading at all.")

    risk_trades = active_trades
    snap = build_risk_snapshot(
        spot=live_spot, vix=live_vix, candles=nifty_candles_df,
        chain=LiveDataFeed.generate_live_option_chain(live_spot, live_vix),
        trades_df=risk_trades, capital=RISK_CAPITAL_INR, vix_history=vix_quote.history,
    )
    brk, reg, mc, inp = snap["breaker"], snap["regime"], snap["monte_carlo"], snap["inputs"]
    var95 = snap["var"]["var_metrics"]["95%"]

    # --- Row 1: four headline cards ---
    k1, k2, k3, k4 = st.columns(4)
    light = brk["light"]
    with k1:
        st.markdown(f"""
        <div class="clean-card hero-card" style="min-height: 150px;">
            <div class="clean-metric-title">Circuit breaker</div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="width: 18px; height: 18px; border-radius: 50%; background: {BREAKER_COLOURS[light]};
                             border: 2px solid #121212; display: inline-block;"></span>
                <span class="clean-metric-val">{light.upper()}</span>
            </div>
            <div class="hero-sub" style="font-size: 0.82rem;">{brk['action_required'].capitalize()}</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(_risk_card(
            "Market regime", reg["trend"].replace("_", " ").title(),
            f"{reg['volatility_regime'].replace('_', ' ').title().replace('Iv', 'IV')} · bias {reg['directional_bias'].lower()}"),
            unsafe_allow_html=True)
    with k3:
        st.markdown(_risk_card(
            "1-day VaR (95%)", f"₹{var95['daily_historical_var_inr']:,.0f}",
            f"Expected shortfall ₹{var95['daily_cvar_expected_shortfall_inr']:,.0f} on a bad day", "#FF8A73"),
            unsafe_allow_html=True)
    with k4:
        ruin = mc["prob_of_ruin_pct"]
        st.markdown(_risk_card(
            "Probability of ruin", f"{ruin:.1f}%",
            f"Chance of a {mc['ruin_threshold_pct']:.0f}%+ drawdown in the next {mc['horizon_trades']} trades",
            "#FF8A73" if ruin >= 10 else "#5FD39A"), unsafe_allow_html=True)

    st.caption(f"The breaker replays the last trading day in your {active_tradebook_name} ({brk['replay_date']}): "
               f"day P&L ₹{brk['current_daily_pnl']:,.0f}, {brk['daily_trade_count']} trades, "
               f"{brk['consecutive_losses']} losses in a row, daily limit ₹{brk['daily_loss_limit_inr']:,.0f} "
               f"(tightened for VIX {live_vix:.1f}). {brk['reason']}")

    # --- Row 2: Monte Carlo cone + recommended strategy ---
    mc_col, st_col = st.columns([3, 2])
    with mc_col:
        st.markdown("#### Where your account could be in 100 trades")
        pct, x = mc["percentiles"], mc["trade_indices"]
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(x=x, y=pct["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_mc.add_trace(go.Scatter(x=x, y=pct["p5"], fill="tonexty", fillcolor="rgba(237, 234, 226, 0.12)",
                                    line=dict(width=0), name="5–95% range"))
        fig_mc.add_trace(go.Scatter(x=x, y=pct["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_mc.add_trace(go.Scatter(x=x, y=pct["p25"], fill="tonexty", fillcolor="rgba(237, 234, 226, 0.25)",
                                    line=dict(width=0), name="25–75% range"))
        fig_mc.add_trace(go.Scatter(x=x, y=pct["p50"], line=dict(color="#D4AF37", width=2.5), name="Median path"))
        fig_mc.add_hline(y=RISK_CAPITAL_INR, line_color="#A39E93", line_dash="dash", line_width=1)
        fig_mc.add_hline(y=RISK_CAPITAL_INR * (1 - mc["ruin_threshold_pct"] / 100), line_color="#FF8A73",
                         line_dash="dot", line_width=1, annotation_text="Ruin line",
                         annotation_font_color="#FF8A73")
        fig_mc.update_layout(template="plotly_dark", paper_bgcolor="#121212", plot_bgcolor="#1E1D1B", height=340,
                             margin=dict(l=10, r=10, t=10, b=40), xaxis_title="Trades from now",
                             yaxis_title="Account value (₹)", hovermode="x unified",
                             legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5))
        st.plotly_chart(fig_mc, width="stretch")
        st.caption(f"{mc['num_simulations']:,} simulated futures, resampled from your past trades. "
                   f"Median ending value ₹{mc['median_final_equity']:,.0f}; the worst 5% end below "
                   f"₹{mc['p5_final_equity']:,.0f}. 95% worst-case drawdown: {mc['mdd_95_pct']:.1f}%.")

    with st_col:
        top = snap["top_strategy"]
        st.markdown("#### Strategy for today's regime")
        if top:
            prof = top["profile"]
            legs_html = "".join(f"<li>{leg}</li>" for leg in top["legs_detail"])
            breakevens = ", ".join(f"{b:,.0f}" for b in prof["breakeven_points"]) or "—"
            st.markdown(f"""
            <div class="clean-card">
                <div style="font-size: 1.15rem; font-weight: 700;">{top['strategy_name']}</div>
                <div style="font-size: 0.85rem; color: #A39E93; margin-bottom: 8px;">{top['regime_fit']}</div>
                <ul style="font-size: 0.88rem; margin: 0 0 8px 18px; padding: 0;">{legs_html}</ul>
                <div style="display: flex; gap: 18px; font-size: 0.85rem;">
                    <div><span style="color:#A39E93;">Max profit</span><br><b style="color:#5FD39A;">₹{prof['max_profit_inr']:,.0f}</b></div>
                    <div><span style="color:#A39E93;">Max loss</span><br><b style="color:#FF8A73;">₹{prof['max_loss_inr']:,.0f}</b></div>
                    <div><span style="color:#A39E93;">Breakeven</span><br><b>{breakevens}</b></div>
                </div>
            </div>""", unsafe_allow_html=True)
            fig_st = go.Figure()
            fig_st.add_hline(y=0, line_color="#A39E93", line_dash="dash", line_width=1)
            fig_st.add_trace(go.Scatter(x=prof["price_range"], y=prof["expiry_payoff_curve"], name="At expiry",
                                        line=dict(color="#5FD39A", width=2), fill="tozeroy",
                                        fillcolor="rgba(95, 211, 154, 0.10)"))
            fig_st.add_trace(go.Scatter(x=prof["price_range"], y=prof["t0_payoff_curve"], name="Today (T+0)",
                                        line=dict(color="#B9B2A3", width=2, dash="dot")))
            fig_st.add_vline(x=live_spot, line_color="#EDEAE2", line_width=1)
            fig_st.update_layout(template="plotly_dark", paper_bgcolor="#121212", plot_bgcolor="#1E1D1B",
                                 height=230, margin=dict(l=10, r=10, t=10, b=30),
                                 legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5))
            st.plotly_chart(fig_st, width="stretch")
            st.caption(f"Stop: {top['stop_loss_rule']}  ·  Target: {top['target_rule']}")
        else:
            st.info("No defined-risk strategy fits the current regime.")

    # --- Row 3: position sizing + regime inputs ---
    sz_col, in_col = st.columns(2)
    with sz_col:
        sz = snap["sizing"]
        st.markdown("#### How big should your next trade be?")
        edge = ("You have a positive edge" if sz["has_positive_edge"]
                else "No positive edge in your history, so trade the minimum")
        lots = sz["recommended_lots"]
        st.markdown(f"""
        <div class="clean-card">
            <div class="clean-metric-title">Quarter-Kelly position size</div>
            <div class="clean-metric-val nowrap-num">{lots} lot{'s' if lots != 1 else ''}
                <span style="font-size: 0.95rem; color: #A39E93;">({sz['total_quantity']} qty)</span></div>
            <div style="font-size: 0.85rem; color: #A39E93; margin-top: 6px;">
                {edge}. Win rate {sz['win_rate_pct']:.1f}%, payoff ratio {sz['payoff_ratio']:.2f}.
                Risk ₹{sz['actual_total_risk_inr']:,.0f} ({sz['actual_risk_pct_of_capital']:.2f}% of
                ₹{RISK_CAPITAL_INR:,.0f}) with a {DEFAULT_STOP_LOSS_POINTS:.0f}-point stop.
            </div>
        </div>""", unsafe_allow_html=True)

    with in_col:
        st.markdown("#### What the regime is based on")
        ivp_note = "estimated, no VIX history" if inp["iv_percentile_estimated"] else "vs last 12 months"
        rows = "".join([
            _row("EMA 20 / EMA 50 (5-min)", f"{inp['ema_20']:,.0f} / {inp['ema_50']:,.0f}"),
            _row("ADX 14 (trend strength)", f"{inp['adx_14']:.1f}"),
            _row(f"India VIX · IV percentile ({ivp_note})", f"{live_vix:.2f} · {inp['iv_percentile']:.0f}%"),
            _row("Put-call ratio (simulated OI)", f"{inp['pcr_oi']:.2f}"),
            _row("Max pain (simulated OI)", f"{inp['max_pain']:,.0f}"),
        ])
        st.markdown(f"<div class='clean-card' style='font-size: 0.9rem; line-height: 1.9;'>{rows}</div>",
                    unsafe_allow_html=True)
        st.caption("Open interest comes from the app's simulated option chain, so PCR and max pain are illustrative.")

    # --- Row 4: every defined-risk strategy, addable to the portfolio ---
    def _strategy_card(strat: dict, key_prefix: str, min_height: int = 0) -> None:
        prof = strat["profile"]
        credit = prof["net_premium_total_inr"]
        flow = (f"Collect ₹{credit:,.0f} credit" if credit > 0 else f"Pay ₹{abs(credit):,.0f} debit")
        breakevens = ", ".join(f"{b:,.0f}" for b in prof["breakeven_points"]) or "—"
        legs_html = "".join(f"<li>{leg}</li>" for leg in strat["legs_detail"])
        fit_badge = ("<span class='pill-badge pill-optimal'>Fits today</span>" if strat["fits_regime"]
                     else f"<span class='pill-badge' style='background: #2E2C29; color: #A39E93;'>{strat['category'].title()}</span>")
        st.markdown(f"""
        <div class="clean-card" style="min-height: {min_height}px;">
            <div style="display: flex; justify-content: space-between; align-items: center; gap: 10px;">
                <div style="font-size: 1.05rem; font-weight: 700;">{strat['strategy_name']}</div>{fit_badge}
            </div>
            <div style="font-size: 0.82rem; color: #A39E93; margin: 4px 0 8px;">{strat['regime_fit']}</div>
            <ul style="font-size: 0.85rem; margin: 0 0 10px 18px; padding: 0;">{legs_html}</ul>
            <div style="display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.82rem;">
                <div><span style="color:#A39E93;">Max profit</span><br><b class="val-green">₹{prof['max_profit_inr']:,.0f}</b></div>
                <div><span style="color:#A39E93;">Max loss</span><br><b class="val-red">₹{prof['max_loss_inr']:,.0f}</b></div>
                <div><span style="color:#A39E93;">Breakeven</span><br><b>{breakevens}</b></div>
            </div>
            <div style="font-size: 0.8rem; color: #A39E93; margin-top: 8px;">{flow} · {len(strat['legs'])} legs of {strat['legs'][0].quantity} qty</div>
        </div>""", unsafe_allow_html=True)
        if st.button(f"+ Add {strat['strategy_name']} to portfolio", key=f"{key_prefix}_{strat['strategy_name']}",
                     width="stretch"):
            group = st.session_state.portfolio_manager.add_strategy(strat["strategy_name"], strat["legs"])
            st.toast(f"Added {group} ({len(strat['legs'])} legs) to your portfolio.")

    st.divider()
    st.markdown("#### Every defined-risk strategy")
    st.write("Each one caps the worst-case loss in advance. Add any of them to the Active Portfolio tab "
             "to track it live; legs are added at today's model price.")
    fitting = [s for s in snap["strategies"] if s["fits_regime"]]
    others = [s for s in snap["strategies"] if not s["fits_regime"]]
    if fitting:
        for col, strat in zip(st.columns(max(len(fitting), 2)), fitting):
            with col:
                _strategy_card(strat, "fit")
    else:
        st.info("No strategy fits today's regime cleanly. The options below suit other market conditions.")
    with st.expander(f"Strategies for other market conditions ({len(others)})"):
        for row_start in range(0, len(others), 3):
            for col, strat in zip(st.columns(3), others[row_start:row_start + 3]):
                with col:
                    _strategy_card(strat, "other", min_height=330)


# ==============================================================================
# TAB 2: OPTION SAFETY & HEALTH SCANNER (OPTION CHAIN)
# ==============================================================================
with tab2:
    st.markdown("#### Option Safety & Health Scanner (Option Chain)")
    st.write("Evaluate live health across the option chain and instantly add trades to your portfolio.")

    # --- Controls: expiry and detail level ---
    expiries = upcoming_expiries(datetime.date.today())
    ctl_exp, ctl_greeks = st.columns([2, 1])
    with ctl_exp:
        expiry_by_label = {e.label: e for e in expiries}
        chosen_expiry = expiry_by_label[st.selectbox("Expiry", list(expiry_by_label), key="chain_expiry")]
    with ctl_greeks:
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        show_greeks = st.toggle("Show Greeks and reasons", key="chain_show_greeks")
    # On expiry day about a quarter of a trading day's time value is left on average.
    chain_dte = chosen_expiry.days_to_expiry if chosen_expiry.days_to_expiry > 0 else 0.25
    st.caption(f"Pricing for expiry {chosen_expiry.expiry:%A %d %B %Y}. Nifty weekly options expire on Tuesdays "
               "and the monthly on the last Tuesday; exchange holidays that move an expiry are not adjusted.")

    atm_base = round(live_spot / 50.0) * 50.0
    strike_list = [int(atm_base + i * 50) for i in range(-5, 6)]

    # Simulated open interest for the same strikes: max pain and the biggest OI walls.
    oi_chain = LiveDataFeed.generate_live_option_chain(live_spot, live_vix, days_to_expiry=chain_dte,
                                                       num_strikes_each_side=5)
    max_pain_strike = int(compute_max_pain(oi_chain))
    resistance_strike = int(oi_chain.loc[oi_chain["ce_oi"].idxmax(), "strike"])
    support_strike = int(oi_chain.loc[oi_chain["pe_oi"].idxmax(), "strike"])
    strike_tags = {}
    for tag_strike, tag in ((max_pain_strike, "Max pain"), (resistance_strike, "Resistance"),
                            (support_strike, "Support")):
        strike_tags.setdefault(tag_strike, []).append(tag)

    st.markdown("<br>", unsafe_allow_html=True)

    # Header Row
    head_style = "color: #A39E93; font-size: 0.8rem; font-weight: 700; white-space: nowrap;"
    h1, h2, h3, h4, h5, h6 = st.columns([1.5, 2.5, 1.5, 2.5, 1.5, 2])
    h1.markdown(f"<div style='text-align: right; {head_style}'>Call LTP</div>", unsafe_allow_html=True)
    h2.markdown(f"<div style='text-align: right; {head_style}'>Call Verdict</div>", unsafe_allow_html=True)
    h3.markdown(f"<div style='text-align: center; {head_style}'>Strike</div>", unsafe_allow_html=True)
    h4.markdown(f"<div style='text-align: left; {head_style}'>Put Verdict</div>", unsafe_allow_html=True)
    h5.markdown(f"<div style='text-align: left; {head_style}'>Put LTP</div>", unsafe_allow_html=True)
    h6.markdown(f"<div style='text-align: center; {head_style}'>Action</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin: 5px 0px 10px 0px; border-top: 1px solid rgba(237, 234, 226, 0.12);'>", unsafe_allow_html=True)

    price_style = ("font-weight: 600; font-family: Manrope, sans-serif; font-variant-numeric: tabular-nums; "
                   "font-size: 1.1rem; color: #EDEAE2; padding-top: 5px;")
    pill_for = {"OPTIMAL": "pill-optimal", "SPECULATIVE": "pill-speculative"}

    def _greeks_line(ev: dict, side: str) -> str:
        return (f"<b style='color: #EDEAE2;'>{side}</b> Δ {ev['delta']:+.2f} · θ −₹{abs(ev['theta_daily_inr']):,.1f}/day "
                f"({ev['daily_theta_burn_pct']:.1f}% of premium) · IV {ev['iv_percent']:.1f}%<br>"
                f"<span style='color: #A39E93;'>{ev['summary']}</span>")

    for strike in strike_list:
        c_eval = OptionEvaluator.evaluate_option_health(live_spot, strike, OptionType.CALL, TradeAction.BUY, chain_dte, live_vix)
        p_eval = OptionEvaluator.evaluate_option_health(live_spot, strike, OptionType.PUT, TradeAction.BUY, chain_dte, live_vix)
        c_pill = pill_for.get(c_eval["health_rating"], "pill-hazardous")
        p_pill = pill_for.get(p_eval["health_rating"], "pill-hazardous")

        col1, col2, col3, col4, col5, col6 = st.columns([1.5, 2.5, 1.5, 2.5, 1.5, 2])
        col1.markdown(f"<div style='text-align: right; {price_style}'>₹{c_eval['premium']:.1f}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div style='text-align: right; padding-top: 5px;'><span class='pill-badge {c_pill}' style='font-size:0.75rem; padding: 4px 8px;'>{c_eval['health_badge']}</span></div>", unsafe_allow_html=True)

        tags_html = "".join(f"<div style='font-size: 0.68rem; font-weight: 700; color: #D4AF37; line-height: 1.2;'>{t}</div>"
                            for t in strike_tags.get(strike, []))
        if strike == atm_base:
            col3.markdown(f"<div style='text-align: center; font-weight: 800; font-size: 1.2rem; color: #121212; background: #EDEAE2; border-radius: 6px; padding: 2px 0px;'>{strike}<div style='font-size: 0.68rem; font-weight: 700; color: #5C5850;'>ATM</div></div>{tags_html}", unsafe_allow_html=True)
        else:
            col3.markdown(f"<div style='text-align: center; font-weight: 700; font-size: 1.1rem; color: #EDEAE2; padding-top: 5px;'>{strike}{tags_html}</div>", unsafe_allow_html=True)

        col4.markdown(f"<div style='text-align: left; padding-top: 5px;'><span class='pill-badge {p_pill}' style='font-size:0.75rem; padding: 4px 8px;'>{p_eval['health_badge']}</span></div>", unsafe_allow_html=True)
        col5.markdown(f"<div style='text-align: left; {price_style}'>₹{p_eval['premium']:.1f}</div>", unsafe_allow_html=True)

        with col6:
            b1, b2 = st.columns(2)
            if b1.button("+ CE", key=f"add_ce_{strike}"):
                st.session_state.portfolio_manager.add_position(float(strike), OptionType.CALL, TradeAction.BUY, 50, c_eval['premium'], f"NIFTY {strike} CE")
                st.toast(f"Added NIFTY {strike} CE to Portfolio!")
            if b2.button("+ PE", key=f"add_pe_{strike}"):
                st.session_state.portfolio_manager.add_position(float(strike), OptionType.PUT, TradeAction.BUY, 50, p_eval['premium'], f"NIFTY {strike} PE")
                st.toast(f"Added NIFTY {strike} PE to Portfolio!")

        if show_greeks:
            g_call, g_put = st.columns(2)
            g_call.markdown(f"<div style='font-size: 0.8rem; line-height: 1.5; padding: 2px 0 6px;'>{_greeks_line(c_eval, 'CE')}</div>", unsafe_allow_html=True)
            g_put.markdown(f"<div style='font-size: 0.8rem; line-height: 1.5; padding: 2px 0 6px;'>{_greeks_line(p_eval, 'PE')}</div>", unsafe_allow_html=True)

        st.markdown("<hr style='margin: 2px 0px; padding: 0px; border-top: 1px solid #2E2C29;'>", unsafe_allow_html=True)

    # --- Open interest by strike ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Open interest by strike")
    oi_sorted = oi_chain.sort_values("strike")
    oi_labels = [str(int(k)) for k in oi_sorted["strike"]]
    fig_oi = go.Figure()
    fig_oi.add_trace(go.Bar(y=oi_labels, x=-oi_sorted["ce_oi"], orientation="h", name="Call OI (resistance)",
                            marker_color="#FF8A73", customdata=oi_sorted["ce_oi"],
                            hovertemplate="Strike %{y}<br>Call OI %{customdata:,}<extra></extra>"))
    fig_oi.add_trace(go.Bar(y=oi_labels, x=oi_sorted["pe_oi"], orientation="h", name="Put OI (support)",
                            marker_color="#5FD39A",
                            hovertemplate="Strike %{y}<br>Put OI %{x:,}<extra></extra>"))
    # On a category axis a numeric-looking label is read as a slot number, so place by position.
    max_pain_pos = oi_labels.index(str(max_pain_strike)) if str(max_pain_strike) in oi_labels else None
    if max_pain_pos is not None:
        fig_oi.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=max_pain_pos, y1=max_pain_pos,
                         line=dict(color="#D4AF37", width=2, dash="dash"))
        fig_oi.add_annotation(xref="paper", x=1, yref="y", y=max_pain_pos, text=f"Max pain {max_pain_strike}",
                              showarrow=False, xanchor="right", yanchor="bottom", font=dict(color="#D4AF37", size=12))
    max_oi = float(max(oi_sorted["ce_oi"].max(), oi_sorted["pe_oi"].max())) * 1.1
    fig_oi.update_layout(template="plotly_dark", paper_bgcolor="#121212", plot_bgcolor="#1E1D1B", height=420,
                         barmode="overlay", bargap=0.25, margin=dict(l=10, r=10, t=10, b=40),
                         xaxis=dict(range=[-max_oi, max_oi], title="← Calls · contracts · Puts →",
                                    tickvals=[-max_oi * 0.6, 0, max_oi * 0.6],
                                    ticktext=[f"{max_oi * 0.6:,.0f}", "0", f"{max_oi * 0.6:,.0f}"]),
                         yaxis=dict(title="Strike", type="category"),  # strikes are labels, not a number scale
                         legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5))
    st.plotly_chart(fig_oi, width="stretch")
    st.caption(f"Biggest call wall at {resistance_strike} (likely resistance), biggest put wall at {support_strike} "
               f"(likely support), max pain at {max_pain_strike}. Open interest here is simulated for the demo; "
               "prices and Greeks come from live spot and India VIX.")

# ==============================================================================
# TAB 3: TRADE TESTER (SHOULD I TAKE OR DROP?)
# ==============================================================================
with tab3:
    st.subheader("Trade Tester: Will this trade give Profit or Loss?")
    st.write("Test your trade before putting in real money: a verdict, a second opinion from the ML model, "
             "a check against your daily loss limit, and a map of profit and loss as Nifty moves and time passes.")

    t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns([1, 1, 1, 1, 1.4])
    with t_col1:
        t_strike = st.selectbox("Strike Price", strike_list, index=8, key="test_stk_sel")
    with t_col2:
        t_type_str = st.selectbox("Option Type", ["CALL (CE)", "PUT (PE)"], key="test_typ_sel")
        t_type = OptionType.CALL if "CALL" in t_type_str else OptionType.PUT
    with t_col3:
        t_act_str = st.selectbox("Action", ["BUY (Long)", "SELL (Short)"], key="test_act_sel")
        t_act = TradeAction.BUY if "BUY" in t_act_str else TradeAction.SELL
    with t_col4:
        t_lots = st.number_input("Lots (50 qty each)", min_value=1, max_value=20, value=1, key="test_lots")
    with t_col5:
        t_expiry = expiry_by_label[st.selectbox("Expiry", list(expiry_by_label), key="test_expiry")]
    t_dte = t_expiry.days_to_expiry if t_expiry.days_to_expiry > 0 else 0.25

    sim_res = OptionEvaluator.test_trade_scenarios(
        spot_price=live_spot, strike_price=float(t_strike), option_type=t_type, action=t_act,
        days_to_expiry=t_dte, india_vix=live_vix, num_lots=int(t_lots),
    )
    h_data = sim_res["health"]
    t_qty = sim_res["total_quantity"]

    # --- Verdict banner, in the app's palette ---
    DECISION_COLOURS = {"TAKE": ("#5FD39A", "rgba(95, 211, 154, 0.10)"),
                        "HEDGE": ("#F0913A", "rgba(240, 145, 58, 0.10)"),
                        "DROP": ("#FF8A73", "rgba(255, 138, 115, 0.10)")}
    dec_fg, dec_bg = DECISION_COLOURS.get(sim_res["decision"], ("#EDEAE2", "#1E1D1B"))
    st.markdown(f"""
    <div style="background: {dec_bg}; border: 2px solid {dec_fg}; border-radius: 16px; padding: 20px 24px; margin: 15px 0;">
        <div style="font-size: 1.4rem; font-weight: 800; color: {dec_fg}; margin-bottom: 6px;">{sim_res['decision_badge']}</div>
        <div style="font-size: 1rem; color: #EDEAE2; margin-bottom: 6px;"><strong>Why:</strong> {sim_res['decision_reason']}</div>
        <div style="font-size: 0.9rem; color: #A39E93;"><strong>Expert Tip:</strong> {sim_res['suggested_alternative']}</div>
    </div>""", unsafe_allow_html=True)

    # --- Four cards: cost, chance of profit, decay, risk budget ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="clean-card" style="min-height: 150px;">
            <div class="clean-metric-title">{'Premium paid' if t_act == TradeAction.BUY else 'Premium collected'}</div>
            <div class="clean-metric-val nowrap-num">₹{sim_res['total_investment_inr']:,.0f}</div>
            <div style="font-size: 0.8rem; color: #A39E93;">{t_lots} lot ({t_qty} qty) @ ₹{h_data['premium']:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="clean-card" style="min-height: 150px;">
            <div class="clean-metric-title">Chance of profit</div>
            <div class="clean-metric-val {'val-green' if h_data['prob_of_profit_pct'] >= 50 else 'val-red'}">{h_data['prob_of_profit_pct']:.0f}%</div>
            <div style="font-size: 0.8rem; color: #A39E93;">From delta {h_data['delta']:+.2f}</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        # Theta is a per-day rate; with under a day left (or a huge rate) the real loss is capped by the premium.
        decay_inr = min(abs(h_data['theta_daily_inr']) * t_qty, sim_res['total_investment_inr'])
        decay_capped = h_data['daily_theta_burn_pct'] >= 100.0 or t_dte < 1
        decay_value = (f"{'−' if t_act == TradeAction.BUY else '+'}₹{decay_inr:,.0f}"
                       + ("" if decay_capped else "/day"))
        decay_note = ("Most of the premium decays before today's close" if decay_capped
                      else f"{h_data['daily_theta_burn_pct']:.1f}% of the premium per day")
        st.markdown(f"""
        <div class="clean-card" style="min-height: 150px;">
            <div class="clean-metric-title">Time decay if Nifty stays flat</div>
            <div class="clean-metric-val nowrap-num {'val-red' if t_act == TradeAction.BUY else 'val-green'}">{decay_value}</div>
            <div style="font-size: 0.8rem; color: #A39E93;">{decay_note}</div>
        </div>""", unsafe_allow_html=True)

    t_grid = OptionEvaluator.pnl_grid(live_spot, float(t_strike), t_type, t_act, t_dte, live_vix,
                                      h_data["premium"], t_qty)
    if t_act == TradeAction.BUY:
        t_max_loss = sim_res["total_investment_inr"]
        t_loss_note = "the most a bought option can lose"
    else:
        t_max_loss = max(0.0, -float(t_grid.loc[[-2.0, 2.0], "Expiry"].min()))
        t_loss_note = "on a 2% move against you; a naked sale can lose more"
    t_limit = DrawdownContainmentEngine.calculate_adaptive_daily_limit(RISK_CAPITAL_INR, india_vix=live_vix)
    t_budget = check_trade_budget(t_max_loss, t_limit, t_max_loss / int(t_lots) if t_lots else t_max_loss)
    with k4:
        budget_cls = "val-green" if t_budget.fits else "val-red"
        budget_line = ("Fits inside today's limit" if t_budget.fits else
                       (f"Too big: {t_budget.max_lots_within_limit} lot{'s' if t_budget.max_lots_within_limit != 1 else ''} would fit"
                        if t_budget.max_lots_within_limit else "Too big even at 1 lot: use a spread"))
        st.markdown(f"""
        <div class="clean-card" style="min-height: 150px;">
            <div class="clean-metric-title">Worst case vs daily loss limit</div>
            <div class="clean-metric-val nowrap-num {budget_cls}">{t_budget.limit_used_ratio:.1f}×</div>
            <div style="font-size: 0.8rem; color: #A39E93;">₹{t_max_loss:,.0f} vs ₹{t_limit:,.0f} limit. {budget_line}.</div>
        </div>""", unsafe_allow_html=True)
    st.caption(f"Worst case is {t_loss_note}. Daily limit: 1.5% of ₹{RISK_CAPITAL_INR:,.0f}, tightened for VIX {live_vix:.1f}. "
               f"Quarter-Kelly sizing from your tradebook suggests {snap['sizing']['recommended_lots']} lot.")

    # --- Second opinion from the ML model ---
    if t_act == TradeAction.BUY:
        ml_prob = get_predictor().probability_profitable(
            delta=h_data["delta"], gamma=h_data["gamma"], theta_burn_pct=h_data["daily_theta_burn_pct"],
            vix=live_vix, dte=t_dte)
        ml_says = "TAKE" if ml_prob >= 0.5 else "DROP"
        rules_positive = sim_res["decision"] == "TAKE"
        agree = (ml_says == "TAKE") == rules_positive
        agree_text = ("The rules and the model agree." if agree else
                      "The rules and the model disagree: treat this trade with extra caution.")
        st.markdown(f"""
        <div class="clean-card" style="display: flex; flex-wrap: wrap; gap: 8px 28px; align-items: baseline;">
            <span class="clean-metric-title" style="margin: 0;">Second opinion</span>
            <span style="font-size: 1rem;">Rules: <b>{sim_res['decision']}</b></span>
            <span style="font-size: 1rem;">ML model: <b class="{'val-green' if ml_says == 'TAKE' else 'val-red'}">{ml_prob:.0%} chance profitable</b> → leans {ml_says}</span>
            <span style="font-size: 0.85rem; color: #A39E93;">{agree_text} The model is trained on simulated trades (see the ML tab).</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.caption("Second opinion: the ML model only covers bought options, so it doesn't rate sales.")

    # --- Profit and loss heatmap ---
    st.markdown("#### Profit or loss as Nifty moves and time passes")
    grid_rows = [f"{m:+.1f}%  ({live_spot * (1 + m / 100):,.0f})" for m in t_grid.index]
    z_cap = max(1.0, float(t_grid.abs().to_numpy().max()))
    fig_heat = go.Figure(go.Heatmap(
        z=t_grid.to_numpy(), x=list(t_grid.columns), y=grid_rows, zmid=0, zmin=-z_cap, zmax=z_cap,
        colorscale=[[0, "#FF8A73"], [0.5, "#1E1D1B"], [1, "#5FD39A"]], showscale=False,
        text=[["₹0" if abs(v) < 0.5 else f"{'+' if v > 0 else '−'}₹{abs(v):,.0f}" for v in row] for row in t_grid.to_numpy()],
        texttemplate="%{text}", textfont=dict(size=12, color="#EDEAE2"),
        hovertemplate="Nifty %{y}<br>%{x}: %{text}<extra></extra>", xgap=2, ygap=2,
    ))
    fig_heat.update_layout(template="plotly_dark", paper_bgcolor="#121212", plot_bgcolor="#121212", height=430,
                           margin=dict(l=10, r=10, t=10, b=10),
                           xaxis=dict(side="top", title=None), yaxis=dict(title="Nifty move (level)", autorange="reversed"))
    st.plotly_chart(fig_heat, width="stretch")
    st.caption(f"Expiry {t_expiry.expiry:%d %b}. Green makes money, coral loses it. Before expiry, values are "
               "Black-Scholes at today's volatility; the last column is the payoff at expiry.")

    with st.expander("Exact numbers for 1-day and expiry scenarios"):
        raw_sc = sim_res["scenario_table"]
        st.dataframe(pd.DataFrame({
            "Nifty Move": raw_sc["nifty_move_pct"],
            "Target Nifty Level": raw_sc["target_nifty_spot"],
            "PnL Today (₹)": raw_sc["Today (T+0) PnL (₹)"],
            "PnL Tomorrow (₹)": raw_sc["Tomorrow (T+1) PnL (₹)"],
            "PnL at Expiry (₹)": raw_sc["Expiry (T=0) PnL (₹)"],
        }), width="stretch", hide_index=True)

    if st.button("+ Add this tested trade to portfolio", width="stretch"):
        st.session_state.portfolio_manager.add_position(
            strike=float(t_strike), option_type=t_type, action=t_act, quantity=t_qty,
            entry_price=h_data["premium"], symbol=f"NIFTY {t_strike} {t_type.value}",
        )
        st.success(f"Added NIFTY {t_strike} {t_type.value} to your Portfolio!")

# ==============================================================================
# TAB 4: MY PORTFOLIO & PAST TRADEBOOK
# ==============================================================================
with tab4:
    pm = st.session_state.portfolio_manager
    port_mtm = pm.update_portfolio_mtm(current_spot=live_spot, india_vix=live_vix, days_to_expiry=5)
    if pm.load_error:
        st.warning(pm.load_error)

    # --- Alert settings ---
    set_c1, set_c2, set_c3 = st.columns([1, 1, 2])
    with set_c1:
        stop_pct = st.number_input("Stop-loss (% of premium)", min_value=5, max_value=100, value=int(DEFAULT_STOP_LOSS_PCT),
                                   step=5, key="pf_stop_pct")
    with set_c2:
        target_pct = st.number_input("Target (% of premium)", min_value=5, max_value=500, value=int(DEFAULT_TARGET_PCT),
                                     step=5, key="pf_target_pct")
    with set_c3:
        st.caption("Each position is checked against these. Positions are saved on this computer, so they "
                   "survive a page refresh or restart.")

    # --- Summary cards: P&L, decay, open loss vs the daily limit ---
    pf_limit = DrawdownContainmentEngine.calculate_adaptive_daily_limit(RISK_CAPITAL_INR, india_vix=live_vix)
    p_c1, p_c2, p_c3 = st.columns(3)
    with p_c1:
        tot_pnl = port_mtm["total_mtm_pnl"]
        pnl_cls = "val-green" if tot_pnl >= 0 else "val-red"
        st.markdown(f"""
        <div class="clean-card hero-card" style="min-height: 150px;">
            <div class="clean-metric-title">Total live profit / loss</div>
            <div class="clean-metric-val nowrap-num {pnl_cls}">₹{tot_pnl:,.2f}</div>
            <div class="hero-sub" style="font-size: 0.8rem;">{port_mtm['total_mtm_pct']:+.1f}% on invested capital</div>
        </div>""", unsafe_allow_html=True)
    with p_c2:
        th_val = port_mtm["net_theta_per_day_inr"]
        th_cls = "val-green" if th_val >= 0 else "val-red"
        th_sign = "−" if th_val < 0 else "+"
        st.markdown(f"""
        <div class="clean-card" style="min-height: 150px;">
            <div class="clean-metric-title">Portfolio daily decay</div>
            <div class="clean-metric-val nowrap-num {th_cls}">{th_sign}₹{abs(th_val):,.0f}<span style="font-size: 0.9rem; font-weight: 600;"> / day</span></div>
            <div style="font-size: 0.8rem; color: #A39E93;">{port_mtm['theta_status']}</div>
        </div>""", unsafe_allow_html=True)
    with p_c3:
        open_loss = max(0.0, -tot_pnl)
        used = open_loss / pf_limit if pf_limit > 0 else 0.0
        used_cls = "val-green" if used < 0.5 else ("val-amber" if used < 1.0 else "val-red")
        used_note = ("In profit: no loss against today's limit" if open_loss == 0 else
                     ("Within today's limit" if used < 1.0 else "Over today's limit: the breaker would halt trading"))
        st.markdown(f"""
        <div class="clean-card" style="min-height: 150px;">
            <div class="clean-metric-title">Open loss vs daily limit</div>
            <div class="clean-metric-val nowrap-num {used_cls}">{used:.0%}</div>
            <div style="font-size: 0.8rem; color: #A39E93;">₹{open_loss:,.0f} of ₹{pf_limit:,.0f}. {used_note}.</div>
        </div>""", unsafe_allow_html=True)

    # --- Greeks in plain words ---
    if port_mtm["total_positions"] > 0:
        gw = greeks_in_words(port_mtm, live_spot)
        st.markdown(f"""
        <div class="clean-card">
            <div class="clean-metric-title">What moves your portfolio</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px 28px; font-size: 0.92rem; line-height: 1.5;">
                <div><b>Price</b> <span style="color: #A39E93;">(delta {port_mtm['net_delta']:+,.0f}, gamma {port_mtm['net_gamma']:+.3f})</span><br>{gw['delta_sentence']}</div>
                <div><b>Time</b> <span style="color: #A39E93;">(theta {gw['theta_per_day_inr']:+,.0f}/day)</span><br>{gw['theta_sentence']}</div>
                <div><b>Volatility</b> <span style="color: #A39E93;">(vega {gw['vega_per_point_inr']:+,.0f})</span><br>{gw['vega_sentence']}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    sub_col1, sub_col2 = st.columns([1, 1])

    # Left: Active Positions Book, grouped by strategy
    with sub_col1:
        st.markdown("### My Active Positions")

        if port_mtm["total_positions"] > 0:
            ALERT_STYLE = {"OK": ("#2E2C29", "#A39E93"), "Near stop": ("#F0913A", "#121212"),
                           "Stop hit": ("#FF8A73", "#121212"), "Target hit": ("#5FD39A", "#121212")}
            pos_df = port_mtm["positions_dataframe"]
            pos_df = pos_df.assign(alert=[position_alert(x, stop_pct, target_pct) for x in pos_df["pnl_pct"]])

            def _position_row(row) -> None:
                r_c1, r_c2, r_c3, r_c4, r_c5 = st.columns([2.3, 1.4, 1.4, 1.6, 0.8])
                side = "S" if row["action"] == "SELL" else "B"
                r_c1.markdown(f"<span style='font-size:0.88rem;'><b style='color:#A39E93;'>{side}</b> {row['symbol']} × {row['qty']}</span>", unsafe_allow_html=True)
                r_c2.markdown(f"<span style='font-size:0.85rem; white-space:nowrap; color:#A39E93;'>₹{row['entry_px']:.1f} → ₹{row['ltp']:.1f}</span>", unsafe_allow_html=True)
                pnl = row["mtm_pnl"]
                pnl_color = "#5FD39A" if pnl >= 0 else "#FF8A73"
                r_c3.markdown(f"<span style='font-size:0.9rem; font-weight:600; white-space:nowrap; color:{pnl_color};'>{'−' if pnl < 0 else '+'}₹{abs(pnl):,.0f}</span> <span style='font-size:0.75rem; color:#A39E93;'>{row['pnl_pct']:+.0f}%</span>", unsafe_allow_html=True)
                bg, fg = ALERT_STYLE[row["alert"]]
                r_c4.markdown(f"<span class='pill-badge' style='background:{bg}; color:{fg};'>{row['alert']}</span>", unsafe_allow_html=True)
                with r_c5:
                    if st.button(":material/close:", key=f"del_{row['position_id']}", help="Remove this position"):
                        pm.remove_position(row["position_id"])
                        st.rerun()

            groups = [g for g in pos_df["group"].dropna().unique()]
            for group in groups:
                legs_df = pos_df[pos_df["group"] == group]
                g_pnl = legs_df["mtm_pnl"].sum()
                g_col1, g_col2 = st.columns([3, 1.2])
                g_col1.markdown(f"<div style='font-weight: 700; padding-top: 6px;'>{group} <span style='font-weight: 600; color: {'#5FD39A' if g_pnl >= 0 else '#FF8A73'};'>{'−' if g_pnl < 0 else '+'}₹{abs(g_pnl):,.0f}</span> <span style='font-size: 0.8rem; color: #A39E93;'>{len(legs_df)} legs</span></div>", unsafe_allow_html=True)
                with g_col2:
                    if st.button("Close strategy", key=f"close_{group}", width="stretch"):
                        pm.remove_group(group)
                        st.rerun()
                for _, row in legs_df.iterrows():
                    _position_row(row)
                st.markdown("<hr style='margin: 6px 0px; border-top: 1px solid #2E2C29;'>", unsafe_allow_html=True)

            singles = pos_df[pos_df["group"].isna()]
            if not singles.empty:
                if groups:
                    st.markdown("<div style='font-weight: 700; padding-top: 6px;'>Single positions</div>", unsafe_allow_html=True)
                for _, row in singles.iterrows():
                    _position_row(row)

            st.markdown("<hr style='margin: 10px 0px; border-top: 1px solid rgba(237, 234, 226, 0.12);'>", unsafe_allow_html=True)
            act_c1, act_c2 = st.columns(2)
            with act_c1:
                st.download_button("Download as CSV", data=pm.to_csv(live_spot, live_vix),
                                   file_name=f"sentinel_portfolio_{datetime.date.today():%Y%m%d}.csv",
                                   mime="text/csv", width="stretch")
            with act_c2:
                if st.button("Clear All Positions", type="secondary", width="stretch"):
                    pm.clear_portfolio()
                    st.rerun()
        else:
            st.info("No active positions yet. Add trades from the Risk Command Center, the option chain or the Trade Tester.")

    # Right: Payoff Chart
    with sub_col2:
        st.markdown("### Portfolio Payoff Chart")
        
        if port_mtm["total_positions"] > 0:
            legs = []
            for p in pm.positions:
                legs.append(StrategyLeg(
                    action=p.action,
                    option_type=p.option_type,
                    strike=p.strike,
                    expiry_date=None,
                    quantity=p.quantity,
                    premium=p.entry_price
                ))

            if legs:
                port_iv = max(0.08, live_vix / 100.0)  # VIX 17.42 -> 0.1742, same as portfolio MTM
                profile = PayoffCalculator.calculate_strategy_profile(legs, live_spot, 5, 0.07, port_iv)
                
                fig = go.Figure()
                fig.add_hline(y=0, line_color="#A39E93", line_dash="dash", line_width=1)
                
                fig.add_trace(go.Scatter(
                    x=profile["price_range"], 
                    y=profile["expiry_payoff_curve"], 
                    mode="lines", 
                    name="Expiry Payoff", 
                    line=dict(color="#5FD39A", width=2),
                    fill='tozeroy',
                    fillcolor='rgba(95, 211, 154, 0.1)'
                ))
                
                fig.add_trace(go.Scatter(
                    x=profile["price_range"], 
                    y=profile["t0_payoff_curve"], 
                    mode="lines", 
                    name="T+0 Payoff (Live)", 
                    line=dict(color="#B9B2A3", width=2, dash="dot")
                ))
                
                fig.add_vline(x=live_spot, line_color="#EDEAE2", line_width=1.5, annotation_text="Live Spot")
                
                fig.update_layout(
                    template="plotly_dark", 
                    paper_bgcolor="#121212", 
                    plot_bgcolor="#1E1D1B", 
                    height=380, 
                    margin=dict(l=10, r=10, t=20, b=50),
                    legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, width="stretch")
                
                sc_c1, sc_c2 = st.columns(2)
                sc_c1.metric("Max Profit", f"₹{profile['max_profit_inr']:,.0f}")
                sc_c2.metric("Max Loss", f"₹{profile['max_loss_inr']:,.0f}")
        else:
            st.info("Add options to see the combined payoff graph.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()
    
    # Bottom: Past Tradebook Auditor & Mistakes
    st.markdown("### Mistakes Detected in Past Trades")
    st.write("Analysis of your historical tradebook to identify capital leakage.")
    
    st.caption(f"Analysing your {active_tradebook_name} ({len(active_trades)} trades). "
               "Upload your own from the Tradebook panel above the tabs.")
    tb_df = active_trades
    if not tb_df.empty:
        audit_res = LeakageDetector.audit_tradebook(tb_df, initial_capital=200000.0)
        sim_res_tb = CounterfactualSimulator.simulate_containment(tb_df, initial_capital=200000.0, max_daily_loss_inr=4000.0)

        # Health Score & Money Saved
        sc_c1, sc_c2 = st.columns(2)
        with sc_c1:
            b_score = audit_res["behavioral_health_score"]
            sc_cls = "val-green" if b_score >= 75 else ("val-amber" if b_score >= 50 else "val-red")
            st.markdown(f"""
            <div class="clean-card">
                <div class="clean-metric-title">Discipline Score</div>
                <div class="clean-metric-val {sc_cls}">{b_score}/100</div>
                <div style="font-size: 0.8rem; color: #A39E93;">Emotional Control Rating</div>
            </div>
            """, unsafe_allow_html=True)

        with sc_c2:
            saved_cap = sim_res_tb["net_capital_saved"]
            st.markdown(f"""
            <div class="clean-card">
                <div class="clean-metric-title">Money Saved with Stop-Loss</div>
                <div class="clean-metric-val val-green">+₹{saved_cap:,.2f}</div>
                <div style="font-size: 0.8rem; color: #A39E93;">Drawdown reduced by {sim_res_tb['drawdown_reduction_pct']:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Mistakes List
        for leak in audit_res["leakage_breakdown"]:
            if leak["financial_damage"] > 0:
                st.markdown(f"- **{leak['leak_type']}**: Lost **₹{leak['financial_damage']:,.0f}** ({leak['description']})")

        # Expandable Advanced Monte Carlo & Curve
        with st.expander("View Future Monte Carlo Simulation & Equity Curve"):
            tl = sim_res_tb["comparison_curve"]
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=tl["date"], y=tl["actual_equity"], mode="lines", name="Actual PnL", line=dict(color="#FF8A73", width=2, dash="dot")))
            fig_comp.add_trace(go.Scatter(x=tl["date"], y=tl["sim_equity"], mode="lines", name="With Stop Loss Containment", line=dict(color="#5FD39A", width=3)))
            fig_comp.update_layout(template="plotly_dark", paper_bgcolor="#121212", plot_bgcolor="#1E1D1B", height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_comp, width="stretch")
    else:
        st.info("No tradebook loaded. Upload one from the Tradebook panel above the tabs.")


# ==============================================================================
# TAB 5: ML & BACKTEST ANALYSIS
# ==============================================================================
ML_SOURCES = ("Real Nifty history", "Simulated trades")
SIMULATED_WIN_INR, SIMULATED_LOSS_INR = 5000.0, -3000.0
RULE_MAX_DECAY_PCT, RULE_MAX_VIX = 10.0, 25.0


@st.cache_resource(show_spinner="Building trades from 5 years of Nifty history and training the models...")
def _cached_real_report(allow_download: bool):
    history = load_history(SNAPSHOT_DIR, allow_download=allow_download)
    if history.empty:
        return None, history
    return evaluate_real(build_option_trades(history)), history


@st.cache_resource(show_spinner="Training the models on simulated trades...")
def _cached_sim_report():
    return evaluate_models()


with tab5:
    st.header("Machine Learning & Backtest Engine")
    st.write("A model that rates option trades TAKE or DROP, compared with two alternatives, tested on trades it "
             "has never seen, and backtested at the level of confidence you choose.")

    ml_source = st.radio("Training data", ML_SOURCES, horizontal=True, key="ml_source")
    ml_report, ml_history = (_cached_real_report(not demo_mode) if ml_source == ML_SOURCES[0] else (_cached_sim_report(), None))
    if ml_report is None:
        st.warning("No Nifty history is saved yet and the app is offline, so showing simulated trades. "
                   "Run scripts/save_market_snapshot.py with internet to save the history.")
        ml_report, ml_source = _cached_sim_report(), ML_SOURCES[1]

    is_real = ml_report["source"] == "real"
    if is_real:
        st.info(f"**Real data.** {ml_report['train_size'] + ml_report['test_size']:,} option trades built from "
                f"{len(ml_history):,} trading days of Nifty 50 and India VIX ({ml_history['date'].min():%b %Y} to "
                f"{ml_history['date'].max():%b %Y}). Each day: calls and puts at 5 strikes (−2% to +2%) and 3 expiries "
                f"(2, 7, 14 days), priced with Black-Scholes at that day's close and VIX, held to expiry and labelled by "
                f"what Nifty actually did. Trained on trades before {ml_report['cutoff']:%d %b %Y}, tested on the "
                f"{ml_report['test_size']:,} trades after it; training trades still open at that date are removed.")
    else:
        st.info("**Simulated data.** Trades come from a scoring formula: trained on one random draw (seed 42) and "
                "tested on a separate one (seed 99). Results show how well the models learn the simulated pattern, "
                "not proven performance on real Nifty markets.")

    models = ml_report["models"]
    rf, gb, lr = models["random_forest"], models["gradient_boosting"], models["logistic_regression"]

    # --- 1. How good is the model? ---
    st.subheader("How good is the model?")
    st.write(f"Only {ml_report['profitable_share']:.0%} of the test trades were profitable, so a model that always "
             f"says \"drop\" is already {ml_report['majority_baseline_accuracy']:.0%} accurate. ROC-AUC is the fairer "
             "score: 0.5 is a coin flip, 1.0 is perfect ranking of winners above losers.")

    def _model_row(label: str, key: str, fmt: str = "{:.1%}") -> str:
        values = {name: models[name][key] for name in models}
        best = max(values, key=values.get)
        cells = "".join(
            f"<div style='text-align: right;'><span style='font-size: 0.75rem; color: #A39E93;'>{MODEL_LABELS[name].split(' (')[0]}</span><br>"
            f"<span class='clean-metric-val nowrap-num metric-sm {'val-green' if name == best else ''}'>{fmt.format(values[name])}</span></div>"
            for name in ("random_forest", "gradient_boosting", "logistic_regression"))
        return (f"<div class='clean-card' style='min-height: 128px;'><div class='clean-metric-title'>{label}</div>"
                f"<div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;'>{cells}</div></div>")

    e1, e2 = st.columns(2)
    e1.markdown(_model_row("ROC-AUC on unseen trades", "roc_auc", "{:.3f}"), unsafe_allow_html=True)
    e2.markdown(_model_row(f"Accuracy (bar to beat: {ml_report['majority_baseline_accuracy']:.0%})", "accuracy"), unsafe_allow_html=True)
    e3, e4 = st.columns(2)
    e3.markdown(_model_row("Precision (TAKE calls that won)", "precision"), unsafe_allow_html=True)
    e4.markdown(_model_row("Recall (winners caught at 50%)", "recall"), unsafe_allow_html=True)
    cv_kind = "time-ordered " if is_real else ""
    st.caption(f"{ml_report['cv_folds']}-fold {cv_kind}cross-validated ROC-AUC on the training trades: " + ", ".join(
        f"{MODEL_LABELS[n].split(' (')[0]} {models[n]['cv_auc_mean']:.3f} ± {models[n]['cv_auc_std']:.3f}" for n in models)
        + ". Best value in each card is highlighted.")

    roc_col, cm_col = st.columns([3, 2])
    with roc_col:
        st.markdown("#### ROC curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Coin flip (0.500)", line=dict(color="#6B665D", dash="dash", width=1)))
        for name, colour, dash in (("logistic_regression", "#A39E93", "dot"), ("gradient_boosting", "#B9B2A3", "dash"),
                                   ("random_forest", "#D4AF37", "solid")):
            m = models[name]
            fig_roc.add_trace(go.Scatter(x=m["roc_curve"]["fpr"], y=m["roc_curve"]["tpr"],
                                         name=f"{MODEL_LABELS[name].split(' (')[0]} ({m['roc_auc']:.3f})",
                                         line=dict(color=colour, width=2.5 if name == "random_forest" else 2, dash=dash)))
        fig_roc.update_layout(template="plotly_dark", paper_bgcolor="#121212", plot_bgcolor="#1E1D1B", height=360,
                              margin=dict(l=10, r=10, t=10, b=40),
                              xaxis=dict(title="False alarms (losers called TAKE)", range=[0, 1], tickformat=".0%"),
                              yaxis=dict(title="Winners caught", range=[0, 1], tickformat=".0%"),
                              legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
        st.plotly_chart(fig_roc, width="stretch")
    with cm_col:
        st.markdown("#### Random Forest at 50%: right and wrong calls")
        (tn, fp), (fn, tp) = rf["confusion_matrix"]
        fig_cm = go.Figure(go.Heatmap(
            z=[[1, 0], [0, 1]], x=["Model says DROP", "Model says TAKE"], y=["Actually lost", "Actually won"],
            text=[[f"Correct DROP<br><b>{tn:,}</b>", f"Wrong TAKE<br><b>{fp:,}</b>"],
                  [f"Missed winner<br><b>{fn:,}</b>", f"Correct TAKE<br><b>{tp:,}</b>"]],
            texttemplate="%{text}", textfont=dict(color="#121212", size=13),
            colorscale=[[0, "#FF8A73"], [1, "#5FD39A"]], showscale=False, hoverinfo="skip", xgap=4, ygap=4))
        fig_cm.update_layout(template="plotly_dark", paper_bgcolor="#121212", plot_bgcolor="#121212", height=360,
                             margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_cm, width="stretch")

    ranked = sorted(models, key=lambda n: models[n]["roc_auc"], reverse=True)
    spread = models[ranked[0]]["roc_auc"] - models[ranked[-1]]["roc_auc"]
    takeaway = (f"{MODEL_LABELS[ranked[0]].split(' (')[0]} ranks trades best (ROC-AUC {models[ranked[0]]['roc_auc']:.3f}), "
                f"but all three are within {spread:.3f} of each other.")
    if is_real:
        takeaway += (f" On real data the signal is real but modest. At a 50% cut-off the Random Forest catches only "
                     f"{rf['recall']:.0%} of winners, because most bought options lose: use the slider below to see what a "
                     "lower bar trades off.")
    else:
        takeaway += (" The simulated trades follow a straight-line rule, which a linear model captures directly; switch to "
                     "real Nifty history to see how the models do on messy, real data.")
    st.info(f"Honest takeaway: {takeaway}")

    # --- 2. Backtest with a confidence threshold ---
    st.divider()
    st.subheader("Backtest: how sure must the model be?")
    test_frame = ml_report["test_frame"]
    if is_real:
        bt_pnl = test_frame["pnl_inr"].to_numpy()
        per_day = len(test_frame) / max(1, test_frame["entry_date"].nunique())
        bt_note = (f"Real P&L of each test trade (1 lot, held to expiry), in date order: about {per_day:.0f} trades a "
                   "day, so the starting capital is set high enough to compare strategies. Buying options mostly "
                   "loses, so taking everything is the baseline to beat.")
    else:
        bt_pnl = np.where(test_frame["is_profitable"].to_numpy() == 1, SIMULATED_WIN_INR, SIMULATED_LOSS_INR)
        bt_note = "Simulated trades with a fixed +₹5,000 win and −₹3,000 loss."
    bt_c1, bt_c2 = st.columns([2, 1])
    with bt_c1:
        threshold = st.slider("Take a trade only if the Random Forest's chance of profit is at least",
                              min_value=0.20, max_value=0.80, value=0.40 if is_real else 0.50, step=0.05,
                              format="%.2f", key="ml_threshold")
    with bt_c2:
        bt_capital = st.number_input("Starting capital (₹)", min_value=50000.0, max_value=10000000.0,
                                     value=2000000.0 if is_real else 100000.0, step=100000.0, key="ml_capital")
    rules_ok = ((test_frame["theta_burn_pct"] <= RULE_MAX_DECAY_PCT) & (test_frame["vix"] <= RULE_MAX_VIX)).astype(float).to_numpy()
    strategies = {
        "Take every trade": threshold_backtest(bt_pnl, np.ones(len(bt_pnl)), 0.5, bt_capital),
        "Rule-based filter": threshold_backtest(bt_pnl, rules_ok, 0.5, bt_capital),
        f"ML at {threshold:.0%}+": threshold_backtest(bt_pnl, rf["test_probabilities"], threshold, bt_capital),
    }
    s_cols = st.columns(3)
    for col, (name, res) in zip(s_cols, strategies.items()):
        gain = res["final_capital"] - bt_capital
        col.markdown(f"""
        <div class="clean-card" style="min-height: 150px;">
            <div class="clean-metric-title">{name}</div>
            <div class="clean-metric-val nowrap-num metric-sm {'val-green' if gain >= 0 else 'val-red'}">{'+' if gain >= 0 else '−'}₹{abs(gain):,.0f}</div>
            <div style="font-size: 0.8rem; color: #A39E93;">{res['trades']:,} trades · {res['win_rate']:.1f}% won · worst drawdown {res['max_drawdown_pct']:.1f}%</div>
            {f"<div style='font-size: 0.8rem; color: #FF8A73; font-weight: 700;'>Account wiped out after {res['ruined_after_trades']:,} trades</div>" if res['ruined_after_trades'] else ""}
        </div>""", unsafe_allow_html=True)
    fig_bt = go.Figure()
    for (name, res), colour in zip(strategies.items(), ("#FF8A73", "#A39E93", "#D4AF37")):
        fig_bt.add_trace(go.Scatter(y=res["equity_curve"], name=name, mode="lines",
                                    line=dict(color=colour, width=2.5 if name.startswith("ML") else 1.8)))
    fig_bt.add_hline(y=bt_capital, line_color="#6B665D", line_dash="dash", line_width=1)
    fig_bt.update_layout(template="plotly_dark", paper_bgcolor="#121212", plot_bgcolor="#1E1D1B", height=360,
                         margin=dict(l=10, r=10, t=10, b=40), xaxis_title="Test trades, in date order",
                         yaxis_title="Account value (₹)",
                         legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
    st.plotly_chart(fig_bt, width="stretch")
    st.caption(f"{bt_note} Rule-based filter: skip trades losing over {RULE_MAX_DECAY_PCT:.0f}% a day to time decay "
               f"or with India VIX above {RULE_MAX_VIX:.0f}. Trading stops if the account reaches zero. "
               "Careful: choosing the threshold after seeing these results is optimistic; in live use it must be "
               "fixed in advance on older data.")

    # --- 3. Why this prediction? ---
    st.divider()
    st.subheader("Why this prediction?")
    probs = rf["test_probabilities"]
    picks = {"The trade the model liked most": int(np.argmax(probs)),
             "The trade the model liked least": int(np.argmin(probs)),
             "A borderline trade (closest to your threshold)": int(np.argmin(np.abs(probs - threshold)))}
    pick_label = st.selectbox("Explain", list(picks), key="ml_explain_pick")
    pick = picks[pick_label]
    pick_row = test_frame.iloc[[pick]]
    bias, contrib = explain_forest_prediction(rf["fitted_model"], pick_row[FEATURE_NAMES])
    p = float(probs[pick])
    if is_real:
        tr = pick_row.iloc[0]
        outcome = (f"Bought {tr['option_type']} {tr['strike']:,.0f} on {tr['entry_date']:%d %b %Y} with Nifty at "
                   f"{tr['spot']:,.0f}, {tr['days_to_expiry']:.0f} days to expiry, for ₹{tr['premium']:,.1f}. "
                   f"Real result at expiry: {'+' if tr['pnl_inr'] >= 0 else '−'}₹{abs(tr['pnl_inr']):,.0f} per lot.")
    else:
        outcome = f"Simulated trade: it {'won' if pick_row['is_profitable'].iloc[0] == 1 else 'lost'}."
    st.markdown(f"""
    <div class="clean-card">
        <div style="font-size: 1.05rem;">Model's chance of profit: <b class="{'val-green' if p >= threshold else 'val-red'}">{p:.0%}</b>
        → {'TAKE' if p >= threshold else 'DROP'} at your {threshold:.0%} threshold</div>
        <div style="font-size: 0.85rem; color: #A39E93; margin-top: 4px;">{outcome}</div>
    </div>""", unsafe_allow_html=True)
    FEATURE_TEXT = {"delta": "Delta", "gamma": "Gamma", "theta_burn_pct": "Time decay %/day", "vix": "India VIX",
                    "days_to_expiry": "Days to expiry", "moneyness": "Moneyness"}
    order = sorted(contrib, key=lambda f: abs(contrib[f]))
    fig_ex = go.Figure(go.Bar(
        x=[contrib[f] * 100 for f in order],
        y=[f"{FEATURE_TEXT[f]} = {pick_row[f].iloc[0]:.3g}" for f in order], orientation="h",
        marker_color=["#5FD39A" if contrib[f] >= 0 else "#FF8A73" for f in order],
        text=[f"{contrib[f] * 100:+.1f} pts" for f in order], textposition="outside",
        hovertemplate="%{y}: %{x:+.1f} percentage points<extra></extra>"))
    fig_ex.update_layout(template="plotly_dark", paper_bgcolor="#121212", plot_bgcolor="#1E1D1B", height=330,
                         margin=dict(l=10, r=40, t=10, b=40),
                         xaxis_title=f"Push on the chance of profit (percentage points, from a {bias:.0%} average)")
    st.plotly_chart(fig_ex, width="stretch")
    st.caption(f"Each bar is how much that feature moved this trade's chance of profit away from the average "
               f"trade ({bias:.0%}); green pushes toward TAKE, coral toward DROP. The bars add up exactly to {p:.0%}.")

# Simple Footer
st.markdown("---")
st.caption("sentinel. · Predictive Drawdown Containment System · B.Tech Minor Project")
