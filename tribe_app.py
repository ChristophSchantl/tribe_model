# streamlit_app.py
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*figure layout has changed to tight.*")

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from math import sqrt
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
from zoneinfo import ZoneInfo

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────
# Config / Globals
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Signal-basierte Strategie Backtest", layout="wide")
LOCAL_TZ = ZoneInfo("Europe/Zurich")

# ─────────────────────────────────────────────────────────────
# Sidebar / Parameter
# ─────────────────────────────────────────────────────────────
st.sidebar.header("Parameter")
tickers_input = st.sidebar.text_input("Tickers (Comma-separated)", value="BABA, RQ0.F, VOW3.DE, INTC, BIDU, 0700.HK, LUMN, 2318.HK")
TICKERS = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

START_DATE = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
END_DATE   = st.sidebar.date_input("End Date", value=pd.to_datetime(datetime.now(LOCAL_TZ).date()))

LOOKBACK = st.sidebar.number_input("Lookback (Tage)", min_value=10, max_value=252, value=60, step=5)
HORIZON  = st.sidebar.number_input("Horizon (Tage)", min_value=1, max_value=10, value=2)
THRESH   = st.sidebar.number_input("Threshold für Target", min_value=0.0, max_value=0.1, value=0.02, step=0.005, format="%.3f")

ENTRY_PROB = st.sidebar.slider("Entry Threshold (P(Signal))", min_value=0.0, max_value=1.0, value=0.63, step=0.01)
EXIT_PROB  = st.sidebar.slider("Exit Threshold (P(Signal))",  min_value=0.0, max_value=1.0, value=0.46, step=0.01)

COMMISSION   = st.sidebar.number_input("Commission (ad valorem, z.B. 0.001=10bp)", min_value=0.0, max_value=0.02, value=0.004, step=0.0001, format="%.4f")
SLIPPAGE_BPS = st.sidebar.number_input("Slippage (bp je Ausführung)", min_value=0, max_value=50, value=5, step=1)
POS_FRAC     = st.sidebar.slider("Max. Positionsgröße (% des Kapitals)", min_value=0.1, max_value=1.0, value=1.0, step=0.1)

# Positionssizing: fixed vs. ATR-risk
sizing_mode = st.sidebar.selectbox("Positionsgröße", ["Fixed Fraction", "ATR Risk %"], index=0)
atr_lookback = st.sidebar.number_input("ATR Lookback (Tage)", min_value=5, max_value=100, value=14, step=1)
atr_k        = st.sidebar.number_input("ATR-Multiplikator (Stop-Distanz in ATR)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
risk_per_trade_pct = st.sidebar.number_input("Risiko pro Trade (% vom Kapital)", min_value=0.1, max_value=5.0, value=1.0, step=0.1) / 100.0

INIT_CAP = st.sidebar.number_input("Initial Capital  (€)", min_value=1000.0, value=10_000.0, step=1000.0, format="%.2f")

# Intraday-Tail Optionen
use_live = st.sidebar.checkbox("Letzten Tag intraday aggregieren (falls verfügbar)", value=True)
intraday_interval = st.sidebar.selectbox("Intraday-Intervall (Tail & 5-Tage-Chart)", ["1m", "2m", "5m", "15m"], index=2)
fallback_last_session = st.sidebar.checkbox("Fallback: letzte Session verwenden (wenn heute leer)", value=False)

exec_mode = st.sidebar.selectbox("Execution Mode", ["Next Open (backtest+live)", "Market-On-Close (live only)"])
moc_cutoff_min = st.sidebar.number_input("MOC Cutoff (Minuten vor Close, nur live)", min_value=5, max_value=60, value=15, step=5)

# Neuer Umschalter: Intraday-Chart-Typ
intraday_chart_type = st.sidebar.selectbox(
    "Intraday-Chart",
    ["Candlestick (OHLC)", "Close-Linie"],
    index=0
)

st.sidebar.markdown("**Modellparameter**")
n_estimators  = st.sidebar.number_input("n_estimators",  min_value=10, max_value=500, value=100, step=10)
learning_rate = st.sidebar.number_input("learning_rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f")
max_depth     = st.sidebar.number_input("max_depth",     min_value=1, max_value=10, value=3, step=1)

MODEL_PARAMS = dict(n_estimators=int(n_estimators), learning_rate=float(learning_rate), max_depth=int(max_depth), random_state=42)

# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────
def show_styled_or_plain(df: pd.DataFrame, styler):
    try:
        html = getattr(styler, "to_html", None)
        if callable(html):
            st.markdown(html(), unsafe_allow_html=True)
        else:
            raise AttributeError("Der übergebene Styler hat keine to_html-Methode")
    except Exception as e:
        st.warning(f"Styled-Tabelle konnte nicht gerendert werden, zeige einfache Tabelle. ({e})")
        st.dataframe(df)

def slope(arr: np.ndarray) -> float:
    x = np.arange(len(arr))
    return np.polyfit(x, arr, 1)[0]

def last_timestamp_info(df: pd.DataFrame, meta: Optional[dict] = None):
    ts = df.index[-1]
    msg = f"Letzter Datenpunkt: {ts.strftime('%Y-%m-%d %H:%M %Z')}"
    if meta and meta.get("tail_is_intraday") and meta.get("tail_ts") is not None:
        msg += f" (intraday bis {meta['tail_ts'].strftime('%H:%M %Z')})"
    st.caption(msg)

# Name-Lookup (gecached)
@st.cache_data(show_spinner=False, ttl=24*60*60)
def get_ticker_name(ticker: str) -> str:
    try:
        tk = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.get_info()
        except Exception:
            info = getattr(tk, "info", {}) or {}
        for k in ("shortName", "longName", "displayName", "companyName", "name"):
            if k in info and info[k]:
                return str(info[k])
    except Exception:
        pass
    return ticker

# ATR
def add_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"]  - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    return atr

# ─────────────────────────────────────────────────────────────
# Daten: 1D-Historie + NUR-HEUTE Intraday-Tail
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=120)
def get_price_data_tail_intraday(
    ticker: str,
    years: int = 2,
    use_tail: bool = True,
    interval: str = "5m",
    fallback_last_session: bool = False,
    exec_mode_key: str = "Next Open (backtest+live)",
    moc_cutoff_min_val: int = 15,
) -> Tuple[pd.DataFrame, dict]:
    tk = yf.Ticker(ticker)

    df = tk.history(period=f"{years}y", interval="1d", auto_adjust=True, actions=False)
    if df.empty:
        raise ValueError(f"Keine Daten für {ticker}")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(LOCAL_TZ)

    meta = {"tail_is_intraday": False, "tail_ts": None}

    if not use_tail:
        df.dropna(subset=["High", "Low", "Close"], inplace=True)
        return df, meta

    try:
        intraday = tk.history(period="1d", interval=interval, auto_adjust=True, actions=False, prepost=False)
        if not intraday.empty:
            if intraday.index.tz is None:
                intraday.index = intraday.index.tz_localize("UTC")
            intraday.index = intraday.index.tz_convert(LOCAL_TZ)
        else:
            intraday = pd.DataFrame()
    except Exception:
        intraday = pd.DataFrame()

    if exec_mode_key.startswith("Market-On-Close") and not intraday.empty:
        now_local = datetime.now(LOCAL_TZ)
        cutoff_time = now_local - timedelta(minutes=int(moc_cutoff_min_val))
        intraday = intraday.loc[:cutoff_time]

    if intraday.empty and fallback_last_session:
        try:
            intraday5 = tk.history(period="5d", interval=interval, auto_adjust=True, actions=False, prepost=False)
            if not intraday5.empty:
                if intraday5.index.tz is None:
                    intraday5.index = intraday5.index.tz_localize("UTC")
                intraday5.index = intraday5.index.tz_convert(LOCAL_TZ)
                last_session_date = intraday5.index[-1].date()
                intraday = intraday5.loc[str(last_session_date)]
        except Exception:
            pass

    if not intraday.empty:
        last_bar = intraday.iloc[-1]
        day_key = pd.Timestamp(last_bar.name.date(), tz=LOCAL_TZ)
        daily_row = {
            "Open":   float(intraday["Open"].iloc[0]),
            "High":   float(intraday["High"].max()),
            "Low":    float(intraday["Low"].min()),
            "Close":  float(last_bar["Close"]),
            "Volume": float(intraday["Volume"].sum()),
        }
        df.loc[day_key] = daily_row
        df = df.sort_index()
        meta["tail_is_intraday"] = True
        meta["tail_ts"] = last_bar.name

    df.dropna(subset=["High", "Low", "Close"], inplace=True)
    return df, meta

# ─────────────────────────────────────────────────────────────
# Intraday-Fetch für letzten 5 Handelstage (für Neben-Chart)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=120)
def get_intraday_last_n_sessions(ticker: str, sessions: int = 5, days_buffer: int = 10, interval: str = "5m") -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    intr = tk.history(period=f"{days_buffer}d", interval=interval, auto_adjust=True, actions=False, prepost=False)
    if intr.empty:
        return intr
    if intr.index.tz is None:
        intr.index = intr.index.tz_localize("UTC")
    intr.index = intr.index.tz_convert(LOCAL_TZ)

    unique_dates = pd.Index(intr.index.normalize().unique())
    keep_dates = set(unique_dates[-sessions:])
    mask = intr.index.normalize().isin(keep_dates)
    return intr.loc[mask].copy()

# ─────────────────────────────────────────────────────────────
# Features & Training ohne Leakage
# ─────────────────────────────────────────────────────────────
def make_features(df: pd.DataFrame, lookback: int, horizon: int) -> pd.DataFrame:
    feat = df.copy()
    feat["Range"]     = feat["High"].rolling(lookback).max() - feat["Low"].rolling(lookback).min()
    feat["SlopeHigh"] = feat["High"].rolling(lookback).apply(slope, raw=True)
    feat["SlopeLow"]  = feat["Low"].rolling(lookback).apply(slope, raw=True)
    feat = feat.iloc[lookback-1:].copy()
    feat["FutureRet"] = feat["Close"].shift(-horizon) / feat["Close"] - 1
    return feat

@st.cache_data(show_spinner=False, ttl=120)
def train_and_signal_no_leak(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    threshold: float,
    model_params: dict,
    atr_lookback: int
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict], dict]:
    feat = make_features(df, lookback, horizon)
    feat["ATR"] = add_atr(df, n=atr_lookback).reindex(feat.index)

    hist = feat.iloc[:-1].dropna(subset=["FutureRet"]).copy()
    if len(hist) < 30:
        raise ValueError("Zu wenige Datenpunkte nach Preprocessing für das Modell.")

    hist["Target"] = (hist["FutureRet"] > threshold).astype(int)
    X_cols = ["Range","SlopeHigh","SlopeLow"]
    X_train, y_train = hist[X_cols].values, hist["Target"].values

    scaler = StandardScaler().fit(X_train)
    model  = GradientBoostingClassifier(**model_params).fit(scaler.transform(X_train), y_train)

    feat["SignalProb"] = model.predict_proba(scaler.transform(feat[X_cols].values))[:,1]
    feat_bt = feat.iloc[:-1].copy()  # letzte Zeile = live/out-of-sample

    df_bt, trades = backtest_next_open(
        feat_bt, ENTRY_PROB, EXIT_PROB, COMMISSION, SLIPPAGE_BPS,
        INIT_CAP, POS_FRAC, sizing_mode, risk_per_trade_pct, atr_k
    )
    metrics = compute_performance(df_bt, trades, INIT_CAP)
    return feat, df_bt, trades, metrics

# ─────────────────────────────────────────────────────────────
# Backtester: t-Signal → t+1 Open (incl. Prob & HoldDays)
# ─────────────────────────────────────────────────────────────
def backtest_next_open(
    df: pd.DataFrame,
    entry_thr: float,
    exit_thr: float,
    commission: float,
    slippage_bps: int,
    init_cap: float,
    pos_frac: float,
    sizing_mode: str,
    risk_per_trade_pct: float,
    atr_k: float,
) -> Tuple[pd.DataFrame, List[dict]]:
    df = df.copy()
    n = len(df)
    if n < 2:
        raise ValueError("Zu wenige Datenpunkte für Backtest.")

    cash_gross = init_cap
    cash_net   = init_cap
    shares     = 0.0
    in_pos     = False
    cost_basis_gross = 0.0
    cost_basis_net   = 0.0
    last_entry_date: Optional[pd.Timestamp] = None

    equity_gross, equity_net, trades = [], [], []
    cum_pl_net = 0.0

    for i in range(n):
        # --- Ausführung am Open von t (Signal von t-1) ---
        if i > 0:
            open_today = float(df["Open"].iloc[i])
            slip_buy  = open_today * (1 + slippage_bps / 10000.0)
            slip_sell = open_today * (1 - slippage_bps / 10000.0)
            prob_prev = float(df["SignalProb"].iloc[i-1])
            date_exec = df.index[i]

            if (not in_pos) and prob_prev > entry_thr:
                # Positionsgröße
                if sizing_mode == "ATR Risk %":
                    atr_prev = df.get("ATR", pd.Series(index=df.index)).iloc[i-1]
                    if np.isfinite(atr_prev) and atr_prev > 0:
                        risk_budget    = risk_per_trade_pct * cash_net
                        risk_per_share = max(atr_prev * atr_k, 1e-8)
                        shares_by_risk = risk_budget / risk_per_share
                        shares_by_cash = (cash_net * pos_frac) / slip_buy
                        target_shares  = max(min(shares_by_risk, shares_by_cash), 0.0)
                        fee_entry      = (target_shares * slip_buy) * commission
                        cost           = target_shares * slip_buy
                    else:
                        invest_net = cash_net * pos_frac
                        fee_entry  = invest_net * commission
                        cost       = invest_net - fee_entry
                        target_shares = max(cost / slip_buy, 0.0)
                else:
                    invest_net   = cash_net * pos_frac
                    fee_entry    = invest_net * commission
                    cost         = invest_net - fee_entry
                    target_shares = max(cost / slip_buy, 0.0)

                if target_shares > 0 and (target_shares * slip_buy + fee_entry) <= cash_net + 1e-9:
                    shares = target_shares
                    cost_basis_gross = shares * slip_buy
                    cost_basis_net   = shares * slip_buy + fee_entry
                    cash_gross -= cost_basis_gross
                    cash_net   -= cost_basis_net
                    in_pos = True
                    last_entry_date = date_exec
                    trades.append({
                        "Date": date_exec, "Typ": "Entry", "Price": round(slip_buy, 4),
                        "Shares": round(shares, 4), "Gross P&L": 0.0,
                        "Fees": round(fee_entry, 2), "Net P&L": 0.0,
                        "kum P&L": round(cum_pl_net, 2), "Prob": round(prob_prev, 4),
                        "HoldDays": np.nan
                    })

            elif in_pos and prob_prev < exit_thr:
                gross_value = shares * slip_sell
                fee_exit    = gross_value * commission
                pnl_gross   = gross_value - cost_basis_gross
                pnl_net     = (gross_value - fee_exit) - cost_basis_net

                cash_gross += gross_value
                cash_net   += (gross_value - fee_exit)

                hold_days = (date_exec - last_entry_date).days if last_entry_date is not None else np.nan
                in_pos = False
                shares = 0.0
                cost_basis_gross = 0.0
                cost_basis_net   = 0.0

                cum_pl_net += pnl_net
                trades.append({
                    "Date": date_exec, "Typ": "Exit", "Price": round(slip_sell, 4),
                    "Shares": 0.0, "Gross P&L": round(pnl_gross, 2),
                    "Fees": round(fee_exit, 2), "Net P&L": round(pnl_net, 2),
                    "kum P&L": round(cum_pl_net, 2), "Prob": round(prob_prev, 4),
                    "HoldDays": hold_days
                })
                last_entry_date = None

        # --- Bewertung am Tagesende (Close) ---
        close_today = float(df["Close"].iloc[i])
        equity_gross.append(cash_gross + (shares * close_today if in_pos else 0.0))
        equity_net.append(cash_net + (shares * close_today if in_pos else 0.0))

    df_bt = df.copy()
    df_bt["Equity_Gross"] = equity_gross
    df_bt["Equity_Net"]   = equity_net
    return df_bt, trades

# ─────────────────────────────────────────────────────────────
# Performance-Kennzahlen
# ─────────────────────────────────────────────────────────────
def compute_performance(df_bt: pd.DataFrame, trades: List[dict], init_cap: float) -> dict:
    net_ret = (df_bt["Equity_Net"].iloc[-1] / init_cap - 1) * 100
    rets = df_bt["Equity_Net"].pct_change().dropna()
    vol_ann = rets.std() * sqrt(252) * 100
    sharpe = (rets.mean() * 252) / (rets.std() * sqrt(252)) if rets.std() else np.nan
    dd = (df_bt["Equity_Net"] - df_bt["Equity_Net"].cummax()) / df_bt["Equity_Net"].cummax()
    max_dd = dd.min() * 100
    calmar = net_ret / abs(max_dd) if max_dd < 0 else np.nan
    gross_ret = (df_bt["Equity_Gross"].iloc[-1] / init_cap - 1) * 100
    bh_ret = (df_bt["Close"].iloc[-1] / df_bt["Close"].iloc[0] - 1) * 100
    fees = sum(t["Fees"] for t in trades)
    phase = "Open" if trades and trades[-1]["Typ"] == "Entry" else "Flat"
    completed = sum(1 for t in trades if t["Typ"] == "Exit")
    net_eur = df_bt["Equity_Net"].iloc[-1] - init_cap
    return {
        "Strategy Net (%)": round(net_ret, 2),
        "Strategy Gross (%)": round(gross_ret, 2),
        "Buy & Hold Net (%)": round(bh_ret, 2),
        "Volatility (%)": round(vol_ann, 2),
        "Sharpe-Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_dd, 2),
        "Calmar-Ratio": round(calmar, 2),
        "Fees (€)": round(fees, 2),
        "Phase": phase,
        "Number of Trades": completed,
        "Net P&L (€)": round(net_eur, 2),
    }

# Round-Trips (Entry→Exit) inkl. Haltedauer
def compute_round_trips(all_trades: Dict[str, List[dict]]) -> pd.DataFrame:
    rows = []
    for tk, tr in all_trades.items():
        name = get_ticker_name(tk)
        current_entry = None
        for ev in tr:
            if ev["Typ"] == "Entry":
                current_entry = ev
            elif ev["Typ"] == "Exit" and current_entry is not None:
                entry_date = pd.to_datetime(current_entry["Date"])
                exit_date  = pd.to_datetime(ev["Date"])
                hold_days  = (exit_date - entry_date).days
                shares     = float(current_entry.get("Shares", 0.0))
                entry_p    = float(current_entry.get("Price", np.nan))
                exit_p     = float(ev.get("Price", np.nan))
                fee_e      = float(current_entry.get("Fees", 0.0))
                fee_x      = float(ev.get("Fees", 0.0))
                pnl_net    = float(ev.get("Net P&L", 0.0))
                cost_net   = shares * entry_p + fee_e
                ret_pct    = (pnl_net / cost_net * 100.0) if cost_net else np.nan
                rows.append({
                    "Ticker": tk, "Name": name,
                    "Entry Date": entry_date, "Exit Date": exit_date,
                    "Hold (days)": hold_days,
                    "Entry Prob": current_entry.get("Prob", np.nan),
                    "Exit Prob":  ev.get("Prob", np.nan),
                    "Shares": round(shares, 4),
                    "Entry Price": round(entry_p, 4), "Exit Price": round(exit_p, 4),
                    "PnL Net (€)": round(pnl_net, 2), "Fees (€)": round(fee_e + fee_x, 2),
                    "Return (%)": round(ret_pct, 2),
                })
                current_entry = None
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# Haupt
# ─────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size: 36px;'>📈 AI Signal-based Trading-Strategy</h1>", unsafe_allow_html=True)

results = []
all_trades: Dict[str, List[dict]] = {}
all_dfs:   Dict[str, pd.DataFrame] = {}
all_feat:  Dict[str, pd.DataFrame] = {}

for ticker in TICKERS:
    with st.expander(f"🔍 Analyse für {ticker}", expanded=False):
        st.subheader(f"{ticker}")
        try:
            # 1D + nur-heute Intraday Tail
            df_full, meta = get_price_data_tail_intraday(
                ticker, years=2, use_tail=use_live,
                interval=intraday_interval, fallback_last_session=fallback_last_session,
                exec_mode_key=exec_mode, moc_cutoff_min_val=moc_cutoff_min
            )
            df = df_full.loc[str(START_DATE):str(END_DATE)].copy()
            last_timestamp_info(df, meta)

            # Trainieren + Backtest
            feat, df_bt, trades, metrics = train_and_signal_no_leak(
                df, LOOKBACK, HORIZON, THRESH, MODEL_PARAMS, atr_lookback
            )
            metrics["Ticker"] = ticker
            results.append(metrics)
            all_trades[ticker] = trades
            all_dfs[ticker] = df_bt
            all_feat[ticker] = feat

            # Kennzahlen + Trade-Zahl
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Strategie Netto (%)", f"{metrics['Strategy Net (%)']:.2f}")
            c2.metric("Buy & Hold (%)",      f"{metrics['Buy & Hold Net (%)']:.2f}")
            c3.metric("Sharpe",               f"{metrics['Sharpe-Ratio']:.2f}")
            c4.metric("Max Drawdown (%)",     f"{metrics['Max Drawdown (%)']:.2f}")
            c5.metric("Trades (Round-Trips)", f"{int(metrics['Number of Trades'])}")

            # Charts nebeneinander
            chart_cols = st.columns(2)

            # --- Daily Preis mit farbigen Segmenten + Events ---
            df_plot = feat.copy()
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["Close"], mode="lines", name="Close",
                line=dict(color="rgba(0,0,0,0.4)", width=1),
                hovertemplate="Datum: %{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>"
            ))
            signal_probs = df_plot["SignalProb"]
            norm = (signal_probs - signal_probs.min()) / (signal_probs.max() - signal_probs.min() + 1e-9)
            for i in range(len(df_plot) - 1):
                seg_x = df_plot.index[i:i+2]
                seg_y = df_plot["Close"].iloc[i:i+2]
                color_seg = px.colors.sample_colorscale(px.colors.diverging.RdYlGn, float(norm.iloc[i]))[0]
                price_fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode="lines", showlegend=False,
                                               line=dict(color=color_seg, width=2), hoverinfo="skip"))
            trades_df = pd.DataFrame(trades)
            if not trades_df.empty:
                trades_df["Date"] = pd.to_datetime(trades_df["Date"])
                entries = trades_df[trades_df["Typ"]=="Entry"]; exits = trades_df[trades_df["Typ"]=="Exit"]
                price_fig.add_trace(go.Scatter(
                    x=entries["Date"], y=entries["Price"], mode="markers", name="Entry",
                    marker_symbol="triangle-up", marker=dict(size=12, color="green"),
                    hovertemplate="Entry<br>Datum:%{x|%Y-%m-%d}<br>Preis:%{y:.2f}<extra></extra>"
                ))
                price_fig.add_trace(go.Scatter(
                    x=exits["Date"], y=exits["Price"], mode="markers", name="Exit",
                    marker_symbol="triangle-down", marker=dict(size=12, color="red"),
                    hovertemplate="Exit<br>Datum:%{x|%Y-%m-%d}<br>Preis:%{y:.2f}<extra></extra>"
                ))
            price_fig.update_layout(
                title=f"{ticker}: Preis mit Signal-Wahrscheinlichkeit (Daily)",
                xaxis_title="Datum", yaxis_title="Preis",
                height=420, margin=dict(t=50, b=30, l=40, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            with chart_cols[0]:
                st.plotly_chart(price_fig, use_container_width=True)

            # --- Intraday-Chart (letzte 5 Handelstage) mit Umschalter ---
            intra = get_intraday_last_n_sessions(ticker, sessions=5, days_buffer=10, interval=intraday_interval)
            with chart_cols[1]:
                if intra.empty:
                    st.info("Keine Intraday-Daten verfügbar (Ticker/Intervall/Zeitraum).")
                else:
                    intr_fig = go.Figure()

                    if intraday_chart_type == "Candlestick (OHLC)":
                        intr_fig.add_trace(
                            go.Candlestick(
                                x=intra.index,
                                open=intra["Open"], high=intra["High"],
                                low=intra["Low"],  close=intra["Close"],
                                name="OHLC (intraday)",
                                increasing_line_width=1, decreasing_line_width=1
                            )
                        )
                    else:
                        intr_fig.add_trace(
                            go.Scatter(
                                x=intra.index, y=intra["Close"],
                                mode="lines", name="Close (intraday)",
                                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Close: %{y:.2f}<extra></extra>"
                            )
                        )

                    # Events der letzten 5 Handelstage
                    if not trades_df.empty:
                        tdf = trades_df.copy()
                        tdf["Date"] = pd.to_datetime(tdf["Date"])
                        last_days = set(pd.Index(intra.index.normalize().unique()))
                        ev_recent = tdf[tdf["Date"].dt.normalize().isin(last_days)].copy()

                        for typ, color, symbol in [("Entry","green","triangle-up"), ("Exit","red","triangle-down")]:
                            xs, ys = [], []
                            for d, day_slice in intra.groupby(intra.index.normalize()):
                                hit = ev_recent[(ev_recent["Typ"] == typ) & (ev_recent["Date"].dt.normalize() == d)]
                                if hit.empty:
                                    continue
                                ts0 = day_slice.index.min()  # Session-Start
                                if intraday_chart_type == "Candlestick (OHLC)":
                                    y_val = float(hit["Price"].iloc[-1])           # Exec-Preis (Open inkl. Slippage)
                                else:
                                    y_val = float(day_slice["Close"].iloc[0])      # auf Linie snappen
                                xs.append(ts0); ys.append(y_val)
                            if xs:
                                intr_fig.add_trace(
                                    go.Scatter(
                                        x=xs, y=ys, mode="markers", name=typ,
                                        marker_symbol=symbol, marker=dict(size=11, color=color),
                                        hovertemplate=f"{typ}<br>%{{x|%Y-%m-%d %H:%M}}<br>Preis: %{{y:.2f}}<extra></extra>"
                                    )
                                )

                    intr_fig.update_layout(
                        title=f"{ticker}: Intraday – letzte 5 Handelstage ({intraday_interval})",
                        xaxis_title="Zeit", yaxis_title="Preis",
                        height=420, margin=dict(t=50, b=30, l=40, r=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    for _, day_slice in intra.groupby(intra.index.normalize()):
                        intr_fig.add_vline(x=day_slice.index.min(), line_width=1, line_dash="dot", opacity=0.3)

                    st.plotly_chart(intr_fig, use_container_width=True)

            # Equity-Kurve (unter den Charts)
            eq = go.Figure()
            eq.add_trace(go.Scatter(x=df_bt.index, y=df_bt["Equity_Net"], name="Strategy Net Equity (Next Open)",
                        mode="lines", hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}€<extra></extra>"))
            bh_curve = INIT_CAP * df_bt["Close"] / df_bt["Close"].iloc[0]
            eq.add_trace(go.Scatter(x=df_bt.index, y=bh_curve, name="Buy & Hold", mode="lines",
                                    line=dict(dash="dash", color="black")))
            eq.update_layout(title=f"{ticker}: Net Equity-Kurve vs. Buy & Hold", xaxis_title="Datum", yaxis_title="Equity (€)",
                             height=400, margin=dict(t=50, b=30, l=40, r=20),
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(eq, use_container_width=True)

            # Trades-Tabelle je Ticker
            with st.expander(f"Trades (Next Open) für {ticker}", expanded=False):
                if not trades_df.empty:
                    df_tr = trades_df.copy()
                    df_tr["Ticker"] = ticker
                    df_tr["Name"] = get_ticker_name(ticker)
                    df_tr["Date"] = pd.to_datetime(df_tr["Date"])
                    df_tr["DateStr"] = df_tr["Date"].dt.strftime("%Y-%m-%d")
                    df_tr["CumPnL"] = df_tr.where(df_tr["Typ"]=="Exit")["Net P&L"].cumsum().fillna(method="ffill").fillna(0)
                    df_tr = df_tr.rename(columns={"Net P&L":"PnL","Prob":"Signal Prob","HoldDays":"Hold (days)"})
                    disp_cols = ["Ticker","Name","DateStr","Typ","Price","Shares","Signal Prob","Hold (days)","PnL","CumPnL","Fees"]
                    styled = df_tr[disp_cols].rename(columns={"DateStr":"Date"}).style.format({
                        "Price":"{:.2f}","Shares":"{:.4f}","Signal Prob":"{:.4f}","PnL":"{:.2f}","CumPnL":"{:.2f}","Fees":"{:.2f}"
                    })
                    show_styled_or_plain(df_tr[disp_cols].rename(columns={"DateStr":"Date"}), styled)
                    st.download_button(
                        label=f"Trades {ticker} als CSV",
                        data=df_tr[["Ticker","Name","Date","Typ","Price","Shares","Signal Prob","Hold (days)","PnL","CumPnL","Fees"]]
                            .to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8"),
                        file_name=f"trades_{ticker}.csv", mime="text/csv"
                    )
                else:
                    st.info("Keine Trades vorhanden.")

        except Exception as e:
            st.error(f"Fehler bei {ticker}: {e}")

# ─────────────────────────────────────────────────────────────
# Summary / Open Positions / Events-Filter / Round-Trips
# ─────────────────────────────────────────────────────────────
if results:
    summary_df = pd.DataFrame(results).set_index("Ticker")
    summary_df["Net P&L (%)"] = (summary_df["Net P&L (€)"] / INIT_CAP) * 100

    total_net_pnl   = summary_df["Net P&L (€)"].sum()
    total_fees      = summary_df["Fees (€)"].sum()
    total_gross_pnl = total_net_pnl + total_fees
    total_trades    = summary_df["Number of Trades"].sum()
    total_capital   = INIT_CAP * len(summary_df)
    total_net_return_pct   = total_net_pnl / total_capital * 100
    total_gross_return_pct = total_gross_pnl / total_capital * 100

    st.subheader("📊 Summary of all Tickers (Next Open Backtest)")
    cols = st.columns(4)
    cols[0].metric("Cumulative Net P&L (€)",  f"{total_net_pnl:,.2f}")
    cols[1].metric("Cumulative Trading Costs (€)", f"{total_fees:,.2f}")
    cols[2].metric("Cumulative Gross P&L (€)", f"{total_gross_pnl:,.2f}")
    cols[3].metric("Total Number of Trades",   f"{int(total_trades)}")

    # Gesamtperformance (%)
    total_strategy_net_pct   = total_net_return_pct
    total_strategy_gross_pct = total_gross_return_pct
    bh_total_pct = float(summary_df["Buy & Hold Net (%)"].dropna().mean()) if "Buy & Hold Net (%)" in summary_df.columns else float("nan")

    cols_pct = st.columns(3)
    cols_pct[0].metric("Strategy Net (%) – total",   f"{total_strategy_net_pct:.2f}")
    cols_pct[1].metric("Strategy Gross (%) – total", f"{total_strategy_gross_pct:.2f}")
    cols_pct[2].metric("Buy & Hold Net (%) – total", f"{bh_total_pct:.2f}")

    def color_phase_html(val):
        colors = {"Open": "#d0ebff", "Flat": "#f0f0f0"}
        return f"background-color: {colors.get(val, '#ffffff')};"

    styled = (
        summary_df.style
        .format({
            "Strategy Net (%)":"{:.2f}","Strategy Gross (%)":"{:.2f}",
            "Buy & Hold Net (%)":"{:.2f}","Volatility (%)":"{:.2f}",
            "Sharpe-Ratio":"{:.2f}","Max Drawdown (%)":"{:.2f}",
            "Calmar-Ratio":"{:.2f}","Fees (€)":"{:.2f}",
            "Net P&L (%)":"{:.2f}","Net P&L (€)":"{:.2f}"
        })
        .applymap(lambda v: "font-weight: bold;" if isinstance(v,(int,float)) else "", subset=pd.IndexSlice[:,["Sharpe-Ratio"]])
        .applymap(color_phase_html, subset=["Phase"])
        .set_caption("Strategy-Performance per Ticker (Next Open Execution)")
    )
    show_styled_or_plain(summary_df, styled)
    st.download_button(
        "Summary als CSV herunterladen",
        summary_df.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="strategy_summary.csv", mime="text/csv"
    )

    # ─────────────────────────────────────────────────────────
    # Open Positions
    # ─────────────────────────────────────────────────────────
    st.subheader("📋 Open Positions (Next Open Backtest)")
    open_positions = []
    for ticker, trades in all_trades.items():
        if trades and trades[-1]["Typ"]=="Entry":
            last_entry = next(t for t in reversed(trades) if t["Typ"]=="Entry")
            prob = all_feat[ticker]["SignalProb"].iloc[-1]
            entry_ts = pd.to_datetime(last_entry["Date"])
            open_positions.append({
                "Ticker": ticker, "Name": get_ticker_name(ticker),
                "Entry Date": entry_ts, "Entry Price": round(last_entry["Price"],2),
                "Current Prob.": round(float(prob),4),
            })
    if open_positions:
        open_df = pd.DataFrame(open_positions).sort_values("Entry Date", ascending=False)
        open_df_display = open_df.copy()
        open_df_display["Entry Date"] = open_df_display["Entry Date"].dt.strftime("%Y-%m-%d")
        styled_open = open_df_display.style.format({"Entry Price":"{:.2f}","Current Prob.":"{:.4f}"})
        show_styled_or_plain(open_df_display, styled_open)
        st.download_button(
            "Offene Positionen als CSV",
            open_df.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8"),
            file_name="open_positions.csv", mime="text/csv"
        )
    else:
        st.success("Keine offenen Positionen.")

    # ─────────────────────────────────────────────────────────
    # Round-Trips (Entry → Exit) – mit Filter
    # ─────────────────────────────────────────────────────────
    rt_df = compute_round_trips(all_trades)

    if not rt_df.empty:
        st.subheader("🔁 Abgeschlossene Trades (Round-Trips) – Filter")

        # Basiswerte/Defaults robust bestimmen
        rt_df["Entry Date"] = pd.to_datetime(rt_df["Entry Date"])
        rt_df["Exit Date"]  = pd.to_datetime(rt_df["Exit Date"])
        # Safety: fehlende Spalten auffüllen
        for c in ["Entry Prob","Exit Prob","Return (%)","PnL Net (€)","Fees (€)","Hold (days)"]:
            if c not in rt_df.columns:
                rt_df[c] = np.nan

        r_min_d, r_max_d = rt_df["Entry Date"].min().date(), rt_df["Entry Date"].max().date()
        r_ticks = sorted(rt_df["Ticker"].unique().tolist())

        # Bereiche für Slider sicher bestimmen (mit sinnvollen Defaults)
        def finite_minmax(series, fallback=(0.0, 1.0)):
            s = pd.to_numeric(series, errors="coerce")
            lo, hi = float(np.nanmin(s.values)), float(np.nanmax(s.values))
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo, hi = fallback
            if lo == hi:
                lo, hi = fallback
            return lo, hi

        # UI-Filter
        r1, r2, r3 = st.columns([1.1, 1.1, 1.5])
        with r1:
            rt_tick_sel = st.multiselect("Ticker (Round-Trips)", options=r_ticks, default=r_ticks)
            hd_min, hd_max = int(np.nanmin(rt_df["Hold (days)"].values)), int(np.nanmax(rt_df["Hold (days)"].values))
            if not np.isfinite(hd_min): hd_min = 0
            if not np.isfinite(hd_max): hd_max = 60
            rt_hold = st.slider("Haltedauer (Tage)", min_value=int(hd_min), max_value=int(hd_max),
                                value=(int(hd_min), int(hd_max)), step=1, key="rt_hold")
        with r2:
            rt_date = st.date_input("Zeitraum (Entry-Datum)", value=(r_min_d, r_max_d),
                                    min_value=r_min_d, max_value=r_max_d, key="rt_date")
            ep_lo, ep_hi = finite_minmax(rt_df["Entry Prob"], (0.0, 1.0))
            xp_lo, xp_hi = finite_minmax(rt_df["Exit Prob"],  (0.0, 1.0))
            rt_ep = st.slider("Entry-Prob.", min_value=0.0, max_value=1.0,
                              value=(max(0.0, ep_lo), min(1.0, ep_hi)), step=0.01)
            rt_xp = st.slider("Exit-Prob.",  min_value=0.0, max_value=1.0,
                              value=(max(0.0, xp_lo), min(1.0, xp_hi)), step=0.01)
        with r3:
            ret_lo, ret_hi = finite_minmax(rt_df["Return (%)"], (-100.0, 200.0))
            pnl_lo, pnl_hi = finite_minmax(rt_df["PnL Net (€)"], (-INIT_CAP, INIT_CAP))
            rt_ret = st.slider("Return (%)", min_value=float(ret_lo), max_value=float(ret_hi),
                               value=(float(ret_lo), float(ret_hi)), step=0.5)
            rt_pnl = st.slider("PnL Net (€)", min_value=float(pnl_lo), max_value=float(pnl_hi),
                               value=(float(pnl_lo), float(pnl_hi)), step=10.0)

        rds, rde = (rt_date if isinstance(rt_date, tuple) else (r_min_d, r_max_d))

        # Mask anwenden
        mask_rt = (
            rt_df["Ticker"].isin(rt_tick_sel) &
            (rt_df["Entry Date"].dt.date.between(rds, rde)) &
            (rt_df["Hold (days)"].fillna(-1).between(rt_hold[0], rt_hold[1])) &
            (rt_df["Entry Prob"].fillna(0.0).between(rt_ep[0], rt_ep[1])) &
            (rt_df["Exit Prob"].fillna(0.0).between(rt_xp[0], rt_xp[1])) &
            (pd.to_numeric(rt_df["Return (%)"], errors="coerce").fillna(-9e9).between(rt_ret[0], rt_ret[1])) &
            (pd.to_numeric(rt_df["PnL Net (€)"], errors="coerce").fillna(-9e9).between(rt_pnl[0], rt_pnl[1]))
        )

        rt_f = rt_df.loc[mask_rt].copy()
        # Anzeige aufbereiten
        rt_f_disp = rt_f.copy()
        rt_f_disp["Entry Date"] = rt_f_disp["Entry Date"].dt.strftime("%Y-%m-%d")
        rt_f_disp["Exit Date"]  = rt_f_disp["Exit Date"].dt.strftime("%Y-%m-%d")

        # Integer-Format für Haltedauer ohne Komma
        if "Hold (days)" in rt_f_disp.columns:
            rt_f_disp["Hold (days)"] = rt_f_disp["Hold (days)"].round().astype("Int64")

        styled_rt = rt_f_disp.style.format({
            "Shares":"{:.4f}",
            "Entry Price":"{:.2f}","Exit Price":"{:.2f}",
            "PnL Net (€)":"{:.2f}","Fees (€)":"{:.2f}","Return (%)":"{:.2f}",
            "Entry Prob":"{:.4f}","Exit Prob":"{:.4f}"
        })
        show_styled_or_plain(rt_f_disp, styled_rt)

        st.download_button(
            "Round-Trips (gefiltert) als CSV",
            rt_f_disp.to_csv(index=False).encode("utf-8"),
            file_name="round_trips_filtered.csv", mime="text/csv"
        )
    else:
        st.info("Noch keine abgeschlossenen Round-Trips vorhanden.")
else:
    st.warning("Noch keine Ergebnisse verfügbar. Stelle sicher, dass mindestens ein Ticker korrekt eingegeben ist und genügend Daten vorhanden sind.")
