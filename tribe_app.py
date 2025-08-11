# streamlit_app.py

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
tickers_input = st.sidebar.text_input("Tickers (Comma-separated)", value="BABA,QBTS,VOW3.DE,INTC")
TICKERS = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

START_DATE = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
END_DATE   = st.sidebar.date_input("End Date", value=pd.to_datetime(datetime.now(LOCAL_TZ).date()))

LOOKBACK = st.sidebar.number_input("Lookback (Tage)", min_value=10, max_value=252, value=60, step=5)
HORIZON  = st.sidebar.number_input("Horizon (Tage)",  min_value=1, max_value=10, value=2)
THRESH   = st.sidebar.number_input("Threshold für Target", min_value=0.0, max_value=0.1, value=0.02, step=0.005, format="%.3f")

ENTRY_PROB = st.sidebar.slider("Entry Threshold (P(Signal))", min_value=0.0, max_value=1.0, value=0.63, step=0.01)
EXIT_PROB  = st.sidebar.slider("Exit Threshold (P(Signal))",  min_value=0.0, max_value=1.0, value=0.46, step=0.01)

COMMISSION    = st.sidebar.number_input("Commission (ad valorem, z.B. 0.001=10bp)", min_value=0.0, max_value=0.02, value=0.004, step=0.0001, format="%.4f")
SLIPPAGE_BPS  = st.sidebar.number_input("Slippage (bp je Ausführung)", min_value=0, max_value=50, value=5, step=1)
POS_FRAC      = st.sidebar.slider("Max. Positionsgröße (% des Kapitals)", min_value=0.1, max_value=1.0, value=1.0, step=0.1)

# Positionssizing: fixed vs. ATR-risk
sizing_mode = st.sidebar.selectbox("Positionsgröße", ["Fixed Fraction", "ATR Risk %"], index=0)
atr_lookback = st.sidebar.number_input("ATR Lookback (Tage)", min_value=5, max_value=100, value=14, step=1)
atr_k        = st.sidebar.number_input("ATR-Multiplikator (Stop-Distanz in ATR)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
risk_per_trade_pct = st.sidebar.number_input("Risiko pro Trade (% vom Kapital)", min_value=0.1, max_value=5.0, value=1.0, step=0.1) / 100.0

INIT_CAP = st.sidebar.number_input("Initial Capital  (€)", min_value=1000.0, value=10_000.0, step=1000.0, format="%.2f")

# Intraday-Tail Optionen
use_live = st.sidebar.checkbox("Letzten Tag intraday aggregieren (falls verfügbar)", value=True)
intraday_interval = st.sidebar.selectbox("Intraday-Intervall (Tail)", ["1m", "2m", "5m"], index=0)
fallback_last_session = st.sidebar.checkbox("Fallback: letzte Session verwenden (wenn heute leer)", value=False)
highlight_intraday_tail = st.sidebar.checkbox("„Heute (intraday)“ im Chart hervorheben", value=True)

exec_mode = st.sidebar.selectbox(
    "Execution Mode",
    ["Next Open (backtest+live)", "Market-On-Close (live only)"]
)
moc_cutoff_min = st.sidebar.number_input("MOC Cutoff (Minuten vor Close, nur live)", min_value=5, max_value=60, value=15, step=5)

st.sidebar.markdown("**Modellparameter**")
n_estimators  = st.sidebar.number_input("n_estimators",  min_value=10, max_value=500, value=100, step=10)
learning_rate = st.sidebar.number_input("learning_rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f")
max_depth     = st.sidebar.number_input("max_depth",     min_value=1, max_value=10, value=3, step=1)

MODEL_PARAMS = dict(
    n_estimators=int(n_estimators),
    learning_rate=float(learning_rate),
    max_depth=int(max_depth),
    random_state=42
)

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
    interval: str = "1m",
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

    # Heutiger Intraday-Snapshot
    try:
        intraday = tk.history(period="1d", interval=interval, auto_adjust=True, actions=False)
        if not intraday.empty:
            if intraday.index.tz is None:
                intraday.index = intraday.index.tz_localize("UTC")
            intraday.index = intraday.index.tz_convert(LOCAL_TZ)
        else:
            intraday = pd.DataFrame()
    except Exception:
        intraday = pd.DataFrame()

    # optional: MOC-Cutoff
    if exec_mode_key.startswith("Market-On-Close") and not intraday.empty:
        now_local = datetime.now(LOCAL_TZ)
        cutoff_time = now_local - timedelta(minutes=int(moc_cutoff_min_val))
        intraday = intraday.loc[:cutoff_time]

    # Fallback: letzte Session (z. B. Wochenende)
    if intraday.empty and fallback_last_session:
        try:
            intraday5 = tk.history(period="5d", interval=interval, auto_adjust=True, actions=False)
            if not intraday5.empty:
                if intraday5.index.tz is None:
                    intraday5.index = intraday5.index.tz_localize("UTC")
                intraday5.index = intraday5.index.tz_convert(LOCAL_TZ)
                last_session_date = intraday5.index[-1].date()
                intraday = intraday5.loc[str(last_session_date)]
        except Exception:
            pass

    if not intraday.empty:
        # Aggregation nur für heutigen/letzten Kalendertag
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

    # ATR (auf Daily) und auf Feature-Index ausrichten
    atr_series = add_atr(df, n=atr_lookback).reindex(feat.index)
    feat["ATR"] = atr_series

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
# Backtester: Signal t -> Ausführung Open t+1 (mit Slippage, PosSize)
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
    """
    Erwartet df mit Spalten: ['Open','Close','High','Low','SignalProb', 'ATR', ...]
    Ausführung: Signal an Tag t => Trade am Open von t+1.
    Equity-Bewertung: Tagesende (Close).
    Kein Pyramiding (0/1-Position).
    """
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

    equity_gross, equity_net, trades = [], [], []
    cum_pl_net = 0.0

    for i in range(n):
        # 1) Orderausführung am heutigen Open (Signal von gestern)
        if i > 0:
            open_today = float(df["Open"].iloc[i])
            slip_buy   = open_today * (1 + slippage_bps / 10000.0)
            slip_sell  = open_today * (1 - slippage_bps / 10000.0)
            prob_prev  = float(df["SignalProb"].iloc[i-1])
            date_exec  = df.index[i]

            if (not in_pos) and prob_prev > entry_thr:
                if sizing_mode == "ATR Risk %":
                    atr_prev = df["ATR"].iloc[i-1] if "ATR" in df.columns else np.nan
                    if not np.isfinite(atr_prev) or atr_prev <= 0:
                        # Fallback: fixed fraction
                        invest_net = cash_net * pos_frac
                        fee_entry  = invest_net * commission
                        target_shares = max((invest_net - fee_entry) / slip_buy, 0.0)
                        cost        = target_shares * slip_buy
                    else:
                        risk_budget      = risk_per_trade_pct * cash_net
                        risk_per_share   = max(atr_prev * atr_k, 1e-8)
                        shares_by_risk   = risk_budget / risk_per_share
                        shares_by_cash   = (cash_net * pos_frac) / slip_buy
                        target_shares    = max(min(shares_by_risk, shares_by_cash), 0.0)
                        cost             = target_shares * slip_buy
                        fee_entry        = cost * commission
                else:
                    # Fixed Fraction
                    invest_net   = cash_net * pos_frac
                    fee_entry    = invest_net * commission
                    target_shares = max((invest_net - fee_entry) / slip_buy, 0.0)
                    cost         = target_shares * slip_buy

                if target_shares > 0 and cost + fee_entry <= cash_net + 1e-9:
                    shares = target_shares
                    cost_basis_gross = cost
                    cost_basis_net   = cost + fee_entry
                    cash_gross -= cost
                    cash_net   -= (cost + fee_entry)
                    in_pos = True
                    trades.append({
                        "Date": date_exec, "Typ": "Entry", "Price": round(slip_buy, 4),
                        "Shares": round(shares, 4), "Gross P&L": 0.0,
                        "Fees": round(fee_entry, 2), "Net P&L": 0.0, "kum P&L": round(cum_pl_net, 2)
                    })

            elif in_pos and prob_prev < exit_thr:
                gross_value = shares * slip_sell
                fee_exit    = gross_value * commission
                pnl_gross   = gross_value - cost_basis_gross
                pnl_net     = (gross_value - fee_exit) - cost_basis_net

                cash_gross += gross_value
                cash_net   += (gross_value - fee_exit)

                in_pos = False
                shares = 0.0
                cost_basis_gross = 0.0
                cost_basis_net   = 0.0

                cum_pl_net += pnl_net
                trades.append({
                    "Date": date_exec, "Typ": "Exit", "Price": round(slip_sell, 4),
                    "Shares": 0.0, "Gross P&L": round(pnl_gross, 2),
                    "Fees": round(fee_exit, 2), "Net P&L": round(pnl_net, 2), "kum P&L": round(cum_pl_net, 2)
                })

        # 2) Tagesende-Bewertung (Close)
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

# Round-Trips (Entry→Exit) mit Haltedauer etc.
def compute_round_trips(all_trades: Dict[str, List[dict]]) -> pd.DataFrame:
    rows = []
    for tk, tr in all_trades.items():
        name = get_ticker_name(tk)
        current_entry = None
        for ev in tr:
            typ = ev.get("Typ")
            if typ == "Entry":
                current_entry = ev
            elif typ == "Exit" and current_entry is not None:
                entry_date  = pd.to_datetime(current_entry["Date"])
                exit_date   = pd.to_datetime(ev["Date"])
                shares      = float(current_entry.get("Shares", 0.0))
                entry_price = float(current_entry.get("Price", np.nan))
                exit_price  = float(ev.get("Price", np.nan))
                fee_entry   = float(current_entry.get("Fees", 0.0))
                fee_exit    = float(ev.get("Fees", 0.0))
                pnl_net     = float(ev.get("Net P&L", 0.0))
                cost_net    = shares * entry_price + fee_entry
                ret_pct     = (pnl_net / cost_net * 100.0) if cost_net else np.nan
                hold_days   = (exit_date - entry_date).days
                rows.append({
                    "Ticker": tk,
                    "Name": name,
                    "Entry Date": entry_date.strftime("%Y-%m-%d"),
                    "Exit Date": exit_date.strftime("%Y-%m-%d"),
                    "Hold (days)": hold_days,
                    "Shares": round(shares, 4),
                    "Entry Price": round(entry_price, 4),
                    "Exit Price": round(exit_price, 4),
                    "PnL Net (€)": round(pnl_net, 2),
                    "Fees (€)": round(fee_entry + fee_exit, 2),
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
                ticker,
                years=2,
                use_tail=use_live,
                interval=intraday_interval,
                fallback_last_session=fallback_last_session,
                exec_mode_key=exec_mode,
                moc_cutoff_min_val=moc_cutoff_min
            )
            # Auf UI-Zeitraum beschränken
            df = df_full.loc[str(START_DATE):str(END_DATE)].copy()
            last_timestamp_info(df, meta)

            # Trainieren + Backtest (Next Open) ohne Leakage (+ATR)
            feat, df_bt, trades, metrics = train_and_signal_no_leak(
                df, LOOKBACK, HORIZON, THRESH, MODEL_PARAMS, atr_lookback
            )
            metrics["Ticker"] = ticker
            results.append(metrics)
            all_trades[ticker] = trades
            all_dfs[ticker] = df_bt
            all_feat[ticker] = feat

            # Kennzahlen
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Strategie Netto (%)", f"{metrics['Strategy Net (%)']:.2f}")
            col2.metric("Buy & Hold (%)",      f"{metrics['Buy & Hold Net (%)']:.2f}")
            col3.metric("Sharpe",               f"{metrics['Sharpe-Ratio']:.2f}")
            col4.metric("Max Drawdown (%)",     f"{metrics['Max Drawdown (%)']:.2f}")

            # Hinweis bei MOC-Modus
            if exec_mode.startswith("Market-On-Close"):
                st.info("MOC-Modus: Live-Signal unten. Backtest-/Kennzahlen oben basieren auf 'Next Open' (robust & ohne Leakage).")

            # Preis + Signal
            df_plot = feat.copy()
            price_fig = go.Figure()
            price_fig.add_trace(
                go.Scatter(
                    x=df_plot.index, y=df_plot["Close"], mode="lines", name="Close",
                    line=dict(color="rgba(0,0,0,0.4)", width=1),
                    hovertemplate="Datum: %{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>"
                )
            )
            signal_probs = df_plot["SignalProb"]
            norm = (signal_probs - signal_probs.min()) / (signal_probs.max() - signal_probs.min() + 1e-9)
            colorscale = px.colors.diverging.RdYlGn
            for i in range(len(df_plot) - 1):
                seg_x = df_plot.index[i : i + 2]
                seg_y = df_plot["Close"].iloc[i : i + 2]
                prob = norm.iloc[i]
                color_seg = px.colors.sample_colorscale(colorscale, prob)[0]
                price_fig.add_trace(
                    go.Scatter(x=seg_x, y=seg_y, mode="lines", showlegend=False, line=dict(color=color_seg, width=2), hoverinfo="skip")
                )

            # Entry/Exit Marker
            trades_df = pd.DataFrame(trades)
            if not trades_df.empty:
                trades_df["Date"] = pd.to_datetime(trades_df["Date"])
                entries = trades_df[trades_df["Typ"] == "Entry"]
                exits   = trades_df[trades_df["Typ"] == "Exit"]
                price_fig.add_trace(
                    go.Scatter(
                        x=entries["Date"], y=entries["Price"], mode="markers",
                        marker_symbol="triangle-up", marker=dict(size=12, color="green"),
                        name="Entry", hovertemplate="Entry<br>Datum: %{x|%Y-%m-%d}<br>Preis: %{y:.2f}<extra></extra>"
                    )
                )
                price_fig.add_trace(
                    go.Scatter(
                        x=exits["Date"], y=exits["Price"], mode="markers",
                        marker_symbol="triangle-down", marker=dict(size=12, color="red"),
                        name="Exit", hovertemplate="Exit<br>Datum: %{x|%Y-%m-%d}<br>Preis: %{y:.2f}<extra></extra>"
                    )
                )

            # Highlight + Badge „heute (intraday)“
            if meta.get("tail_is_intraday") and highlight_intraday_tail:
                last_ts = df_plot.index[-1]
                last_date = last_ts.date()
                x0 = pd.Timestamp(last_date).strftime("%Y-%m-%d")
                x1 = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                price_fig.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor="LightSteelBlue", opacity=0.20, line_width=0,
                    annotation_text=(f"heute (intraday) – bis {meta['tail_ts'].strftime('%H:%M %Z')}"
                                     if meta.get("tail_ts") else "heute (intraday)"),
                    annotation_position="top left"
                )

            price_fig.update_layout(
                title=f"{ticker}: Preis mit Signal-Wahrscheinlichkeit",
                xaxis_title="Datum", yaxis_title="Preis",
                height=400, margin=dict(t=50, b=30, l=40, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(price_fig, use_container_width=True)

            # Equity-Kurve
            equity_fig = go.Figure()
            equity_fig.add_trace(
                go.Scatter(
                    x=df_bt.index, y=df_bt["Equity_Net"], name="Strategy Net Equity (Next Open)", mode="lines",
                    hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}€<extra></extra>"
                )
            )
            bh_curve = INIT_CAP * df_bt["Close"] / df_bt["Close"].iloc[0]
            equity_fig.add_trace(
                go.Scatter(
                    x=df_bt.index, y=bh_curve, name="Buy & Hold", mode="lines",
                    line=dict(dash="dash", color="black"),
                    hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}€<extra></extra>"
                )
            )
            equity_fig.update_layout(
                title=f"{ticker}: Net Equity-Kurve vs. Buy & Hold",
                xaxis_title="Datum", yaxis_title="Equity (€)",
                height=400, margin=dict(t=50, b=30, l=40, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(equity_fig, use_container_width=True)

            # Trades Tabelle (pro Ticker) – mit Name
            with st.expander(f"Trades (Next Open) für {ticker}", expanded=False):
                if not trades_df.empty:
                    df_tr = trades_df.copy()
                    df_tr["Ticker"] = ticker
                    df_tr["Name"] = get_ticker_name(ticker)
                    df_tr["Date"] = pd.to_datetime(df_tr["Date"])
                    df_tr["DateStr"] = df_tr["Date"].dt.strftime("%Y-%m-%d")
                    df_tr["CumPnL"] = df_tr.where(df_tr["Typ"] == "Exit")["Net P&L"].cumsum().fillna(method="ffill").fillna(0)
                    df_tr = df_tr.rename(columns={"Net P&L": "PnL"})
                    display_cols = ["Ticker", "Name", "DateStr", "Typ", "Price", "Shares", "PnL", "CumPnL", "Fees"]
                    df_disp = df_tr[display_cols].rename(columns={"DateStr": "Date"})
                    styled_trades = df_disp.style.format({
                        "Price": "{:.2f}", "Shares": "{:.4f}", "PnL": "{:.2f}",
                        "CumPnL": "{:.2f}", "Fees": "{:.2f}",
                    })
                    show_styled_or_plain(df_disp, styled_trades)
                    st.download_button(
                        label=f"Trades {ticker} als CSV",
                        data=df_tr[["Ticker","Name","Date","Typ","Price","Shares","PnL","CumPnL","Fees"]]
                            .to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8"),
                        file_name=f"trades_{ticker}.csv", mime="text/csv"
                    )
                else:
                    st.info("Keine Trades vorhanden.")

            # MOC-Live-Vorschau
            if exec_mode.startswith("Market-On-Close"):
                live_prob = float(feat["SignalProb"].iloc[-1])
                st.subheader("🕒 MOC Live-Vorschau")
                c1, c2, c3 = st.columns(3)
                c1.metric("Heutige P(Signal)", f"{live_prob:.4f}")
                c2.metric("Entry-Threshold",  f"{ENTRY_PROB:.2f}")
                c3.metric("Exit-Threshold",   f"{EXIT_PROB:.2f}")

                if live_prob > ENTRY_PROB:
                    st.success("📥 Würde heute eine **MOC-Entry-Order** einreichen (Ausführung zum offiziellen Close).")
                elif live_prob < EXIT_PROB:
                    st.warning("📤 Würde heute eine **MOC-Exit-Order** einreichen (Ausführung zum offiziellen Close).")
                else:
                    st.info("⏳ Keine MOC-Order heute (zwischen Entry- und Exit-Schwelle).")

                st.caption("Hinweis: Für echten MOC-Backtest wären historische Intraday-Daten notwendig. "
                           "Die obigen Kennzahlen stammen aus dem robusten 'Next Open'-Backtest.")

        except Exception as e:
            st.error(f"Fehler bei {ticker}: {e}")

# ─────────────────────────────────────────────────────────────
# Summary / Open Positions / Kombinierte Trades + Filter / Round-Trips
# ─────────────────────────────────────────────────────────────
if results:
    summary_df = pd.DataFrame(results).set_index("Ticker")
    summary_df["Net P&L (%)"] = (summary_df["Net P&L (€)"] / INIT_CAP) * 100

    total_net_pnl  = summary_df["Net P&L (€)"].sum()
    total_fees     = summary_df["Fees (€)"].sum()
    total_gross_pnl = total_net_pnl + total_fees
    total_trades   = summary_df["Number of Trades"].sum()
    total_capital  = INIT_CAP * len(summary_df)
    total_net_return_pct   = total_net_pnl / total_capital * 100
    total_gross_return_pct = total_gross_pnl / total_capital * 100

    st.subheader("📊 Summary of all Tickers (Next Open Backtest)")
    cols = st.columns(4)
    cols[0].metric("Cumulative Net P&L (€)",  f"{total_net_pnl:,.2f}")
    cols[1].metric("Cumulative Trading Costs (€)", f"{total_fees:,.2f}")
    cols[2].metric("Cumulative Gross P&L (€)", f"{total_gross_pnl:,.2f}")
    cols[3].metric("Total Number of Trades",   f"{int(total_trades)}")
    st.markdown(
        f"**Total Net Return (%):** {total_net_return_pct:.2f}  \n"
        f"**Total Gross Return (%):** {total_gross_return_pct:.2f}"
    )

    def color_phase_html(val):
        colors = {"Open": "#d0ebff", "Flat": "#f0f0f0"}
        bg = colors.get(val, "#ffffff")
        return f"background-color: {bg};"

    styled = (
        summary_df.style
        .format({
            "Strategy Net (%)": "{:.2f}", "Strategy Gross (%)": "{:.2f}",
            "Buy & Hold Net (%)": "{:.2f}", "Volatility (%)": "{:.2f}",
            "Sharpe-Ratio": "{:.2f}", "Max Drawdown (%)": "{:.2f}",
            "Calmar-Ratio": "{:.2f}", "Fees (€)": "{:.2f}",
            "Net P&L (%)": "{:.2f}", "Net P&L (€)": "{:.2f}"
        })
        .applymap(lambda v: "font-weight: bold;" if isinstance(v, (int, float)) else "", subset=pd.IndexSlice[:, ["Sharpe-Ratio"]])
        .applymap(color_phase_html, subset=["Phase"])
        .set_caption("Strategy-Performance per Ticker (Next Open Execution)")
    )
    show_styled_or_plain(summary_df, styled)
    st.download_button(
        "Summary als CSV herunterladen",
        summary_df.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="strategy_summary.csv", mime="text/csv"
    )

    # Open Positions
    st.subheader("📋 Open Positions (Next Open Backtest)")
    open_positions = []
    for ticker, trades in all_trades.items():
        if trades and trades[-1]["Typ"] == "Entry":
            last_entry = next(t for t in reversed(trades) if t["Typ"] == "Entry")
            prob = all_feat[ticker]["SignalProb"].iloc[-1]
            entry_ts = pd.to_datetime(last_entry["Date"])
            open_positions.append({
                "Ticker": ticker,
                "Name": get_ticker_name(ticker),
                "Entry Date": entry_ts,
                "Entry Price": round(last_entry["Price"], 2),
                "Current Prob.": round(float(prob), 4),
            })
    if open_positions:
        open_df = pd.DataFrame(open_positions)
        sort_order = st.radio("Sortierung nach Eröffnungsdatum", options=["Neueste zuerst", "Älteste zuerst"], horizontal=True, key="sort_open_positions")
        ascending = (sort_order == "Älteste zuerst")
        open_df = open_df.sort_values("Entry Date", ascending=ascending, kind="mergesort")
        open_df_display = open_df.copy()
        open_df_display["Entry Date"] = open_df_display["Entry Date"].dt.strftime("%Y-%m-%d")
        styled_open = open_df_display.style.format({"Entry Price": "{:.2f}", "Current Prob.": "{:.4f}"})
        show_styled_or_plain(open_df_display, styled_open)
        st.download_button(
            "Offene Positionen als CSV",
            open_df.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8"),
            file_name="open_positions.csv", mime="text/csv"
        )
    else:
        st.success("Keine offenen Positionen.")

    # Kombinierte Events (Entries/Exits) + Filter
    if all_trades:
        combined = []
        for tk, tr in all_trades.items():
            tk_name = get_ticker_name(tk)
            for row in tr:
                r = dict(row)
                r["Ticker"] = tk
                r["Name"] = tk_name
                r["Date"] = pd.to_datetime(r["Date"])
                combined.append(r)
        if combined:
            st.subheader("📒 Alle Trades (Events) – Filter")
            comb_df = pd.DataFrame(combined)

            # Filter-UI
            min_d, max_d = comb_df["Date"].min().date(), comb_df["Date"].max().date()
            f_col1, f_col2, f_col3 = st.columns([1,1,2])
            with f_col1:
                type_sel = st.multiselect("Typ", options=["Entry","Exit"], default=["Entry","Exit"])
            with f_col2:
                date_range = st.date_input("Zeitraum", value=(min_d, max_d), min_value=min_d, max_value=max_d)
                if isinstance(date_range, tuple):
                    d_start, d_end = date_range
                else:
                    d_start, d_end = min_d, max_d
            with f_col3:
                st.write("")  # spacing
                st.write("")

            # Anwenden
            mask = comb_df["Typ"].isin(type_sel) & (comb_df["Date"].dt.date.between(d_start, d_end))
            comb_f = comb_df.loc[mask].copy()
            comb_f_disp = comb_f.copy()
            comb_f_disp["Date"] = comb_f_disp["Date"].dt.strftime("%Y-%m-%d")
            wanted = ["Ticker", "Name", "Date", "Typ", "Price", "Shares", "Net P&L", "kum P&L", "Fees"]
            cols_present = [c for c in wanted if c in comb_f_disp.columns]
            comb_f_disp = comb_f_disp[cols_present]
            styled_comb = comb_f_disp.style.format({"Price": "{:.2f}", "Shares": "{:.4f}", "Net P&L": "{:.2f}", "kum P&L": "{:.2f}", "Fees": "{:.2f}"})
            show_styled_or_plain(comb_f_disp, styled_comb)
            st.download_button(
                "Gefilterte Events als CSV",
                comb_f[cols_present].to_csv(index=False).encode("utf-8"),
                file_name="trades_events_filtered.csv", mime="text/csv"
            )

            # Round-Trips mit Haltedauer
            rt_df = compute_round_trips(all_trades)
            if not rt_df.empty:
                st.subheader("🔁 Abgeschlossene Trades (Round-Trips) – inkl. Haltedauer")
                styled_rt = rt_df.style.format({
                    "Entry Price": "{:.2f}", "Exit Price": "{:.2f}",
                    "PnL Net (€)": "{:.2f}", "Fees (€)": "{:.2f}",
                    "Return (%)": "{:.2f}"
                })
                show_styled_or_plain(rt_df, styled_rt)
                st.download_button(
                    "Round-Trips als CSV",
                    rt_df.to_csv(index=False).encode("utf-8"),
                    file_name="round_trips.csv", mime="text/csv"
                )
else:
    st.warning("Noch keine Ergebnisse verfügbar. Stelle sicher, dass mindestens ein Ticker korrekt eingegeben ist und genügend Daten vorhanden sind.")
