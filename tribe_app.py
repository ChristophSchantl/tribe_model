# streamlit_app.py
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*figure layout has changed to tight.*")

import sys
import subprocess
import importlib
from math import sqrt
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from zoneinfo import ZoneInfo

# ─────────────────────────────────────────────────────────────
# Ensure HTML parsers (lxml/html5lib/bs4) are available
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def ensure_html_parsers():
    need = {
        "lxml": "lxml",
        "html5lib": "html5lib",
        "beautifulsoup4": "bs4",
    }
    missing = []
    for pip_name, mod_name in need.items():
        try:
            importlib.import_module(mod_name)
        except Exception:
            missing.append(pip_name)

    if missing:
        st.info("Installiere Parser: " + ", ".join(missing))
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *missing]
            )
        except Exception as e:
            st.warning(f"Installation fehlgeschlagen: {e}")

    # final probe
    for mod_name in need.values():
        importlib.import_module(mod_name)

ensure_html_parsers()

# ─────────────────────────────────────────────────────────────
# Config / Globals
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Signal-based Trading-Strategy", layout="wide")
LOCAL_TZ = ZoneInfo("Europe/Zurich")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def show_styled_or_plain(df: pd.DataFrame, styler):
    try:
        html = getattr(styler, "to_html", None)
        if callable(html):
            st.markdown(html(), unsafe_allow_html=True)
        else:
            raise AttributeError("Styler has no to_html")
    except Exception as e:
        st.warning(f"Styled-Tabelle konnte nicht gerendert werden, zeige einfache Tabelle. ({e})")
        st.dataframe(df)

def slope(arr: np.ndarray) -> float:
    x = np.arange(len(arr))
    return float(np.polyfit(x, arr, 1)[0]) if len(arr) > 1 else 0.0

def normalize_yahoo_symbol(sym: str, suffix: Optional[str] = None) -> str:
    s = str(sym).strip().upper().replace(" ", "")
    s = s.replace(".", "-")
    if suffix and not s.endswith(suffix):
        s = s + suffix
    return s

def last_timestamp_info(df: pd.DataFrame, meta: Optional[dict] = None):
    ts = df.index[-1]
    msg = f"Letzter Datenpunkt: {ts.strftime('%Y-%m-%d %H:%M %Z')}"
    if meta and meta.get("tail_is_intraday") and meta.get("tail_ts") is not None:
        msg += f" (intraday bis {meta['tail_ts'].strftime('%H:%M %Z')})"
    st.caption(msg)

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

# ─────────────────────────────────────────────────────────────
# Index/Universe Loader (Wikipedia) + Fallback
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=6*60*60)
def fetch_sp500_symbols() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        df = next(t for t in tables if "Symbol" in t.columns)
        syms = (
            df["Symbol"].astype(str).str.upper()
            .str.replace(r"\.", "-", regex=True)
            .str.replace(r"\s+", "", regex=True)
            .tolist()
        )
        # Bekannte Sonderfälle korrigieren
        return [s.replace("BRK-B", "BRK-B").replace("BF-B", "BF-B") for s in syms]
    except Exception as e:
        st.warning(f"S&P 500 Liste konnte nicht geladen werden ({e}). Fallback wird genutzt.")
        # sehr kleiner Fallback
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "JPM", "XOM", "LLY"]

@st.cache_data(show_spinner=False, ttl=6*60*60)
def fetch_nasdaq100_symbols() -> List[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        tables = pd.read_html(url)
        # Tabelle mit Symbol/Ticker finden
        df = None
        for t in tables:
            if any(c.lower() in ["ticker", "symbol"] for c in map(str.lower, t.columns)):
                df = t
                break
        if df is None:
            raise ValueError("Keine passende Tabelle gefunden.")
        col = "Ticker" if "Ticker" in df.columns else "Symbol"
        syms = (
            df[col].astype(str).str.upper()
            .str.replace(r"\.", "-", regex=True)
            .str.replace(r"\s+", "", regex=True)
            .tolist()
        )
        return syms
    except Exception as e:
        st.warning(f"NASDAQ-100 Liste konnte nicht geladen werden ({e}). Fallback wird genutzt.")
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "COST", "PEP"]

@st.cache_data(show_spinner=False, ttl=6*60*60)
def fetch_dax40_symbols() -> List[str]:
    url = "https://en.wikipedia.org/wiki/DAX"
    try:
        tables = pd.read_html(url)
        df = None
        for t in tables:
            if any(c.lower() in ["ticker symbol", "ticker", "symbol"] for c in map(str.lower, t.columns)):
                df = t
                break
        if df is None:
            raise ValueError("Keine passende Tabelle gefunden.")
        col = next(c for c in df.columns if c.lower() in ["ticker symbol", "ticker", "symbol"])
        base = (
            df[col].astype(str).str.upper()
            .str.replace(r"\.", "-", regex=True)
            .str.replace(r"\s+", "", regex=True)
        )
        return (base + ".DE").tolist()
    except Exception as e:
        st.warning(f"DAX 40 Liste konnte nicht geladen werden ({e}). Fallback wird genutzt.")
        return [
            "SAP.DE","SIE.DE","ALV.DE","BAS.DE","BAYN.DE","BMW.DE","BEI.DE","CON.DE","DB1.DE","DBK.DE",
            "DTE.DE","EOAN.DE","FRE.DE","HEI.DE","HEN3.DE","IFX.DE","MRK.DE","MUV2.DE","P911.DE","RWE.DE",
            "SRT3.DE","VOW3.DE","PAH3.DE","BNR.DE","DTG.DE","MTX.DE","KNF.DE"
        ]

@st.cache_data(show_spinner=False, ttl=6*60*60)
def fetch_atx_symbols() -> List[str]:
    url = "https://en.wikipedia.org/wiki/ATX_(Austrian_Traded_Index)"
    try:
        tables = pd.read_html(url)
        df = None
        for t in tables:
            if any(c.lower() in ["ticker", "symbol"] for c in map(str.lower, t.columns)):
                df = t
                break
        if df is None:
            raise ValueError("Keine passende Tabelle gefunden.")
        col = "Ticker" if "Ticker" in df.columns else "Symbol"
        base = (
            df[col].astype(str).str.upper()
            .str.replace(r"\.", "-", regex=True)
            .str.replace(r"\s+", "", regex=True)
        )
        return (base + ".VI").tolist()
    except Exception as e:
        st.warning(f"ATX Liste konnte nicht geladen werden ({e}). Fallback wird genutzt.")
        return ["OMV.VI","EBS.VI","VIG.VI","VER.VI","ANDR.VI","RBI.VI","TEL.VI","VOE.VI","BSL.VI","IIA.VI"]

# ─────────────────────────────────────────────────────────────
# Data: Daily + Intraday tail (today only)
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
# Features / Model / Backtest
# ─────────────────────────────────────────────────────────────
def add_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"]  - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def make_features(df: pd.DataFrame, lookback: int, horizon: int) -> pd.DataFrame:
    feat = df.copy()
    feat["Range"]     = feat["High"].rolling(lookback).max() - feat["Low"].rolling(lookback).min()
    feat["SlopeHigh"] = feat["High"].rolling(lookback).apply(slope, raw=True)
    feat["SlopeLow"]  = feat["Low"].rolling(lookback).apply(slope, raw=True)
    feat = feat.iloc[lookback-1:].copy()
    feat["FutureRet"] = feat["Close"].shift(-horizon) / feat["Close"] - 1
    return feat

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
        if i > 0:
            open_today = float(df["Open"].iloc[i])
            slip_buy  = open_today * (1 + slippage_bps / 10000.0)
            slip_sell = open_today * (1 - slippage_bps / 10000.0)
            prob_prev = float(df["SignalProb"].iloc[i-1])
            date_exec = df.index[i]

            if (not in_pos) and prob_prev > entry_thr:
                # Sizing
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
                    "HoldDays": int(hold_days) if np.isfinite(hold_days) else np.nan
                })
                last_entry_date = None

        close_today = float(df["Close"].iloc[i])
        equity_gross.append(cash_gross + (shares * close_today if in_pos else 0.0))
        equity_net.append(cash_net + (shares * close_today if in_pos else 0.0))

    df_bt = df.copy()
    df_bt["Equity_Gross"] = equity_gross
    df_bt["Equity_Net"]   = equity_net
    return df_bt, trades

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

@st.cache_data(show_spinner=False, ttl=300)
def run_backtest_for_ticker(
    ticker: str,
    years: int,
    use_tail: bool,
    intraday_interval: str,
    fallback_last_session: bool,
    exec_mode: str,
    moc_cutoff_min: int,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    lookback: int,
    horizon: int,
    threshold: float,
    model_params: dict,
    entry_prob: float,
    exit_prob: float,
    commission: float,
    slippage_bps: int,
    init_cap: float,
    pos_frac: float,
    sizing_mode: str,
    atr_lookback: int,
    atr_k: float,
    risk_per_trade_pct: float,
):
    # load data
    df_full, meta = get_price_data_tail_intraday(
        ticker, years=years, use_tail=use_tail, interval=intraday_interval,
        fallback_last_session=fallback_last_session, exec_mode_key=exec_mode,
        moc_cutoff_min_val=moc_cutoff_min
    )
    df = df_full.copy()
    if start_date is not None:
        df = df.loc[str(start_date):]
    if end_date is not None:
        df = df.loc[:str(end_date)]

    # features & model
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
    feat_bt = feat.iloc[:-1].copy()

    df_bt, trades = backtest_next_open(
        feat_bt, entry_prob, exit_prob, commission, slippage_bps,
        init_cap, pos_frac, sizing_mode, risk_per_trade_pct, atr_k
    )
    metrics = compute_performance(df_bt, trades, init_cap)
    return metrics, trades, df_bt, feat, meta

# ─────────────────────────────────────────────────────────────
# Sidebar / Parameter
# ─────────────────────────────────────────────────────────────
st.markdown("## 📈 AI Signal-based Trading-Strategy")

# Universum Auswahl
st.sidebar.header("Universum")
universe_source = st.sidebar.selectbox(
    "Quelle",
    ["Manuell (Textfeld)", "CSV Upload (Spalte: Symbol/Ticker)", "S&P 500 (Wikipedia)", "NASDAQ-100 (Wikipedia)", "DAX 40 (Wikipedia)", "ATX (Wikipedia)"],
)

manual_input = st.sidebar.text_area("Manuelle Ticker (kommagetrennt)", value="AAPL,MSFT,NVDA")
csv_file = st.sidebar.file_uploader("CSV Upload", type=["csv"])

# Backtest-Settings
st.sidebar.header("Backtest-Parameter")
START_DATE = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
END_DATE   = st.sidebar.date_input("End Date",   value=pd.to_datetime(datetime.now(LOCAL_TZ).date()))

LOOKBACK = st.sidebar.number_input("Lookback (Tage)", min_value=10, max_value=252, value=60, step=5)
HORIZON  = st.sidebar.number_input("Horizon (Tage)", min_value=1, max_value=10, value=2)
THRESH   = st.sidebar.number_input("Threshold für Target", min_value=0.0, max_value=0.1, value=0.02, step=0.005, format="%.3f")

ENTRY_PROB = st.sidebar.slider("Entry Threshold (P(Signal))", min_value=0.0, max_value=1.0, value=0.63, step=0.01)
EXIT_PROB  = st.sidebar.slider("Exit Threshold (P(Signal))",  min_value=0.0, max_value=1.0, value=0.46, step=0.01)

COMMISSION   = st.sidebar.number_input("Commission (ad valorem, z.B. 0.001=10bp)", min_value=0.0, max_value=0.02, value=0.004, step=0.0001, format="%.4f")
SLIPPAGE_BPS = st.sidebar.number_input("Slippage (bp je Ausführung)", min_value=0, max_value=50, value=5, step=1)
POS_FRAC     = st.sidebar.slider("Max. Positionsgröße (% des Kapitals)", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
INIT_CAP     = st.sidebar.number_input("Initial Capital  (€)", min_value=1000.0, value=10_000.0, step=1000.0, format="%.2f")

sizing_mode = st.sidebar.selectbox("Positionsgröße", ["Fixed Fraction", "ATR Risk %"], index=0)
atr_lookback = st.sidebar.number_input("ATR Lookback (Tage)", min_value=5, max_value=100, value=14, step=1)
atr_k        = st.sidebar.number_input("ATR-Multiplikator (Stop-Distanz in ATR)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
risk_per_trade_pct = st.sidebar.number_input("Risiko pro Trade (% vom Kapital)", min_value=0.1, max_value=5.0, value=1.0, step=0.1) / 100.0

# Intraday Optionen
st.sidebar.header("Intraday-Tail")
use_live = st.sidebar.checkbox("Letzten Tag intraday aggregieren (falls verfügbar)", value=True)
intraday_interval = st.sidebar.selectbox("Intraday-Intervall (Tail & 5-Tage-Chart)", ["1m", "2m", "5m", "15m"], index=2)
fallback_last_session = st.sidebar.checkbox("Fallback: letzte Session verwenden (wenn heute leer)", value=False)
exec_mode = st.sidebar.selectbox("Execution Mode", ["Next Open (backtest+live)", "Market-On-Close (live only)"])
moc_cutoff_min = st.sidebar.number_input("MOC Cutoff (Minuten vor Close, nur live)", min_value=5, max_value=60, value=15, step=5)

# Modell
st.sidebar.header("Modellparameter")
n_estimators  = st.sidebar.number_input("n_estimators",  min_value=10, max_value=500, value=100, step=10)
learning_rate = st.sidebar.number_input("learning_rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f")
max_depth     = st.sidebar.number_input("max_depth",     min_value=1, max_value=10, value=3, step=1)
MODEL_PARAMS = dict(n_estimators=int(n_estimators), learning_rate=float(learning_rate), max_depth=int(max_depth), random_state=42)

# Universum erstellen
def build_universe() -> List[str]:
    if universe_source == "Manuell (Textfeld)":
        return [normalize_yahoo_symbol(t) for t in manual_input.split(",") if t.strip()]
    elif universe_source == "CSV Upload (Spalte: Symbol/Ticker)":
        if csv_file is None:
            st.warning("Bitte CSV hochladen.")
            return []
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            csv_file.seek(0)
            df = pd.read_csv(csv_file, sep=";")
        if "Symbol" in df.columns:
            col = "Symbol"
        elif "Ticker" in df.columns:
            col = "Ticker"
        else:
            col = df.columns[0]
        return [normalize_yahoo_symbol(s) for s in df[col].astype(str).tolist()]
    elif universe_source == "S&P 500 (Wikipedia)":
        return fetch_sp500_symbols()
    elif universe_source == "NASDAQ-100 (Wikipedia)":
        return fetch_nasdaq100_symbols()
    elif universe_source == "DAX 40 (Wikipedia)":
        return fetch_dax40_symbols()
    elif universe_source == "ATX (Wikipedia)":
        return fetch_atx_symbols()
    return []

universe = build_universe()
st.caption(f"Aktives Universum: **{len(universe)}** Ticker")

# ─────────────────────────────────────────────────────────────
# Universum backtesten & ranken
# ─────────────────────────────────────────────────────────────
st.markdown("### 🔎 Universum-Ranking (Sharpe)")
colk1, colk2 = st.columns([1, 2])
with colk1:
    run_rank = st.button("🚀 Universum backtesten & Top-K zeigen")
with colk2:
    TOP_K = st.slider("K (Top-Sharpe)", min_value=3, max_value=50, value=10, step=1)

ranking_df = None
topk = []
per_ticker_cache: Dict[str, dict] = {}

if run_rank and universe:
    progress = st.progress(0.0)
    rows = []
    for i, tk in enumerate(universe):
        progress.progress((i + 1) / max(1, len(universe)))
        try:
            metrics, trades, df_bt, feat, meta = run_backtest_for_ticker(
                ticker=tk, years=2, use_tail=use_live, intraday_interval=intraday_interval,
                fallback_last_session=fallback_last_session, exec_mode=exec_mode, moc_cutoff_min=moc_cutoff_min,
                start_date=pd.to_datetime(START_DATE), end_date=pd.to_datetime(END_DATE),
                lookback=LOOKBACK, horizon=HORIZON, threshold=THRESH, model_params=MODEL_PARAMS,
                entry_prob=ENTRY_PROB, exit_prob=EXIT_PROB, commission=COMMISSION, slippage_bps=SLIPPAGE_BPS,
                init_cap=INIT_CAP, pos_frac=POS_FRAC, sizing_mode=sizing_mode,
                atr_lookback=atr_lookback, atr_k=atr_k, risk_per_trade_pct=risk_per_trade_pct
            )
            metrics = dict(metrics)
            metrics["Ticker"] = tk
            metrics["Name"] = get_ticker_name(tk)
            rows.append(metrics)
            per_ticker_cache[tk] = dict(metrics=metrics, trades=trades, df_bt=df_bt, feat=feat, meta=meta)
        except Exception as e:
            st.warning(f"{tk}: {e}")

    if rows:
        ranking_df = pd.DataFrame(rows).set_index("Ticker").sort_values("Sharpe-Ratio", ascending=False)
        st.dataframe(ranking_df)
        topk = ranking_df.head(TOP_K).index.tolist()
        st.success(f"Top-{len(topk)} nach Sharpe bereit.")
    else:
        st.info("Noch keine Ergebnisse verfügbar. Stelle sicher, dass das Universum gültige Ticker enthält und genügend Daten vorhanden sind.")
else:
    st.info("Nutze oben '🚀 Universum backtesten & Top-K zeigen', um das Ranking zu berechnen.")

# ─────────────────────────────────────────────────────────────
# Detailanalyse für Top-K (oder leeres Ranking → nichts)
# ─────────────────────────────────────────────────────────────
if topk:
    st.markdown("## 🔝 Detailanalyse: Top-K nach Sharpe")

    results = []
    all_trades: Dict[str, List[dict]] = {}
    all_dfs:   Dict[str, pd.DataFrame] = {}
    all_feat:  Dict[str, pd.DataFrame] = {}

    for ticker in topk:
        with st.expander(f"🔍 Analyse für {ticker}", expanded=False):
            st.subheader(f"{ticker} – {get_ticker_name(ticker)}")
            try:
                if ticker in per_ticker_cache:
                    m = per_ticker_cache[ticker]
                    metrics, trades, df_bt, feat, meta = m["metrics"], m["trades"], m["df_bt"], m["feat"], m["meta"]
                else:
                    metrics, trades, df_bt, feat, meta = run_backtest_for_ticker(
                        ticker=ticker, years=2, use_tail=use_live, intraday_interval=intraday_interval,
                        fallback_last_session=fallback_last_session, exec_mode=exec_mode, moc_cutoff_min=moc_cutoff_min,
                        start_date=pd.to_datetime(START_DATE), end_date=pd.to_datetime(END_DATE),
                        lookback=LOOKBACK, horizon=HORIZON, threshold=THRESH, model_params=MODEL_PARAMS,
                        entry_prob=ENTRY_PROB, exit_prob=EXIT_PROB, commission=COMMISSION, slippage_bps=SLIPPAGE_BPS,
                        init_cap=INIT_CAP, pos_frac=POS_FRAC, sizing_mode=sizing_mode,
                        atr_lookback=atr_lookback, atr_k=atr_k, risk_per_trade_pct=risk_per_trade_pct
                    )

                results.append(dict(metrics))
                all_trades[ticker] = trades
                all_dfs[ticker] = df_bt
                all_feat[ticker] = feat

                last_timestamp_info(feat, meta)

                # Kennzahlen
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Strategie Netto (%)", f"{metrics['Strategy Net (%)']:.2f}")
                c2.metric("Buy & Hold (%)",      f"{metrics['Buy & Hold Net (%)']:.2f}")
                c3.metric("Sharpe",               f"{metrics['Sharpe-Ratio']:.2f}")
                c4.metric("Max Drawdown (%)",     f"{metrics['Max Drawdown (%)']:.2f}")
                c5.metric("Trades (Round-Trips)", f"{int(metrics['Number of Trades'])}")

                charts = st.columns(2)

                # Daily Preis + Signalfarbe
                df_plot = feat.copy()
                price_fig = go.Figure()
                price_fig.add_trace(go.Scatter(
                    x=df_plot.index, y=df_plot["Close"], mode="lines", name="Close",
                    line=dict(color="rgba(0,0,0,0.35)", width=1),
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
                    entries = trades_df[trades_df["Typ"] == "Entry"]
                    exits   = trades_df[trades_df["Typ"] == "Exit"]
                    price_fig.add_trace(go.Scatter(
                        x=entries["Date"], y=entries["Price"], mode="markers", name="Entry",
                        marker_symbol="triangle-up", marker=dict(size=12, color="green"),
                        hovertemplate="Entry<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>"
                    ))
                    price_fig.add_trace(go.Scatter(
                        x=exits["Date"], y=exits["Price"], mode="markers", name="Exit",
                        marker_symbol="triangle-down", marker=dict(size=12, color="red"),
                        hovertemplate="Exit<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>"
                    ))
                price_fig.update_layout(
                    title=f"{ticker}: Preis mit Signal-Wahrscheinlichkeit (Daily)",
                    xaxis_title="Datum", yaxis_title="Preis",
                    height=420, margin=dict(t=40, b=30, l=40, r=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                with charts[0]:
                    st.plotly_chart(price_fig, use_container_width=True)

                # Intraday letzte 5 Handelstage
                intra = get_intraday_last_n_sessions(ticker, sessions=5, days_buffer=10, interval=intraday_interval)
                with charts[1]:
                    if intra.empty:
                        st.info("Keine Intraday-Daten verfügbar.")
                    else:
                        intr_fig = go.Figure()
                        intr_fig.add_trace(
                            go.Candlestick(
                                x=intra.index,
                                open=intra["Open"], high=intra["High"],
                                low=intra["Low"],  close=intra["Close"],
                                name="OHLC (intraday)",
                                increasing_line_width=1, decreasing_line_width=1
                            )
                        )
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
                                    ts0 = day_slice.index.min()
                                    y_val = float(hit["Price"].iloc[-1])  # Exec-Preis
                                    xs.append(ts0); ys.append(y_val)
                                if xs:
                                    intr_fig.add_trace(
                                        go.Scatter(
                                            x=xs, y=ys, mode="markers", name=typ,
                                            marker_symbol=symbol, marker=dict(size=11, color=color),
                                            hovertemplate=f"{typ}<br>%{{x|%Y-%m-%d %H:%M}}<br>%{{y:.2f}}<extra></extra>"
                                        )
                                    )
                        intr_fig.update_layout(
                            title=f"{ticker}: Intraday – letzte 5 Handelstage ({intraday_interval})",
                            xaxis_title="Zeit", yaxis_title="Preis",
                            height=420, margin=dict(t=40, b=30, l=40, r=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        for _, day_slice in intra.groupby(intra.index.normalize()):
                            intr_fig.add_vline(x=day_slice.index.min(), line_width=1, line_dash="dot", opacity=0.3)
                        st.plotly_chart(intr_fig, use_container_width=True)

                # Equity-Kurve
                eq = go.Figure()
                eq.add_trace(go.Scatter(x=df_bt.index, y=df_bt["Equity_Net"], name="Strategy Net Equity (Next Open)",
                                        mode="lines", hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}€<extra></extra>"))
                bh_curve = INIT_CAP * df_bt["Close"] / df_bt["Close"].iloc[0]
                eq.add_trace(go.Scatter(x=df_bt.index, y=bh_curve, name="Buy & Hold", mode="lines",
                                        line=dict(dash="dash", color="black")))
                eq.update_layout(title=f"{ticker}: Net Equity-Kurve vs. Buy & Hold", xaxis_title="Datum", yaxis_title="Equity (€)",
                                 height=380, margin=dict(t=40, b=30, l=40, r=20),
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(eq, use_container_width=True)

                # Trades-Tabelle
                with st.expander(f"Trades (Next Open) für {ticker}", expanded=False):
                    if not trades_df.empty:
                        df_tr = trades_df.copy()
                        df_tr["Ticker"] = ticker
                        df_tr["Name"] = get_ticker_name(ticker)
                        df_tr["Date"] = pd.to_datetime(df_tr["Date"])
                        df_tr["DateStr"] = df_tr["Date"].dt.strftime("%Y-%m-%d")
                        df_tr["CumPnL"] = df_tr.where(df_tr["Typ"]=="Exit")["Net P&L"].cumsum().fillna(method="ffill").fillna(0)
                        df_tr = df_tr.rename(columns={"Net P&L":"PnL","Prob":"Signal Prob","HoldDays":"Hold (days)"})
                        # Holding Days ohne Komma → int
                        if "Hold (days)" in df_tr.columns:
                            df_tr["Hold (days)"] = pd.to_numeric(df_tr["Hold (days)"], errors="coerce").astype("Int64")
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

    # Summary über Top-K
    if results:
        summary_df = pd.DataFrame(results).set_index("Ticker")
        summary_df["Net P&L (%)"] = (summary_df["Net P&L (€)"] / INIT_CAP) * 100

        total_net_pnl  = float(summary_df["Net P&L (€)"].sum())
        total_fees     = float(summary_df["Fees (€)"].sum())
        total_gross_pnl = total_net_pnl + total_fees
        total_trades   = int(summary_df["Number of Trades"].sum())
        total_capital  = INIT_CAP * len(summary_df)
        total_net_return_pct   = total_net_pnl / total_capital * 100 if total_capital else 0.0
        total_gross_return_pct = total_gross_pnl / total_capital * 100 if total_capital else 0.0
        bh_total_pct = float(summary_df["Buy & Hold Net (%)"].dropna().mean()) if "Buy & Hold Net (%)" in summary_df.columns else float("nan")

        st.subheader("📊 Summary Top-K (Next Open Backtest)")
        cols = st.columns(5)
        cols[0].metric("Cumulative Net P&L (€)",  f"{total_net_pnl:,.2f}")
        cols[1].metric("Cumulative Trading Costs (€)", f"{total_fees:,.2f}")
        cols[2].metric("Cumulative Gross P&L (€)", f"{total_gross_pnl:,.2f}")
        cols[3].metric("Total Number of Trades",   f"{total_trades}")
        cols[4].metric("Ø Buy & Hold Net (%)", f"{bh_total_pct:.2f}")

        cols_pct = st.columns(2)
        cols_pct[0].metric("Strategy Net (%) – total",   f"{total_net_return_pct:.2f}")
        cols_pct[1].metric("Strategy Gross (%) – total", f"{total_gross_return_pct:.2f}")

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
            .set_caption("Strategy-Performance per Ticker (Top-K, Next Open Execution)")
        )
        show_styled_or_plain(summary_df, styled)
        st.download_button(
            "Summary Top-K als CSV",
            summary_df.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="summary_topk.csv", mime="text/csv"
        )
