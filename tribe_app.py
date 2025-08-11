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
from typing import Tuple, List, Dict
from zoneinfo import ZoneInfo

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
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
END_DATE = st.sidebar.date_input("End Date", value=pd.to_datetime(datetime.now(LOCAL_TZ).date()))

LOOKBACK = st.sidebar.number_input("Lookback (Tage)", min_value=10, max_value=252, value=60, step=5)
HORIZON = st.sidebar.number_input("Horizon (Tage)", min_value=1, max_value=10, value=2)
THRESH = st.sidebar.number_input("Threshold für Target", min_value=0.0, max_value=0.1, value=0.02, step=0.005, format="%.3f")

ENTRY_PROB = st.sidebar.slider("Entry Threshold (P(Signal))", min_value=0.0, max_value=1.0, value=0.63, step=0.01)
EXIT_PROB  = st.sidebar.slider("Exit Threshold (P(Signal))",  min_value=0.0, max_value=1.0, value=0.46, step=0.01)

COMMISSION = st.sidebar.number_input("Commission (ad valorem, z.B. 0.001=10bp)", min_value=0.0, max_value=0.02, value=0.0005, step=0.0001, format="%.4f")
SLIPPAGE_BPS = st.sidebar.number_input("Slippage (bp je Ausführung)", min_value=0, max_value=50, value=5, step=1)
POS_FRAC = st.sidebar.slider("Positionsgröße (% des Kapitals je Trade)", min_value=0.1, max_value=1.0, value=1.0, step=0.1)

INIT_CAP = st.sidebar.number_input("Initial Capital  (€)", min_value=1000.0, value=10_000.0, step=1000.0, format="%.2f")
use_live = st.sidebar.checkbox("Heute Intraday-Preis verwenden (falls verfügbar)", value=True)

exec_mode = st.sidebar.selectbox(
    "Execution Mode",
    ["Next Open (backtest+live)", "Market-On-Close (live only)"]
)
moc_cutoff_min = st.sidebar.number_input("MOC Cutoff (Minuten vor Close, nur live)", min_value=5, max_value=60, value=15, step=5)

st.sidebar.markdown("**Modellparameter**")
n_estimators = st.sidebar.number_input("n_estimators", min_value=10, max_value=500, value=100, step=10)
learning_rate = st.sidebar.number_input("learning_rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f")
max_depth = st.sidebar.number_input("max_depth", min_value=1, max_value=10, value=3, step=1)

MODEL_PARAMS = dict(
    n_estimators=int(n_estimators),
    learning_rate=float(learning_rate),
    max_depth=int(max_depth),
    random_state=42
)

# ─────────────────────────────────────────────────────────────
# Helper für Tabellen-Fallback
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

def last_timestamp_info(df: pd.DataFrame):
    ts = df.index[-1]
    st.caption(f"Letzter Datenpunkt: {ts.strftime('%Y-%m-%d %H:%M %Z')}")

# ─────────────────────────────────────────────────────────────
# Daten: Daily + Intraday-Snapshot mergen
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=120)
def get_price_data(ticker: str, years: int = 2, use_live: bool = True) -> pd.DataFrame:
    """
    Holt 1D-Daten für 'years' Jahre und ergänzt – falls verfügbar – den heutigen
    Balken durch Intraday-Infos (Open/High/Low aggregiert, Close=letzter Print).
    """
    tk = yf.Ticker(ticker)

    # Daily via period ist robuster als start/end bzgl. Inclusivity
    df = tk.history(period=f"{years}y", interval="1d", auto_adjust=True, actions=False)
    if df.empty:
        raise ValueError(f"Keine Daten für {ticker}")

    # Zeitzone säubern
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(LOCAL_TZ)

    # Optional: heutigen Intraday-Balken einpflegen
    if use_live:
        try:
            intraday = tk.history(period="1d", interval="1m", auto_adjust=True, actions=False)
            if not intraday.empty:
                if intraday.index.tz is None:
                    intraday.index = intraday.index.tz_localize("UTC")
                intraday.index = intraday.index.tz_convert(LOCAL_TZ)

                # Falls MOC-Cutoff gewünscht: bis Close - cutoff begrenzen
                # (Close-Zeit kennt yfinance nicht zuverlässig, daher begrenzen wir bis "jetzt - cutoff")
                now_local = datetime.now(LOCAL_TZ)
                cutoff_time = now_local - timedelta(minutes=int(moc_cutoff_min))
                intraday_cut = intraday.loc[:cutoff_time] if exec_mode.startswith("Market-On-Close") else intraday

                if not intraday_cut.empty:
                    last_bar = intraday_cut.iloc[-1]
                    day_key = pd.Timestamp(last_bar.name.date()).replace(tzinfo=LOCAL_TZ)

                    daily_row = {
                        "Open":  float(intraday_cut["Open"].iloc[0]),
                        "High":  float(intraday_cut["High"].max()),
                        "Low":   float(intraday_cut["Low"].min()),
                        "Close": float(last_bar["Close"]),
                        "Volume": float(intraday_cut["Volume"].sum()),
                    }
                    df.loc[day_key] = daily_row
                    df = df.sort_index()
        except Exception:
            # Fallback nur über fast_info.last_price
            try:
                lp = tk.fast_info.last_price
                if np.isfinite(lp):
                    today_key = pd.Timestamp(datetime.now(LOCAL_TZ).date()).replace(tzinfo=LOCAL_TZ)
                    if today_key in df.index:
                        df.loc[today_key, "Close"] = float(lp)
                    else:
                        last_close = float(df["Close"].iloc[-1])
                        df.loc[today_key, ["Open","High","Low","Close","Volume"]] = [last_close, lp, lp, lp, 0.0]
                    df = df.sort_index()
            except Exception:
                pass

    df.dropna(subset=["High", "Low", "Close"], inplace=True)
    return df

# ─────────────────────────────────────────────────────────────
# Features & Training ohne Leakage
# ─────────────────────────────────────────────────────────────
def make_features(df: pd.DataFrame, lookback: int, horizon: int) -> pd.DataFrame:
    feat = df.copy()
    feat["Range"] = feat["High"].rolling(lookback).max() - feat["Low"].rolling(lookback).min()
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
    model_params: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict], dict]:
    """
    Trainiert bis vorletzte Zeile (ohne Leakage), erzeugt Signal-Prob für alle,
    backtestet auf Next-Open-Execution (bis vorletzte Zeile).
    """
    feat = make_features(df, lookback, horizon)

    # Historie fürs Training (bis vorletzte Zeile)
    hist = feat.iloc[:-1].dropna(subset=["FutureRet"]).copy()
    if len(hist) < 30:
        raise ValueError("Zu wenige Datenpunkte nach Preprocessing für das Modell.")

    hist["Target"] = (hist["FutureRet"] > threshold).astype(int)
    X_cols = ["Range","SlopeHigh","SlopeLow"]
    X_train, y_train = hist[X_cols].values, hist["Target"].values

    scaler = StandardScaler().fit(X_train)
    model  = GradientBoostingClassifier(**model_params).fit(scaler.transform(X_train), y_train)

    # Score für alle (inkl. der letzten "live" Zeile)
    feat["SignalProb"] = model.predict_proba(scaler.transform(feat[X_cols].values))[:,1]

    # Backtest nur bis vorletzte Zeile (letzte ist Live/Out-of-sample)
    feat_bt = feat.iloc[:-1].copy()

    df_bt, trades = backtest_next_open(
        feat_bt, ENTRY_PROB, EXIT_PROB, COMMISSION, SLIPPAGE_BPS, INIT_CAP, POS_FRAC
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
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Erwartet df mit Spalten: ['Open','Close','High','Low','SignalProb', ...]
    Ausführung: Signal an Tag t => Trade am Open von t+1.
    Equity-Bewertung: Tagesende (Close).
    Kein Pyramiding (0/1-Position).
    """
    df = df.copy()
    n = len(df)
    if n < 2:
        raise ValueError("Zu wenige Datenpunkte für Backtest.")

    cash_gross = init_cap
    cash_net = init_cap
    shares_gross = 0.0
    shares_net = 0.0
    in_pos = False
    cost_basis_gross = 0.0
    cost_basis_net = 0.0

    equity_gross, equity_net, trades = [], [], []
    cum_pl_net = 0.0

    for i in range(n):
        # 1) Orderausführung am heutigen Open (Signal von gestern)
        if i > 0:
            open_today = float(df["Open"].iloc[i])
            slip_buy  = open_today * (1 + slippage_bps / 10000.0)
            slip_sell = open_today * (1 - slippage_bps / 10000.0)
            prob_prev = float(df["SignalProb"].iloc[i-1])
            date_exec = df.index[i]

            if (not in_pos) and prob_prev > entry_thr:
                invest_gross = cash_gross * pos_frac
                invest_net   = cash_net   * pos_frac
                if invest_net > 0:
                    fee_entry = invest_net * commission
                    shares_gross = invest_gross / slip_buy
                    shares_net   = (invest_net - fee_entry) / slip_buy
                    cost_basis_gross = invest_gross
                    cost_basis_net   = invest_net - fee_entry
                    cash_gross -= invest_gross
                    cash_net   -= invest_net
                    in_pos = True
                    trades.append({
                        "Date": date_exec, "Typ": "Entry", "Price": round(slip_buy, 4),
                        "Shares": round(shares_net, 4), "Gross P&L": 0.0,
                        "Fees": round(fee_entry, 2), "Net P&L": 0.0, "kum P&L": round(cum_pl_net, 2)
                    })

            elif in_pos and prob_prev < exit_thr:
                gross_value = shares_gross * slip_sell
                net_value_before_fee = shares_net * slip_sell
                fee_exit = net_value_before_fee * commission

                pnl_gross = gross_value - cost_basis_gross
                pnl_net   = (net_value_before_fee - fee_exit) - cost_basis_net

                cash_gross += gross_value
                cash_net   += (net_value_before_fee - fee_exit)

                in_pos = False
                shares_gross = 0.0
                shares_net   = 0.0
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
        equity_gross.append(cash_gross + (shares_gross * close_today if in_pos else 0.0))
        equity_net.append(cash_net + (shares_net * close_today if in_pos else 0.0))

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

# ─────────────────────────────────────────────────────────────
# Haupt
# ─────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size: 36px;'>📈 AI Signal-based Trading-Strategy</h1>", unsafe_allow_html=True)

results = []
all_trades: Dict[str, List[dict]] = {}
all_dfs: Dict[str, pd.DataFrame] = {}
all_feat: Dict[str, pd.DataFrame] = {}

for ticker in TICKERS:
    with st.expander(f"🔍 Analyse für {ticker}", expanded=False):
        st.subheader(f"{ticker}")
        try:
            # Daten laden (2 Jahre, inkl. optionalem Intraday-Merge)
            df_full = get_price_data(ticker, years=2, use_live=use_live)
            # Auf UI-Zeitraum beschränken
            df = df_full.loc[str(START_DATE):str(END_DATE)].copy()
            last_timestamp_info(df)

            # Trainieren + Backtest (Next Open) ohne Leakage
            feat, df_bt, trades, metrics = train_and_signal_no_leak(df, LOOKBACK, HORIZON, THRESH, MODEL_PARAMS)
            metrics["Ticker"] = ticker
            results.append(metrics)
            all_trades[ticker] = trades
            all_dfs[ticker] = df_bt
            all_feat[ticker] = feat

            # Kennzahlen
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Strategie Netto (%)", f"{metrics['Strategy Net (%)']:.2f}")
            col2.metric("Buy & Hold (%)", f"{metrics['Buy & Hold Net (%)']:.2f}")
            col3.metric("Sharpe", f"{metrics['Sharpe-Ratio']:.2f}")
            col4.metric("Max Drawdown (%)", f"{metrics['Max Drawdown (%)']:.2f}")

            # Hinweis bei MOC-Modus
            if exec_mode.startswith("Market-On-Close"):
                st.info("MOC-Modus: Live-Signal unten. Backtest-/Kennzahlen oben basieren auf 'Next Open' (robust & ohne Leakage).")

            # Preis + Signal (farbige Segmente)
            df_plot = feat.copy()
            price_fig = go.Figure()
            price_fig.add_trace(
                go.Scatter(
                    x=df_plot.index,
                    y=df_plot["Close"],
                    mode="lines",
                    name="Close",
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
                    go.Scatter(
                        x=seg_x,
                        y=seg_y,
                        mode="lines",
                        showlegend=False,
                        line=dict(color=color_seg, width=2),
                        hoverinfo="skip"
                    )
                )

            trades_df = pd.DataFrame(trades)
            if not trades_df.empty:
                trades_df["Date"] = pd.to_datetime(trades_df["Date"])
                entries = trades_df[trades_df["Typ"] == "Entry"]
                exits = trades_df[trades_df["Typ"] == "Exit"]
                price_fig.add_trace(
                    go.Scatter(
                        x=entries["Date"], y=entries["Price"], mode="markers",
                        marker_symbol="triangle-up", marker=dict(size=12, color="green"),
                        name="Entry",
                        hovertemplate="Entry<br>Datum: %{x|%Y-%m-%d}<br>Preis: %{y:.2f}<extra></extra>"
                    )
                )
                price_fig.add_trace(
                    go.Scatter(
                        x=exits["Date"], y=exits["Price"], mode="markers",
                        marker_symbol="triangle-down", marker=dict(size=12, color="red"),
                        name="Exit",
                        hovertemplate="Exit<br>Datum: %{x|%Y-%m-%d}<br>Preis: %{y:.2f}<extra></extra>"
                    )
                )

            price_fig.update_layout(
                title=f"{ticker}: Preis mit Signal-Wahrscheinlichkeit",
                xaxis_title="Datum",
                yaxis_title="Preis",
                height=400,
                margin=dict(t=50, b=30, l=40, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(price_fig, use_container_width=True)

            # Equity-Kurve (Next Open Backtest)
            equity_fig = go.Figure()
            equity_fig.add_trace(
                go.Scatter(
                    x=df_bt.index,
                    y=df_bt["Equity_Net"],
                    name="Strategy Net Equity (Next Open)",
                    mode="lines",
                    hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}€<extra></extra>"
                )
            )
            bh_curve = INIT_CAP * df_bt["Close"] / df_bt["Close"].iloc[0]
            equity_fig.add_trace(
                go.Scatter(
                    x=df_bt.index,
                    y=bh_curve,
                    name="Buy & Hold",
                    mode="lines",
                    line=dict(dash="dash", color="black"),
                    hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}€<extra></extra>"
                )
            )
            equity_fig.update_layout(
                title=f"{ticker}: Net Equity-Kurve vs. Buy & Hold",
                xaxis_title="Datum",
                yaxis_title="Equity (€)",
                height=400,
                margin=dict(t=50, b=30, l=40, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(equity_fig, use_container_width=True)

            # Trades Tabelle (Next Open Backtest)
            with st.expander(f"Trades (Next Open) für {ticker}", expanded=False):
                if not trades_df.empty:
                    df_tr = trades_df.copy()
                    df_tr["Date"] = df_tr["Date"].dt.strftime("%Y-%m-%d")
                    df_tr["CumPnL"] = df_tr.where(df_tr["Typ"] == "Exit")["Net P&L"].cumsum().fillna(method="ffill").fillna(0)
                    df_tr = df_tr.rename(columns={"Net P&L": "PnL"})
                    display_cols = ["Date", "Typ", "Price", "Shares", "PnL", "CumPnL", "Fees"]
                    styled_trades = df_tr[display_cols].style.format({
                        "Price": "{:.2f}",
                        "Shares": "{:.4f}",
                        "PnL": "{:.2f}",
                        "CumPnL": "{:.2f}",
                        "Fees": "{:.2f}",
                    })
                    show_styled_or_plain(df_tr[display_cols], styled_trades)
                    st.download_button(
                        label="Trades als CSV herunterladen",
                        data=df_tr[display_cols].to_csv(index=False).encode("utf-8"),
                        file_name=f"trades_{ticker}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Keine Trades vorhanden.")

            # LIVE: MOC-Modus Vorschau (ohne historischen Backtest)
            if exec_mode.startswith("Market-On-Close"):
                live_prob = float(feat["SignalProb"].iloc[-1])
                st.subheader("🕒 MOC Live-Vorschau")
                c1, c2, c3 = st.columns(3)
                c1.metric("Heutige P(Signal)", f"{live_prob:.4f}")
                c2.metric("Entry-Threshold", f"{ENTRY_PROB:.2f}")
                c3.metric("Exit-Threshold", f"{EXIT_PROB:.2f}")

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
# Zusammenfassung (Next Open Backtest)
# ─────────────────────────────────────────────────────────────
if results:
    summary_df = pd.DataFrame(results).set_index("Ticker")
    summary_df["Net P&L (%)"] = (summary_df["Net P&L (€)"] / INIT_CAP) * 100

    total_net_pnl = summary_df["Net P&L (€)"].sum()
    total_fees = summary_df["Fees (€)"].sum()
    total_gross_pnl = total_net_pnl + total_fees
    total_trades = summary_df["Number of Trades"].sum()
    total_capital = INIT_CAP * len(summary_df)
    total_net_return_pct = total_net_pnl / total_capital * 100
    total_gross_return_pct = total_gross_pnl / total_capital * 100

    st.subheader("📊 Summary of all Tickers (Next Open Backtest)")
    cols = st.columns(4)
    cols[0].metric("Cumulative Net P&L (€)", f"{total_net_pnl:,.2f}")
    cols[1].metric("Cumulative Trading Costs (€)", f"{total_fees:,.2f}")
    cols[2].metric("Cumulative Gross P&L (€)", f"{total_gross_pnl:,.2f}")
    cols[3].metric("Total Number of Trades", f"{int(total_trades)}")
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
            "Strategy Net (%)": "{:.2f}",
            "Strategy Gross (%)": "{:.2f}",
            "Buy & Hold Net (%)": "{:.2f}",
            "Volatility (%)": "{:.2f}",
            "Sharpe-Ratio": "{:.2f}",
            "Max Drawdown (%)": "{:.2f}",
            "Calmar-Ratio": "{:.2f}",
            "Fees (€)": "{:.2f}",
            "Net P&L (%)": "{:.2f}",
            "Net P&L (€)": "{:.2f}"
        })
        .applymap(lambda v: "font-weight: bold;" if isinstance(v, (int, float)) else "", subset=pd.IndexSlice[:, ["Sharpe-Ratio"]])
        .applymap(color_phase_html, subset=["Phase"])
        .set_caption("Strategy-Performance per Ticker (Next Open Execution)")
    )
    show_styled_or_plain(summary_df, styled)
    st.download_button(
        "Summary als CSV herunterladen",
        summary_df.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="strategy_summary.csv",
        mime="text/csv"
    )

    # Offene Positionen (auf Basis Next Open Backtest)
    open_positions = []
    for ticker, trades in all_trades.items():
        if trades and trades[-1]["Typ"] == "Entry":
            last_entry = next(t for t in reversed(trades) if t["Typ"] == "Entry")
            prob = all_feat[ticker]["SignalProb"].iloc[-1]
            open_positions.append({
                "Ticker": ticker,
                "Entry Date": pd.to_datetime(last_entry["Date"]).strftime("%Y-%m-%d"),
                "Entry Price": round(last_entry["Price"], 2),
                "Current Prob.": round(float(prob), 4),
            })
    st.subheader("📋 Open Positions (Next Open Backtest)")
    if open_positions:
        open_df = pd.DataFrame(open_positions)
        styled_open = open_df.style.format({"Entry Price": "{:.2f}", "Current Prob.": "{:.4f}"})
        show_styled_or_plain(open_df, styled_open)
        st.download_button(
            "Offene Positionen als CSV",
            open_df.to_csv(index=False).encode("utf-8"),
            file_name="open_positions.csv",
            mime="text/csv"
        )
    else:
        st.success("Keine offenen Positionen.")
else:
    st.warning("Noch keine Ergebnisse verfügbar. Stelle sicher, dass mindestens ein Ticker korrekt eingegeben ist und genügend Daten vorhanden sind.")
