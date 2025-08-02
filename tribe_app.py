import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*figure layout has changed to tight.*")

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from math import sqrt
from datetime import datetime
from typing import Tuple, List, Dict

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────
# Config / Sidebar
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Signal-basierte Strategie Backtest", layout="wide")

st.markdown(
    """
    <style>
      .main-header {
          font-size: clamp(2rem, 5vw, 4rem);
          font-weight: 700;
          margin: 0 0 0.5rem;
          line-height: 1.1;
      }
    </style>
    <div class="main-header">📈 AI Signal-based Trading-Strategy</div>
    """,
    unsafe_allow_html=True,
)



st.sidebar.header("Parameter")
tickers_input = st.sidebar.text_input("Tickers (Comma-separated)", value="BABA,QBTS,VOW3.DE,INTC")
TICKERS = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
START_DATE = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
END_DATE = st.sidebar.date_input("End Date", value=pd.to_datetime(datetime.now().date()))
LOOKBACK = st.sidebar.number_input("Lookback (Tage)", min_value=10, max_value=252, value=60, step=5)
HORIZON = st.sidebar.number_input("Horizon (Tage)", min_value=1, max_value=10, value=2)
THRESH = st.sidebar.number_input("Threshold für Target", min_value=0.0, max_value=0.1, value=0.02, step=0.005, format="%.3f")
ENTRY_PROB = st.sidebar.slider("Entry Threshold (P(Signal))", min_value=0.0, max_value=1.0, value=0.63, step=0.01)
EXIT_PROB = st.sidebar.slider("Exit Threshold (P(Signal))", min_value=0.0, max_value=1.0, value=0.46, step=0.01)
COMMISSION = st.sidebar.number_input("Commission (per Trade, Share)", min_value=0.0, max_value=0.02, value=0.005, step=0.001, format="%.4f")
INIT_CAP = st.sidebar.number_input("Initial Capital  (€)", min_value=1000.0, value=10_000.0, step=1000.0, format="%.2f")

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
def show_styled_or_plain(df: pd.DataFrame, styler: pd.io.formats.style.Styler):
    try:
        st.markdown(styler.to_html(), unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Styled table konnte nicht gerendert werden, zeige einfache Tabelle. ({e})")
        st.dataframe(df)

# ─────────────────────────────────────────────────────────────
# Funktionen
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["Close"] = df["Close"].interpolate()
    df.dropna(subset=["High", "Low", "Close"], inplace=True)
    return df

def slope(arr: np.ndarray) -> float:
    x = np.arange(len(arr))
    return np.polyfit(x, arr, 1)[0]

def backtest_brutto_netto(
    df: pd.DataFrame,
    entry_thr: float,
    exit_thr: float,
    commission: float,
    init_cap: float
) -> Tuple[pd.DataFrame, List[dict]]:
    cap_gross, cap_net = init_cap, init_cap
    pos_gross = pos_net = 0
    shares_gross = shares_net = 0.0
    cost_basis_gross = cost_basis_net = 0.0
    equity_gross, equity_net, trades = [], [], []
    cum_pl_net = 0.0

    for date, row in df.iterrows():
        prob, price = row["SignalProb"], row["Close"]
        if pos_net == 0 and prob > entry_thr:
            shares_gross = cap_gross / price
            cost_basis_gross = cap_gross
            cap_gross = 0.0
            pos_gross = 1
            fee_entry = cap_net * commission
            net_cap = cap_net - fee_entry
            shares_net = net_cap / price
            cost_basis_net = net_cap
            cap_net = 0.0
            pos_net = 1
            trades.append({
                "Date": date, "Typ": "Entry", "Price": price,
                "Shares": round(shares_net, 4), "Gross P&L": 0.0,
                "Fees": round(fee_entry, 2), "Net P&L": 0.0, "kum P&L": round(cum_pl_net, 2)
            })
        elif pos_net == 1 and prob < exit_thr:
            gross_exit_value = shares_gross * price
            pnl_gross = gross_exit_value - cost_basis_gross
            cap_gross = gross_exit_value
            pos_gross = 0
            gross_value = shares_net * price
            fee_exit = gross_value * commission
            net_proceeds = gross_value - fee_exit
            pnl_net = net_proceeds - cost_basis_net
            cap_net = net_proceeds
            pos_net = 0
            cum_pl_net += pnl_net
            trades.append({
                "Date": date, "Typ": "Exit", "Price": price,
                "Shares": round(shares_net, 4), "Gross P&L": round(pnl_gross, 2),
                "Fees": round(fee_exit, 2), "Net P&L": round(pnl_net, 2), "kum P&L": round(cum_pl_net, 2)
            })
        equity_gross.append(shares_gross * price if pos_gross else cap_gross)
        equity_net.append(shares_net * price if pos_net else cap_net)

    df_bt = df.copy()
    df_bt["Equity_Gross"] = equity_gross
    df_bt["Equity_Net"] = equity_net
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

@st.cache_data(show_spinner=False)
def train_and_signal(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    threshold: float,
    model_params: dict
) -> Tuple[pd.DataFrame, List[dict], dict]:
    df_local = df.copy()
    df_local["Range"] = df_local["High"].rolling(lookback).max() - df_local["Low"].rolling(lookback).min()
    df_local["SlopeHigh"] = df_local["High"].rolling(lookback).apply(slope, raw=True)
    df_local["SlopeLow"] = df_local["Low"].rolling(lookback).apply(slope, raw=True)
    df_local = df_local.iloc[lookback - 1 :].copy()
    df_local["FutureRet"] = df_local["Close"].shift(-horizon) / df_local["Close"] - 1
    df_local.dropna(inplace=True)
    df_local["Target"] = (df_local["FutureRet"] > threshold).astype(int)

    X, y = df_local[["Range", "SlopeHigh", "SlopeLow"]], df_local["Target"]
    if len(y) < 10:
        raise ValueError("Zu wenige Datenpunkte nach Preprocessing für das Modell.")
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, shuffle=False)
    scaler = StandardScaler().fit(X_train)
    model = GradientBoostingClassifier(**model_params)
    model.fit(scaler.transform(X_train), y_train)
    df_local["SignalProb"] = model.predict_proba(scaler.transform(X))[:, 1]

    df_bt, trades = backtest_brutto_netto(df_local, ENTRY_PROB, EXIT_PROB, COMMISSION, INIT_CAP)
    return df_bt, trades, compute_performance(df_bt, trades, INIT_CAP)

# ─────────────────────────────────────────────────────────────
# Haupt
# ─────────────────────────────────────────────────────────────
st.title("📈 AI Signal-based Trading-Strategy")

results = []
all_trades: Dict[str, List[dict]] = {}
all_dfs: Dict[str, pd.DataFrame] = {}

for ticker in TICKERS:
    with st.expander(f"🔍 Analyse für {ticker}", expanded=False):
        st.subheader(f"{ticker}")
        try:
            df = download_data(ticker, START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"))
            df_bt, trades, metrics = train_and_signal(df, LOOKBACK, HORIZON, THRESH, MODEL_PARAMS)
            metrics["Ticker"] = ticker
            results.append(metrics)
            all_trades[ticker] = trades
            all_dfs[ticker] = df_bt

            # Kennzahlen: Strategie Netto vs. Buy & Hold + Sharpe & Drawdown
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Strategie Netto (%)", f"{metrics['Strategy Net (%)']:.2f}")
            col2.metric("Buy & Hold (%)", f"{metrics['Buy & Hold Net (%)']:.2f}")
            col3.metric("Sharpe", f"{metrics['Sharpe-Ratio']:.2f}")
            col4.metric("Max Drawdown (%)", f"{metrics['Max Drawdown (%)']:.2f}")

            # Preis + Signal
            price_fig = go.Figure()

            # Close-Linie im Hintergrund, dünn und halbtransparent
            price_fig.add_trace(
                go.Scatter(
                    x=df_bt.index,
                    y=df_bt["Close"],
                    mode="lines",
                    name="Close",
                    line=dict(color="rgba(0,0,0,0.4)", width=1),
                    hovertemplate="Datum: %{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>"
                )
            )

            # Signal-farbige Segmente darüber
            signal_probs = df_bt["SignalProb"]
            norm = (signal_probs - signal_probs.min()) / (signal_probs.max() - signal_probs.min() + 1e-9)
            colorscale = px.colors.diverging.RdYlGn
            for i in range(len(df_bt) - 1):
                seg_x = df_bt.index[i : i + 2]
                seg_y = df_bt["Close"].iloc[i : i + 2]
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
                        x=entries["Date"],
                        y=entries["Price"],
                        mode="markers",
                        marker_symbol="triangle-up",
                        marker=dict(size=12, color="green"),
                        name="Entry",
                        hovertemplate="Entry<br>Datum: %{x|%Y-%m-%d}<br>Preis: %{y:.2f}<extra></extra>"
                    )
                )
                price_fig.add_trace(
                    go.Scatter(
                        x=exits["Date"],
                        y=exits["Price"],
                        mode="markers",
                        marker_symbol="triangle-down",
                        marker=dict(size=12, color="red"),
                        name="Exit",
                        hovertemplate="Exit<br>Datum: %{x|%Y-%m-%d}<br>Preis: %{y:.2f}<extra></extra>"
                    )
                )

            price_fig.update_layout(
                title=f"{ticker}: Preis mit Signal-Wahrscheinlichkeit",
                xaxis_title="Datum",
                yaxis_title="Preis (€)",
                height=400,
                margin=dict(t=50, b=30, l=40, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(price_fig, use_container_width=True)

            # Equity-Kurve (nur Net Equity + Buy & Hold)
            equity_fig = go.Figure()
            equity_fig.add_trace(
                go.Scatter(
                    x=df_bt.index,
                    y=df_bt["Equity_Net"],
                    name="Strategy Net Equity",
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

            # Trades Tabelle
            with st.expander(f"Trades für {ticker}", expanded=False):
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
        except Exception as e:
            st.error(f"Fehler bei {ticker}: {e}")


# ─────────────────────────────────────────────────────────────
# Zusammenfassung
# ─────────────────────────────────────────────────────────────
if results:
    summary_df = pd.DataFrame(results).set_index("Ticker")
    bh_true_returns = {}
    for ticker in TICKERS:
        df_full = download_data(ticker, START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"))
        if len(df_full) > 1:
            bh_ret_full = (df_full["Close"].iloc[-1] / df_full["Close"].iloc[0] - 1) * 100
            bh_true_returns[ticker] = bh_ret_full
        else:
            bh_true_returns[ticker] = np.nan
    summary_df["Net P&L (%)"] = (summary_df["Net P&L (€)"] / INIT_CAP) * 100

    total_net_pnl = summary_df["Net P&L (€)"].sum()
    total_fees = summary_df["Fees (€)"].sum()
    total_gross_pnl = total_net_pnl + total_fees
    total_trades = summary_df["Number of Trades"].sum()
    total_capital = INIT_CAP * len(summary_df)
    total_net_return_pct = total_net_pnl / total_capital * 100
    total_gross_return_pct = total_gross_pnl / total_capital * 100

    st.subheader("📊 Summary of all Tickers")
    cols = st.columns(4)
    cols[0].metric("Cumulative Net P&LL (€)", f"{total_net_pnl:,.2f}")
    cols[1].metric("Cumulative Trading Costs (€)", f"{total_fees:,.2f}")
    cols[2].metric("Cumulative Gross P&L (€)", f"{total_gross_pnl:,.2f}")
    cols[3].metric("Total Number of Trades", f"{int(total_trades)}")
    st.markdown(
        f"**Total Net Return (%):** {total_net_return_pct:.2f}  \n"
        f"**Total Gross Return (%):** {total_gross_return_pct:.2f}"
    )

    def color_phase_html(val):
        colors = {
            "Open": "#d0ebff",
            "Flat": "#f0f0f0"
        }
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
        .applymap(
            lambda v: "font-weight: bold;" if isinstance(v, (int, float)) else "",
            subset=pd.IndexSlice[:, ["Sharpe-Ratio"]],
        )
        .applymap(color_phase_html, subset=["Phase"])
        .set_caption("Strategy-Performance per Ticker")
    )
    show_styled_or_plain(summary_df, styled)
    st.download_button(
        "Summary als CSV herunterladen",
        summary_df.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="strategy_summary.csv",
        mime="text/csv"
    )

    # Offene Positionen
    open_positions = []
    for ticker, trades in all_trades.items():
        if trades and trades[-1]["Typ"] == "Entry":
            last_entry = next(t for t in reversed(trades) if t["Typ"] == "Entry")
            prob = all_dfs[ticker]["SignalProb"].iloc[-1]
            open_positions.append({
                "Ticker": ticker,
                "Entry Date": last_entry["Date"].strftime("%Y-%m-%d"),
                "Entry Price": round(last_entry["Price"], 2),
                "Current Prob.": round(prob, 4),
            })
    st.subheader("📋 Open Positions")
    if open_positions:
        open_df = pd.DataFrame(open_positions)
        styled_open = open_df.style.format({
            "Entry Price": "{:.2f}",
            "Current Prob.": "{:.4f}"
        })
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








