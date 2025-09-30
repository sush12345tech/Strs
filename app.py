import streamlit as st
import pandas as pd
from datetime import datetime
from tvDatafeed import TvDatafeed, Interval
from io import BytesIO
import base64

# --- LOGIN SECTION ---
username = st.text_input("Enter TradingView Username/Email")
password = st.text_input("Enter TradingView Password", type="password")

if st.button("Login"):
    try:
        st.session_state.tv = TvDatafeed(username, password)
        st.success("✅ Logged in successfully!")
    except Exception as e:
        st.error(f"❌ Login failed: {e}")

st.title("📊 Stock Stochastic Trade Analyzer (Overlap-aware)")

symbols_input = st.text_area("Enter stock symbols (comma separated):", "AVALON, BLISSGVS, GALLANTT")
symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
exchange = st.selectbox("Select Exchange", ["NSE", "BSE"], index=0)

# --- Helper Functions ---
def classify_overlaps(trades_list, last_date):
    trades_sorted = sorted(trades_list, key=lambda x: x['Entry Date'])
    intervals = [(t['Entry Date'], t['Exit Date'] if t['Exit Date'] else last_date) for t in trades_sorted]
    for i, (s_i, e_i) in enumerate(intervals):
        fully, partial = False, False
        for j, (s_j, e_j) in enumerate(intervals):
            if i == j:
                continue
            intersects = (s_i <= e_j) and (e_i >= s_j)
            if intersects:
                if s_i >= s_j and e_i <= e_j:
                    fully = True
                    break
                else:
                    partial = True
        trades_sorted[i]['Overlap Type'] = "Fully Overlapped" if fully else ("Partially Overlapped" if partial else "No Overlap")
    return trades_sorted

def compute_counts(trades, total_trades):
    counts = {
        "<=5_days": 0, "5_to_10_days": 0, "10_to_20_days": 0, "20_to_30_days": 0, ">30_days": 0,
        "Never Hit": 0, "TargetHit_NoOverlap": 0,
        "No Overlap": 0, "Partially Overlapped": 0, "Fully Overlapped": 0
    }

    for t in trades:
        overlap = t.get("Overlap Type", "No Overlap")
        counts[overlap] += 1

        if t["Outcome"] == "Target Hit":
            hd = t.get("Holding Days", 0)
            if hd <= 5: counts["<=5_days"] += 1
            elif hd <= 10: counts["5_to_10_days"] += 1
            elif hd <= 20: counts["10_to_20_days"] += 1
            elif hd <= 30: counts["20_to_30_days"] += 1
            else: counts[">30_days"] += 1

            if overlap == "No Overlap":
                counts["TargetHit_NoOverlap"] += 1
        else:
            counts["Never Hit"] += 1

    for key in counts:
        counts[key] = counts[key] / total_trades * 100 if total_trades > 0 else 0

    return counts

def calculate_scores(counts):
    normal_score = (
        counts["<=5_days"] * 0.50 +
        counts["5_to_10_days"] * 0.25 +
        counts["10_to_20_days"] * 0.125 +
        counts["20_to_30_days"] * 0.075 +
        counts[">30_days"] * 0.05
    )
    bonus_score = (
        counts.get("TargetHit_NoOverlap", 0) * 0.20 +
        counts.get("Partially Overlapped", 0) * 0.15 +
        counts.get("Fully Overlapped", 0) * 0.10 -
        counts.get("Never Hit", 0) * 0.05
    )
    weighted_score = normal_score + bonus_score
    return round(normal_score, 2), round(bonus_score, 2), round(weighted_score, 2)

# --- MAIN ANALYSIS ---
if st.button("Run Analysis"):
    if "tv" not in st.session_state:
        st.error("⚠️ Please login first!")
    else:
        tv = st.session_state.tv
        summary = []
        trade_logs = {}
        today = datetime.today().strftime('%Y-%m-%d')
        progress = st.progress(0)

        for i, symbol in enumerate(symbols, start=1):
            try:
                df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=1500)
                if df is None or df.empty:
                    st.warning(f"No data for {symbol}")
                    progress.progress(i / len(symbols))
                    continue

                df['datetime'] = pd.to_datetime(df.index)
                df['low'] = pd.to_numeric(df['low'], errors='coerce')
                df['high'] = pd.to_numeric(df['high'], errors='coerce')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')

                # Indicators
                low_min = df['low'].rolling(4).min()
                high_max = df['high'].rolling(4).max()
                raw_k = 100 * (df['close'] - low_min) / (high_max - low_min)
                df['%K'] = raw_k.rolling(3).mean()
                df['%D'] = df['%K'].rolling(3).mean()

                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(2).mean()
                avg_loss = loss.rolling(2).mean()
                rs = avg_gain / avg_loss
                df['RSI_2'] = 100 - (100 / (1 + rs))
                df['200DMA'] = df['close'].rolling(200).mean()

                # Entry signals
                entries = df[(df['%K'] < 20) & (df['%D'] < 20) & (df['RSI_2'] < 15) & (df['close'] > df['200DMA'])].copy()
                trades_list = []
                last_date = df['datetime'].max()

                for _, row in entries.iterrows():
                    entry_date = row['datetime']
                    entry_price = row['close']
                    target_price = round(1.05 * entry_price, 2)
                    future_data = df[df['datetime'] > entry_date]
                    hit_target = future_data[future_data['high'] >= target_price]

                    if not hit_target.empty:
                        exit_date = hit_target.iloc[0]['datetime']
                        exit_price = hit_target.iloc[0]['high']
                        trades_list.append({
                            "Entry Date": entry_date,
                            "Entry Price": entry_price,
                            "Exit Date": exit_date,
                            "Exit Hit Price": exit_price,
                            "Outcome": "Target Hit",
                            "Holding Days": (exit_date - entry_date).days
                        })
                    else:
                        trades_list.append({
                            "Entry Date": entry_date,
                            "Entry Price": entry_price,
                            "Exit Date": None,
                            "Exit Hit Price": None,
                            "Outcome": "Open Trade",
                            "Holding Days": None
                        })

                trades_with_overlap = classify_overlaps(trades_list, last_date)
                total_trades = len(trades_with_overlap)
                counts = compute_counts(trades_with_overlap, total_trades)
                normal_score, bonus_score, weighted_score = calculate_scores(counts)

                summary.append({
                    "Stock": symbol,
                    "Total Trades": total_trades,
                    "<=5 days %": round(counts["<=5_days"], 2),
                    "5-10 days %": round(counts["5_to_10_days"], 2),
                    "10-20 days %": round(counts["10_to_20_days"], 2),
                    "20-30 days %": round(counts["20_to_30_days"], 2),
                    ">30 days %": round(counts[">30_days"], 2),
                    "Never Hit %": round(counts["Never Hit"], 2),
                    "No Overlap %": round(counts["No Overlap"], 2),
                    "Partially Overlapped %": round(counts["Partially Overlapped"], 2),
                    "Fully Overlapped %": round(counts["Fully Overlapped"], 2),
                    "TargetHit & NoOverlap %": round(counts["TargetHit_NoOverlap"], 2),
                    "Normal Score": normal_score,
                    "Bonus Points": bonus_score,
                    "Weighted Score": weighted_score
                })

                trade_logs[symbol] = trades_with_overlap
            except Exception as e:
                st.error(f"Error fetching {symbol}: {e}")

            progress.progress(i / len(symbols))

        if summary:
            df_summary = pd.DataFrame(summary).sort_values(by="Weighted Score", ascending=False)

            st.subheader("📌 Stock Summary with Download")

            # Build custom HTML table with download buttons
            table_html = "<table><thead><tr>"
            for col in df_summary.columns:
                table_html += f"<th>{col}</th>"
            table_html += "<th>Download</th></tr></thead><tbody>"

            for idx, row in df_summary.iterrows():
                table_html += "<tr>"
                for col in df_summary.columns:
                    table_html += f"<td>{row[col]}</td>"

                # Create Excel in memory for this row
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    pd.DataFrame([row]).to_excel(writer, sheet_name="Summary", index=False)
                    pd.DataFrame(trade_logs[row['Stock']]).to_excel(writer, sheet_name="Trades", index=False)
                buffer.seek(0)
                b64 = base64.b64encode(buffer.read()).decode()
                filename = f"{row['Stock']}_trades_{today}.xlsx"
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Download</a>'

                table_html += f"<td>{href}</td></tr>"

            table_html += "</tbody></table>"

            st.markdown(table_html, unsafe_allow_html=True)
