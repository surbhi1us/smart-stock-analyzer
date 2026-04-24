import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import plotly.express as px
import requests
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# =========================================================
# Page Config
# =========================================================
st.set_page_config(page_title="StockPort Stock Analyzer", page_icon="A", layout="wide")

# Load VADER (NLTK) sentiment tool to score news headlines as positive, negative, or neutral
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# =========================================================
# Custom CSS - Premium editorial dark theme
# =========================================================

def load_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# =========================================================
# Session State
# =========================================================
if "selected_stocks" not in st.session_state: #keeps user’s portfolio
    st.session_state.selected_stocks = []
if "chat_history" not in st.session_state: #keeps conversation alive
    st.session_state.chat_history = []
if "analysis_results" not in st.session_state: #stores computed results
    st.session_state.analysis_results = None

# =========================================================
# Masthead
# =========================================================
st.markdown("""
<div class="masthead">
    <p class="masthead-title">StockPort</p>
    <span class="masthead-sub">Stock Intelligence Platform</span>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Ticker Search
# =========================================================
def search_ticker(keyword):
    if not keyword:
        return []

    keyword = keyword.strip()
    results = []

    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={keyword}&quotesCount=10&newsCount=0"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5) #header to mimic a browser and avoid request blocking

        if response.status_code == 200:
            data = response.json()

            for quote in data.get("quotes", []):
                symbol = quote.get("symbol", "")
                name = quote.get("longname") or quote.get("shortname") or symbol
                quote_type = quote.get("quoteType", "")

                if symbol and quote_type in ["EQUITY", "ETF"]: #To ignore irrelevant results (options, currencies, etc.)
                    results.append((f"{name} ({symbol})", symbol))
    except:
        pass

    seen = set()
    cleaned = []
    for display, symbol in results:
        if symbol not in seen:
            seen.add(symbol)
            cleaned.append((display, symbol))

    return cleaned[:10]

def get_stock_news(ticker, max_items=8):
    try:
        t = yf.Ticker(ticker)
        items = t.get_news(count=max_items, tab="news")
        cleaned = []

        for item in items:
            content = item.get("content", {}) if isinstance(item, dict) else {} #Handles inconsistent API structure

            title = content.get("title", "") or item.get("title", "")
            summary = content.get("summary", "") or content.get("description", "")
            provider = content.get("provider", {}) if isinstance(content.get("provider", {}), dict) else {}
            publisher = provider.get("displayName", "")
            canonical = content.get("canonicalUrl", {}) if isinstance(content.get("canonicalUrl", {}), dict) else {}
            link = canonical.get("url", "") or item.get("link", "")

            text = f"{title}. {summary}".strip()

            if title:
                cleaned.append({
                    "ticker": ticker,
                    "title": title,
                    "summary": summary,
                    "publisher": publisher,
                    "link": link,
                    "text": text
                })

        return cleaned
    except:
        return []
    
def analyze_sentiment(text):
    if not text or not text.strip():
        return 0.0, "Neutral"

    compound = sia.polarity_scores(text)["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return compound, label
def build_sentiment_summary(tickers):
    stock_rows = []
    news_rows = []

    for ticker in tickers:
        news_items = get_stock_news(ticker, max_items=8)

        if not news_items:
            stock_rows.append({
                "Stock": ticker,
                "Sentiment Score": 0.0,
                "Label": "Neutral",
                "Articles": 0
            })
            continue

        scores = []

        for item in news_items:
            score, label = analyze_sentiment(item["text"])
            scores.append(score)

            news_rows.append({
                "Stock": ticker,
                "Headline": item["title"],
                "Source": item["publisher"],
                "Sentiment": label,
                "Sentiment Score": score,
                "Link": item["link"]
            })

        avg_score = float(np.mean(scores)) if scores else 0.0

        if avg_score >= 0.05:
            final_label = "Positive"
        elif avg_score <= -0.05:
            final_label = "Negative"
        else:
            final_label = "Neutral"

        stock_rows.append({
            "Stock": ticker,
            "Sentiment Score": avg_score,
            "Label": final_label,
            "Articles": len(scores)
        })

    stock_df = pd.DataFrame(stock_rows)
    news_df = pd.DataFrame(news_rows)

    if stock_df.empty:
        return None, None, "Neutral", 0, 0, 0

    pos_count = int((stock_df["Label"] == "Positive").sum())
    neg_count = int((stock_df["Label"] == "Negative").sum())
    neu_count = int((stock_df["Label"] == "Neutral").sum())

    overall_score = stock_df["Sentiment Score"].mean()

    if overall_score >= 0.05:
        market_label = "Positive"
    elif overall_score <= -0.05:
        market_label = "Negative"
    else:
        market_label = "Neutral"

    return stock_df, news_df, market_label, pos_count, neg_count, neu_count

# =========================================================
# AI Chatbot
# =========================================================
def get_ai_response(user_message, context_data=None):
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

    if not api_key:
        return "⚠️ AI assistant not enabled. Please set GROQ_API_KEY."

    system_prompt = """
You are StockPort AI, a financial education assistant.
Explain finance concepts simply so beginners can understand.

Important:
- Avoid jargon
- Explain terms like Sharpe ratio, volatility, Monte Carlo
- Be educational not advisory
- Never give direct investment advice
"""

    if context_data:
        system_prompt += f"\n\nUser portfolio data:\n{context_data}"

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=600,
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Groq error: {str(e)}"
    
# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.markdown('<p class="section-label">Stock Search</p>', unsafe_allow_html=True)
    keyword = st.text_input("Company name or ticker", placeholder="Apple, ORCL, RELIANCE.NS")

    if keyword and len(keyword) >= 2:
        with st.spinner("Searching..."):
            search_results = search_ticker(keyword)

        if search_results:
            display_to_ticker = {d: t for d, t in search_results}
            choice = st.selectbox("Select", list(display_to_ticker.keys()), key="stock_selector", label_visibility="collapsed")
            if st.button("Add to portfolio", type="primary"):
                sel = display_to_ticker[choice]
                if sel not in st.session_state.selected_stocks:
                    st.session_state.selected_stocks.append(sel)
                    st.success(f"Added {sel}")
                    st.rerun()
                else:
                    st.warning("Already in portfolio")
        else:
            st.warning("We couldn’t find that stock. Try a company name like Apple or a ticker like AAPL.")

    st.markdown('<hr class="divider" style="margin:1.5rem 0;">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Portfolio</p>', unsafe_allow_html=True)

    if st.session_state.selected_stocks:
        for i, s in enumerate(st.session_state.selected_stocks):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"<span class='stock-chip'>{s}</span>", unsafe_allow_html=True)
            if c2.button("x", key=f"del_{i}"):
                st.session_state.selected_stocks.pop(i)
                st.rerun()
    else:
        st.caption("No stocks added yet")

# =========================================================
# Default state — no stocks
# =========================================================
if not st.session_state.selected_stocks:
    st.markdown('<p class="section-label">Quick Start</p>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    quick = {"Apple": "AAPL", "Microsoft": "MSFT", "Tesla": "TSLA", "Alphabet": "GOOGL"}
    for col, (name, ticker) in zip([col1, col2, col3, col4], quick.items()):
        with col:
            if st.button(name):
                st.session_state.selected_stocks.append(ticker)
                st.rerun()
    st.stop()

# =========================================================
# Settings row
# =========================================================
col1, col2, col3 = st.columns(3)
with col1:
    money = st.number_input("Investment per stock ($)", 1000, 500000, 10000, step=1000)
with col2:
    start = st.date_input("Analysis start date", datetime.now() - timedelta(days=365))
with col3:
    goal = st.selectbox("Investment goal", ["Balanced (Risk-Adjusted)", "Maximum Return", "Minimum Risk"])

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# =========================================================
# Analyze Button
# =========================================================
if st.button("Run Analysis", type="primary", width="stretch"):
    results = [] # Store per-stock performance metrics
    curves = {} # Store portfolio value over time for each stock
    returns_df = pd.DataFrame() # Store daily returns for correlation analysis
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(st.session_state.selected_stocks):
        status_text.text(f"Fetching {ticker}... ({idx+1}/{len(st.session_state.selected_stocks)})")
        progress_bar.progress((idx + 1) / len(st.session_state.selected_stocks))
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start, end=datetime.now())  # Add end=date
            if len(hist) < 60:  # More lenient
                st.warning(f"{ticker}: Short data ({len(hist)} days)")
                continue
                
            if hist.empty or "Close" not in hist:
                continue

            price = hist["Close"].dropna()
            if price.empty:
                continue

            shares = money / price.iloc[0] # Simulate investing fixed amount -> calculate number of shares
            value = price * shares
            curves[ticker] = value
            
            daily_returns = value.pct_change().dropna()
            if len(daily_returns) == 0:
                continue
            daily_returns.index = pd.to_datetime(daily_returns.index).tz_localize(None)
            returns_df[ticker] = daily_returns
            
            total_return = (value.iloc[-1] / money - 1) * 100
            annual_vol = daily_returns.std() * np.sqrt(252) * 100
            rf_rate = 0.02 / 252 # Assume constant risk-free rate (2% annually)
            daily_mean = daily_returns.mean()
            std = daily_returns.std()
            if std > 1e-6:
                sharpe = ((daily_mean - rf_rate) / std) * np.sqrt(252)
            else:
                sharpe = 0
            
            results.append({
                'Stock': ticker,
                'Profit %': total_return,
                'Risk %': annual_vol,
                'Sharpe': sharpe,
                'Final $': value.iloc[-1],
                'Start $': money,
                'Gain $': value.iloc[-1] - money
            })
        except Exception as e:
            st.error(f"{ticker}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        st.error("We couldn’t analyze these stocks with the current inputs. Try a longer date range or choose more common tickers.")
        st.stop()
    
    df = pd.DataFrame(results)
    df = df.dropna()  # Drop bad rows
    
    # Safe sorting with fillna
    if goal == "Maximum Return":
        df_ranked = df.sort_values('Profit %', ascending=False, na_position='last').reset_index(drop=True)
        metric_name = "Return"
    elif goal == "Minimum Risk":
        df_ranked = df.sort_values('Risk %', ascending=True, na_position='last').reset_index(drop=True)
        metric_name = "Risk"
    else:  # Balanced
        df['Sharpe'] = df['Sharpe'].fillna(0)
        df_ranked = df.sort_values('Sharpe', ascending=False).reset_index(drop=True)
        metric_name = "Sharpe Ratio"
    
    # Reindex for ranks
    df_ranked.index = df_ranked.index + 1
    sentiment_df, news_df, market_label, pos_count, neg_count, neu_count = build_sentiment_summary(
        st.session_state.selected_stocks
    )
    # Store & rerun
    st.session_state.analysis_results = {
        "df_ranked": df_ranked,
        "best": df_ranked.iloc[0],
        "curves": curves,
        "returns_df": returns_df,
        "metric_name": metric_name,
        "goal": goal,
        "money": money,
        "sentiment_df": sentiment_df,
        "news_df": news_df,
        "market_label": market_label,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "neu_count": neu_count
    }
    st.rerun()

# =========================================================
# Results section
# =========================================================
if st.session_state.analysis_results:
    res = st.session_state.analysis_results
    df_ranked = res["df_ranked"]
    best = res["best"]
    curves = res["curves"]
    returns_df = res["returns_df"]
    metric_name = res["metric_name"]
    money = res["money"]
    sentiment_df = res.get("sentiment_df")
    news_df = res.get("news_df")
    market_label = res.get("market_label", "Neutral")
    pos_count = res.get("pos_count", 0)
    neg_count = res.get("neg_count", 0)
    neu_count = res.get("neu_count", 0)

    # --- Winner card ---
    profit_color = "#4ade80" if best['Profit %'] >= 0 else "#f87171"
    st.caption("This is the stock that best matched your selected goal over the chosen time period.")
    st.markdown(f"""
    <div class="winner-card animate">
        <p style="font-size:0.68rem;font-weight:600;color:#52525b;letter-spacing:0.15em;text-transform:uppercase;margin:0 0 0.75rem 0;">
            Ranked #1 by {metric_name}
        </p>
        <p class="winner-ticker">{best['Stock']}</p>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:2rem;margin-top:1.25rem;">
            <div>
                <p class="winner-label">Return</p>
                <p class="winner-value" style="color:{profit_color}">{best['Profit %']:+.2f}%</p>
            </div>
            <div>
                <p class="winner-label">Final Value</p>
                <p class="winner-value">${(best['Final $'] if pd.notna(best['Final $']) else 0):,.0f}</p>
            </div>
            <div>
                <p class="winner-label">Annualized Volatility</p>
                <p class="winner-value">{best['Risk %']:.2f}%</p>
            </div>
            <div>
                <p class="winner-label">Sharpe Ratio</p>
                <p class="winner-value">{best['Sharpe']:.3f}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("What do these metrics mean?"):
        st.write("**Return**: Total gain or loss over the selected period.")
        st.write("**Final Value**: How much your investment would be worth at the end.")
        st.write("**Annualized Volatility**: How much the stock tends to move up and down. Higher means more risk.")
        st.write("**Sharpe Ratio**: Return earned for the level of risk taken. Higher is generally better.")


    if best['Profit %'] > 0 and best['Risk %'] < df_ranked['Risk %'].mean():
        insight = f"{best['Stock']} stood out by combining positive returns with relatively lower risk."
    elif best['Profit %'] > 0:
        insight = f"{best['Stock']} gave the strongest return, but its risk should also be considered."
    else:
        insight = f"No stock performed strongly in this period, but {best['Stock']} still ranked best for your selected goal."

    st.markdown(
        f"""
        <div style="background:#111113;border:1px solid #27272a;border-radius:8px;padding:0.9rem 1rem;margin:1rem 0 1.25rem 0;">
            <span style="color:#d4af37;font-size:0.72rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;">Key Insight</span><br>
            <span style="color:#e4e4e7;font-size:0.95rem;">{insight}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    # --- Full Ranked Table ---
    st.markdown(
        '<p class="section-label" style="margin-bottom:0.75rem;">All stocks — ranked</p>',
        unsafe_allow_html=True
    )
    st.caption("This table compares all selected stocks based on return, risk, and overall performance.")
    table_df = df_ranked.copy()
    table_df = table_df.reset_index().rename(columns={"index": "Rank"})
    table_df["Rank"] = table_df["Rank"].apply(lambda x: f"#{x}")

    styled_df = table_df[["Rank", "Stock", "Profit %", "Risk %", "Sharpe", "Final $", "Gain $"]].rename(
        columns={"Stock": "Ticker"}
    )

    def color_profit(val):
        try:
            return "color: #4ade80;" if float(val) >= 0 else "color: #f87171;"
        except:
            return ""

    st.dataframe(
        styled_df.style.format({
            "Profit %": "{:+.2f}%",
            "Risk %": "{:.2f}%",
            "Sharpe": "{:.3f}",
            "Final $": "${:,.0f}",
            "Gain $": "${:+,.0f}",
        }).map(color_profit, subset=["Profit %", "Gain $"]),
        width="stretch",
        hide_index=True
    )
    # --- Tabs ---
    tabs = st.tabs(["Growth Curves", "Sentiment & News", "Correlations", "Monte Carlo Simulation (GBM)","My Portfolio", "AI Assistant"])

    # Growth
    with tabs[0]:
        st.markdown('<p class="section-label">Growth Overview</p>', unsafe_allow_html=True)
        st.caption(
            "This section shows how each stock performed during your selected time period. "
            "The first chart shows how your investment changed over time, and the second chart gives a quick final comparison."
        )

        best_ticker = best["Stock"]

        st.markdown('<p class="section-label">Investment Value Over Time</p>', unsafe_allow_html=True)
        st.caption("This chart shows how much your investment would be worth over time in each stock.")

        fig = go.Figure()
        for t, v in curves.items():
            is_best = (t == best_ticker)
            fig.add_trace(go.Scatter(
                x=v.index,
                y=v.values,
                name=t,
                line=dict(
                    width=2.7 if is_best else 1.6,
                    color="#d4af37" if is_best else None
                ),
                opacity=1 if is_best else 0.6,
                hovertemplate=f"<b>{t}</b><br>Date: %{{x|%d %b %Y}}<br>Value: $%{{y:,.0f}}<extra></extra>"
            ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=470,
            hovermode="x unified",
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Date",
            font=dict(family="DM Sans", color="#a1a1aa"),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#27272a", borderwidth=1),
            xaxis=dict(showgrid=False, zeroline=False, color="#52525b"),
            yaxis=dict(showgrid=True, gridcolor="#1c1c1f", zeroline=False, color="#52525b"),
            margin=dict(l=0, r=0, t=20, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "How to read this: if one line stays above the others, that stock would have given your investment a higher value during this period."
        )

        st.markdown('<p class="section-label" style="margin-top:1.25rem;">Final Growth Comparison</p>', unsafe_allow_html=True)
        st.caption(
            "This chart gives a quick summary of which stock grew the most overall from your selected start date."
        )

        final_growth_rows = []
        for t, v in curves.items():
            growth_pct = ((v.iloc[-1] / v.iloc[0]) - 1) * 100
            final_growth_rows.append({
                "Stock": t,
                "Growth %": growth_pct
            })

        final_growth_df = pd.DataFrame(final_growth_rows).sort_values("Growth %", ascending=True)

        final_growth_df["Color"] = final_growth_df.apply(
            lambda row: "#d4af37" if row["Stock"] == best_ticker else (
                "#4ade80" if row["Growth %"] > 0 else "#f87171" if row["Growth %"] < 0 else "#a1a1aa"
            ),
            axis=1
        )

        bar_fig = go.Figure()

        bar_fig.add_trace(go.Bar(
            x=final_growth_df["Growth %"],
            y=final_growth_df["Stock"],
            orientation="h",
            marker_color=final_growth_df["Color"],
            text=[f"{x:+.2f}%" for x in final_growth_df["Growth %"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Growth: %{x:.2f}%<extra></extra>"
        ))

        bar_fig.add_vline(
            x=0,
            line_dash="dot",
            line_color="#52525b",
            line_width=1
        )

        bar_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            xaxis_title="Overall Growth (%)",
            yaxis_title="",
            font=dict(family="DM Sans", color="#a1a1aa"),
            xaxis=dict(showgrid=True, gridcolor="#1c1c1f", zeroline=False, color="#52525b"),
            yaxis=dict(showgrid=False, color="#a1a1aa"),
            margin=dict(l=0, r=40, t=20, b=0)
        )

        st.plotly_chart(bar_fig, use_container_width=True)

        st.info(
            "How to read this: bars to the right of zero mean the stock gained value overall. "
            "Bars to the left mean it lost value overall."
        )

    # --- Sentiment cards ---
    with tabs[1]:
        if sentiment_df is not None and not sentiment_df.empty:
            st.markdown('<p class="section-label">Market Mood</p>', unsafe_allow_html=True)
            st.caption("This shows whether recent news around your selected stocks has been mostly positive, negative, or mixed.")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Market Mood", market_label)
            with c2:
                st.metric("Positive News Stocks", pos_count)
            with c3:
                st.metric("Negative News Stocks", neg_count)
            with c4:
                st.metric("Mixed News Stocks", neu_count)

            st.caption("Based on recent Yahoo Finance news headlines scored with VADER sentiment.")
            st.markdown('<p class="section-label">Sentiment per Stock</p>', unsafe_allow_html=True)

            plot_df = sentiment_df.sort_values("Sentiment Score", ascending=True).copy()
            colors = plot_df["Sentiment Score"].apply(
                lambda x: "#4ade80" if x >= 0.05 else "#f87171" if x <= -0.05 else "#a1a1aa"
            )

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=plot_df["Sentiment Score"],
                y=plot_df["Stock"],
                orientation="h",
                marker_color=list(colors),
                text=[f"{x:+.3f}" for x in plot_df["Sentiment Score"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>"
            ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=420,
                xaxis_title="Sentiment Score (-1 negative to +1 positive)",
                yaxis_title="",
                font=dict(family="DM Sans", color="#a1a1aa"),
                xaxis=dict(showgrid=True, gridcolor="#1c1c1f", zeroline=True, zerolinecolor="#52525b"),
                yaxis=dict(showgrid=False),
                margin=dict(l=0, r=30, t=20, b=0)
            )
            st.plotly_chart(fig, width="stretch")

            st.caption("Positive score means recent news sounds more favorable. Negative score means recent news sounds more unfavorable.")

            if news_df is not None and not news_df.empty:
                st.markdown('<p class="section-label" style="margin-top:1rem;">Latest News Headlines</p>', unsafe_allow_html=True)

                stock_filter = st.selectbox(
                    "Select stock",
                    ["All"] + sorted(news_df["Stock"].unique().tolist()),
                    key="news_stock_filter"
                )

                filtered_news = news_df if stock_filter == "All" else news_df[news_df["Stock"] == stock_filter]

                for _, row in filtered_news.head(12).iterrows():
                    color = "#4ade80" if row["Sentiment"] == "Positive" else "#f87171" if row["Sentiment"] == "Negative" else "#a1a1aa"
                    st.markdown(
                        f"""
                        <div style="background:#111113;border:1px solid #27272a;border-left:3px solid {color};
                                    padding:0.85rem 1rem;border-radius:8px;margin-bottom:0.6rem;">
                            <div style="font-size:0.72rem;color:#71717a;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.35rem;">
                                {row['Stock']} · {row['Source'] if pd.notna(row['Source']) and row['Source'] else 'News Source'} · {row['Sentiment']}
                            </div>
                            <div style="font-size:0.92rem;color:#e4e4e7;line-height:1.45;">
                                {row['Headline']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("No sentiment data available yet for the selected stocks.")

    # Correlations
    with tabs[2]:
        st.caption("This shows which stocks tend to move together. Lower similarity usually means better diversification.")
        if len(returns_df.columns) > 1:
            corr = returns_df.corr()

            if corr.empty or corr.isna().all().all():
                st.warning("Not enough overlapping return data to calculate correlations for these stocks.")
            else:
                corr = corr.fillna(0)

                fig = px.imshow(
                    corr,
                    color_continuous_scale=[
                        [0.0, "#1a1812"],
                        [0.2, "#2c2616"],
                        [0.4, "#4c3e18"],
                        [0.5, "#6b571a"],
                        [0.7, "#a8841a"],
                        [0.85, "#d9b62c"],
                        [1.0, "#f4d03f"]
                    ],
                    zmin=-1,
                    zmax=1,
                    aspect='auto',
                    text_auto=".2f",
                    labels=dict(color="Correlation")
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=460,
                    font=dict(family="DM Sans", color="#a1a1aa"),
                    margin=dict(l=0, r=0, t=20, b=0),
                    coloraxis_colorbar=dict(tickfont=dict(family="JetBrains Mono"))
                )

                st.plotly_chart(fig, width="stretch")
                
        else:
            st.info("Add at least 2 stocks to compare how they move relative to each other.")

    # Monte Carlo Simulation (GBM)
    with tabs[3]:
        st.markdown('<p class="section-label">Monte Carlo Simulation (GBM)</p>', unsafe_allow_html=True)
        st.caption(
            "This section helps you explore what could happen next for the top-ranked stock. "
            "Instead of giving one exact future price, it creates many possible future scenarios "
            "so you can see a likely range of outcomes."
        )

        st.markdown(
            """
            <div style="background:#111113;border:1px solid #27272a;border-radius:8px;padding:0.9rem 1rem;margin:0.8rem 0 1rem 0;">
                <span style="color:#d4af37;font-size:0.72rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;">Easy explanation</span><br>
                <span style="color:#e4e4e7;font-size:0.95rem;line-height:1.6;">
                    <b>Monte Carlo simulation</b> means the app creates many possible future paths instead of one fixed guess.<br>
                    <b>Geometric Brownian Motion (GBM)</b> is the math rule used to move the stock forward step by step.<br>
                    We use GBM because stock prices usually move in a random way, but they should stay positive.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        best_ticker = best["Stock"] #only simulates for the top-ranked stock to keep it focused and manageable.

        if best_ticker in curves:
            series = curves[best_ticker].dropna() #past portfolio value of the best stock

            if len(series) < 60:
                st.warning("Not enough historical data to run the simulation properly.")
            else:
                log_returns = np.log(series / series.shift(1)).dropna()

                if log_returns.empty:
                    st.warning("Not enough return data to run the simulation.")
                else:
                    mu = log_returns.mean() #average daily return (drift)
                    sigma = log_returns.std() #volatility of returns

                    horizon_days = st.slider(
                        "How far ahead to explore",
                        min_value=30,
                        max_value=252,
                        value=252,
                        step=21,
                        key="gbm_days",
                        help="Choose how many trading days ahead you want to test. Example: 252 trading days is about 1 year."
                    )

                    num_sims = st.slider(
                        "How many future scenarios to test",
                        min_value=100,
                        max_value=1000,
                        value=500,
                        step=100,
                        key="gbm_sims",
                        help="Choose how many possible future paths to generate. More scenarios give a broader picture, but they are still not predictions."
                    )

                    st.markdown(
                        """
                        <div style="background:#111113;border:1px solid #27272a;border-radius:8px;padding:0.9rem 1rem;margin:0.8rem 0 1rem 0;">
                            <span style="color:#d4af37;font-size:0.72rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;">What this means</span><br>
                            <span style="color:#e4e4e7;font-size:0.95rem;line-height:1.55;">
                                <b>How far ahead to explore</b> tells the app how far into the future to test.<br>
                                <b>How many future scenarios to test</b> tells the app how many possible future paths to create.<br>
                                This helps you see a <b>range</b> of possible outcomes instead of one fixed answer.
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    dt = 1 / 252
                    S0 = series.iloc[-1]

                    np.random.seed(42)
                    Z = np.random.normal(0, 1, (horizon_days, num_sims))

                    prices = np.zeros((horizon_days + 1, num_sims))
                    prices[0] = S0

                    for t in range(1, horizon_days + 1):
                        prices[t] = prices[t - 1] * np.exp(
                            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t - 1]
                        )

                    ending_values = prices[-1]
                    p10, p50, p90 = np.percentile(ending_values, [10, 50, 90])

                    drift_annual = mu * 252 * 100
                    vol_annual = sigma * np.sqrt(252) * 100

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Current Value", f"${S0:,.0f}")
                    with c2:
                        st.metric("Estimated Drift", f"{drift_annual:.2f}%")
                    with c3:
                        st.metric("Estimated Volatility", f"{vol_annual:.2f}%")
                    with c4:
                        st.metric("Simulated Paths", f"{num_sims}")

                    st.caption(
                        "Estimated drift and volatility are calculated from the stock's historical log returns."
                    )

                    left, right = st.columns([1, 3])

                    with left:
                        st.markdown('<p class="section-label">Possible Outcomes</p>', unsafe_allow_html=True)

                        for label, val, col in [
                            ("Lower Case (10th percentile)", p10, "#f87171"),
                            ("Middle Case (50th percentile)", p50, "#e4e4e7"),
                            ("Upper Case (90th percentile)", p90, "#4ade80"),
                        ]:
                            st.markdown(
                                f"""
                                <div class="metric-card" style="margin-bottom:0.75rem;">
                                    <p class="metric-label">{label}</p>
                                    <p class="metric-value" style="color:{col};font-size:1.05rem;">${val:,.0f}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        st.caption(
                            "These values are not guaranteed outcomes. "
                            "They are just examples of where the stock could end up under this model."
                        )

                    with right:
                        fig = go.Figure()

                        show_paths = min(120, num_sims)
                        x_vals = np.arange(horizon_days + 1)

                        for i in range(show_paths):
                            fig.add_trace(go.Scatter(
                                x=x_vals,
                                y=prices[:, i],
                                mode="lines",
                                line=dict(width=1, color="rgba(161,161,170,0.18)"),
                                showlegend=False,
                                hoverinfo="skip"
                            ))

                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=np.percentile(prices, 50, axis=1),
                            mode="lines",
                            name="Median Path",
                            line=dict(color="#d4af37", width=3)
                        ))

                        for val, color, label in [
                            (p10, "#f87171", "10%"),
                            (p50, "#d4af37", "50%"),
                            (p90, "#4ade80", "90%"),
                        ]:
                            fig.add_hline(
                                y=val,
                                line_dash="dot",
                                line_color=color,
                                line_width=1.5,
                                annotation_text=label,
                                annotation_font_color=color
                            )

                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=420,
                            font=dict(family="DM Sans", color="#a1a1aa"),
                            yaxis_title="Simulated Portfolio Value ($)",
                            xaxis_title="Trading Days Ahead",
                            xaxis=dict(showgrid=False, zeroline=False, color="#52525b"),
                            yaxis=dict(showgrid=True, gridcolor="#1c1c1f", zeroline=False, color="#52525b"),
                            margin=dict(l=0, r=0, t=20, b=0),
                            legend=dict(bgcolor="rgba(0,0,0,0)")
                        )

                        st.plotly_chart(fig, width="stretch")

                    st.markdown(
                        """
                        <div style="background:#111113;border:1px solid #27272a;border-radius:8px;padding:0.9rem 1rem;margin-top:1rem;">
                            <span style="color:#d4af37;font-size:0.72rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;">How to read this</span><br>
                            <span style="color:#e4e4e7;font-size:0.95rem;line-height:1.5;">
                                Each faint line is one possible future path.  
                                The gold line is the middle path.  
                                The red and green levels show lower and higher ending ranges.
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.warning("The top-ranked stock is not available for simulation.")

    # My Portfolio
    with tabs[4]:
        st.markdown('<p class="section-label">My Portfolio</p>', unsafe_allow_html=True)

        st.caption(
            "Build your own portfolio using the stocks selected in the sidebar. "
            "Choose how much money goes into each stock and see how your full portfolio would have performed."
        )

        st.markdown(
            """
            <div style="background:#111113;border:1px solid #27272a;border-radius:8px;padding:1rem 1.1rem;margin:0.75rem 0 1rem 0;">
                <span style="color:#d4af37;font-size:0.72rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;">How to use this tab</span><br>
                <span style="color:#e4e4e7;font-size:0.95rem;line-height:1.65;">
                    <b>Step 1:</b> Enter your starting investment amount.<br>
                    <b>Step 2:</b> Set the share of portfolio for each stock.<br>
                    <b>Step 3:</b> The last stock adjusts automatically so the total stays at <b>100%</b>.<br>
                    <b>Step 4:</b> Choose a time period and view how the portfolio performed.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        if len(st.session_state.selected_stocks) < 2:
            st.info(
                "Add at least 2 stocks from the sidebar to use My Portfolio. "
                "Then come back here to divide your investment across them."
            )
            st.stop()

        st.info("You choose the stock mix. The app calculates how that full portfolio would have performed.")

        portfolio_period_map = {
            "1 Week": "7d",
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y"
        }

        c1, c2 = st.columns([1, 1])

        with c1:
            total_investment = st.number_input(
                "Starting investment amount",
                min_value=1000.0,
                value=10000.0,
                step=500.0,
                key="portfolio_total_investment"
            )

        with c2:
            portfolio_period_label = st.selectbox(
                "Observation period",
                list(portfolio_period_map.keys()),
                index=1,
                key="portfolio_period_label"
            )

        portfolio_period = portfolio_period_map[portfolio_period_label]

        st.markdown("#### Portfolio allocation")
        st.caption("Set what percentage of your total portfolio goes into each stock. The last stock adjusts automatically so the total stays 100%.")

        stocks = st.session_state.selected_stocks
        weights = {}
        editable_stocks = stocks[:-1]
        auto_stock = stocks[-1]

        default_weight = round(100 / len(stocks), 2)

        if "portfolio_weight_defaults_for" not in st.session_state:
            st.session_state.portfolio_weight_defaults_for = []

        if st.session_state.portfolio_weight_defaults_for != stocks:
            for i, ticker in enumerate(stocks):
                key = f"portfolio_weight_input_{ticker}"
                if i < len(stocks) - 1:
                    st.session_state[key] = default_weight
            st.session_state.portfolio_weight_defaults_for = stocks.copy()

        alloc_cols = st.columns(min(3, len(stocks)))
        running_total = 0.0

        for i, ticker in enumerate(editable_stocks):
            with alloc_cols[i % len(alloc_cols)]:
                val = st.number_input(
                    f"{ticker} share of portfolio (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=1.0,
                    key=f"portfolio_weight_input_{ticker}"
                )
                weights[ticker] = float(val)
                running_total += float(val)

        auto_weight = round(100.0 - running_total, 2)
        display_auto_weight = max(auto_weight, 0.0)
        weights[auto_stock] = display_auto_weight

        with alloc_cols[(len(stocks) - 1) % len(alloc_cols)]:
            st.number_input(
                f"{auto_stock} share of portfolio (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(display_auto_weight),
                step=1.0,
                key=f"portfolio_weight_auto_{auto_stock}",
                disabled=True
            )

        if auto_weight < 0:
            st.warning(
                f"The entered weights exceed 100%. Please reduce the other stock weights so {auto_stock} can remain at 0% or higher."
            )
            st.stop()

        total_weight = sum(weights.values())
        st.success(f"Total portfolio share: {total_weight:.2f}%")

        portfolio_rows = []
        portfolio_growth = pd.DataFrame()

        for ticker in stocks:
            try:
                hist = yf.Ticker(ticker).history(period=portfolio_period)

                if hist.empty or "Close" not in hist.columns:
                    continue

                hist = hist.dropna(subset=["Close"]).copy()

                if hist.empty or len(hist) < 2:
                    continue

                start_price = float(hist["Close"].iloc[0])
                end_price = float(hist["Close"].iloc[-1])

                allocation_pct = weights[ticker]
                invested_amount = total_investment * (allocation_pct / 100.0)

                shares_bought = invested_amount / start_price if start_price > 0 else 0
                end_value = shares_bought * end_price
                profit_loss = end_value - invested_amount
                return_pct = (profit_loss / invested_amount) * 100 if invested_amount > 0 else 0

                normalized_value = (hist["Close"] / start_price) * invested_amount if start_price > 0 else hist["Close"] * 0
                portfolio_growth[ticker] = normalized_value

                portfolio_rows.append({
                    "Ticker": ticker,
                    "Portfolio Share (%)": allocation_pct,
                    "Amount Invested": invested_amount,
                    "Start Price": start_price,
                    "End Price": end_price,
                    "Shares Bought": shares_bought,
                    "Start Value": invested_amount,
                    "End Value": end_value,
                    "Profit/Loss": profit_loss,
                    "Return (%)": return_pct
                })

            except Exception:
                continue

        if not portfolio_rows:
            st.error("Unable to build portfolio data for the selected stocks in this period.")
            st.stop()

        portfolio_df = pd.DataFrame(portfolio_rows)

        portfolio_df["Amount Invested"] = portfolio_df["Amount Invested"].round(2)
        portfolio_df["Start Price"] = portfolio_df["Start Price"].round(2)
        portfolio_df["End Price"] = portfolio_df["End Price"].round(2)
        portfolio_df["Shares Bought"] = portfolio_df["Shares Bought"].round(4)
        portfolio_df["Start Value"] = portfolio_df["Start Value"].round(2)
        portfolio_df["End Value"] = portfolio_df["End Value"].round(2)
        portfolio_df["Profit/Loss"] = portfolio_df["Profit/Loss"].round(2)
        portfolio_df["Return (%)"] = portfolio_df["Return (%)"].round(2)

        total_start_value = portfolio_df["Start Value"].sum()
        total_end_value = portfolio_df["End Value"].sum()
        total_profit = total_end_value - total_start_value
        portfolio_return_pct = (total_profit / total_start_value) * 100 if total_start_value != 0 else 0

        st.markdown("#### Portfolio summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Starting amount", f"${total_start_value:,.2f}")
        m2.metric("Ending amount", f"${total_end_value:,.2f}")
        m3.metric("Total profit/loss", f"${total_profit:,.2f}")
        m4.metric("Portfolio return", f"{portfolio_return_pct:.2f}%")

        st.markdown("#### Stock-wise portfolio table")
        st.caption(
            "This table shows how your starting amount was divided across the selected stocks, "
            "what price each stock had at the start and end of the chosen period, and the resulting performance."
        )

        display_portfolio_df = portfolio_df.copy()
        display_portfolio_df["Amount Invested"] = display_portfolio_df["Amount Invested"].map(lambda x: f"${x:,.2f}")
        display_portfolio_df["Start Price"] = display_portfolio_df["Start Price"].map(lambda x: f"${x:,.2f}")
        display_portfolio_df["End Price"] = display_portfolio_df["End Price"].map(lambda x: f"${x:,.2f}")
        display_portfolio_df["Start Value"] = display_portfolio_df["Start Value"].map(lambda x: f"${x:,.2f}")
        display_portfolio_df["End Value"] = display_portfolio_df["End Value"].map(lambda x: f"${x:,.2f}")
        display_portfolio_df["Profit/Loss"] = display_portfolio_df["Profit/Loss"].map(lambda x: f"${x:,.2f}")
        display_portfolio_df["Shares Bought"] = display_portfolio_df["Shares Bought"].map(lambda x: f"{x:,.4f}")
        display_portfolio_df["Return (%)"] = display_portfolio_df["Return (%)"].map(lambda x: f"{x:.2f}%")

        st.dataframe(
            display_portfolio_df,
            use_container_width=True,
            hide_index=True
        )

        if not portfolio_growth.empty:
            portfolio_growth = portfolio_growth.sort_index()
            portfolio_growth["Portfolio Total"] = portfolio_growth.sum(axis=1)

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=portfolio_growth.index,
                    y=portfolio_growth["Portfolio Total"],
                    mode="lines",
                    name="Portfolio Total",
                    line=dict(width=4, color="#d4af37"),
                    hovertemplate="<b>Portfolio Total</b><br>Date: %{x|%d %b %Y}<br>Value: $%{y:,.2f}<extra></extra>"
                )
            )

            fig.update_layout(
                title=f"My Portfolio Performance — {portfolio_period_label}",
                template="plotly_dark",
                height=500,
                paper_bgcolor="#111113",
                plot_bgcolor="#111113",
                font=dict(color="#e4e4e7", family="DM Sans"),
                showlegend=False,
                margin=dict(l=20, r=20, t=60, b=20),
                yaxis_title="Portfolio Value ($)",
                xaxis_title=""
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Quick explanation")
        st.success(
            f"If you had started with ${total_investment:,.2f}, your portfolio would be worth "
            f"${total_end_value:,.2f} after {portfolio_period_label.lower()}, "
            f"for a total return of {portfolio_return_pct:.2f}%."
        )

    # AI Assistant
    with tabs[5]:
        st.markdown('<p class="section-label">Ask the AI Assistant</p>', unsafe_allow_html=True)
        st.caption("Ask for simple explanations of any metric, stock result, or chart shown above.")
        # Context summary for the AI
        if st.session_state.analysis_results:
            r = st.session_state.analysis_results
            ctx_lines = [f"User analyzed: {', '.join(r['df_ranked']['Stock'].tolist())}"]
            ctx_lines.append(f"Goal: {r['goal']}")
            ctx_lines.append(f"Best pick: {r['best']['Stock']} with {r['best']['Profit %']:+.2f}% return, Sharpe {r['best']['Sharpe']:.2f}, Risk {r['best']['Risk %']:.2f}%")
            for _, row in r['df_ranked'].iterrows():
                ctx_lines.append(f"#{row.name} {row['Stock']}: Return {row['Profit %']:+.2f}%, Risk {row['Risk %']:.2f}%, Sharpe {row['Sharpe']:.3f}, Final ${row['Final $']:,.0f}")
            context = "\n".join(ctx_lines)
        else:
            context = None

        # Chat display
        if st.session_state.chat_history:
            chat_html = '<div class="chat-container">'
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    chat_html += f"""
                    <div class="chat-msg-user">
                        <div>
                            <p class="chat-sender chat-sender-user" style="text-align:right;">You</p>
                            <div class="chat-bubble-user">{msg['content']}</div>
                        </div>
                    </div>"""
                else:
                    chat_html += f"""
                    <div class="chat-msg-ai">
                        <div>
                            <p class="chat-sender chat-sender-ai">StockPort AI</p>
                            <div class="chat-bubble-ai">{msg['content']}</div>
                        </div>
                    </div>"""
            chat_html += '</div>'
            st.markdown(chat_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="chat-container" style="display:flex;align-items:center;justify-content:center;min-height:120px;">
                <p style="color:#52525b;font-size:0.88rem;">Ask a question about your stocks, metrics, or investment concepts.</p>
            </div>
            """, unsafe_allow_html=True)

        # Suggested questions
        st.markdown('<p class="section-label" style="margin-top:1rem;">Suggested</p>', unsafe_allow_html=True)
        sq_col1, sq_col2, sq_col3 = st.columns(3)
        suggested = [
            "Explain the Sharpe ratio in simple terms",
            f"Why might {best['Stock']} be the best pick?",
            "What does annualized volatility mean?"
        ]
        for col, question in zip([sq_col1, sq_col2, sq_col3], suggested):
            with col:
                if st.button(question, key=f"sq_{hash(question)}"):
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    with st.spinner("Thinking..."):
                        reply = get_ai_response(question, context)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

        # Chat input
        user_input = st.text_input("Your question", placeholder="What does a negative Sharpe ratio mean?", key="chat_input", label_visibility="collapsed")
        c1, c2 = st.columns([5, 1])
        with c2:
            send = st.button("Send", type="primary", width="stretch")

        if send and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            with st.spinner("Thinking..."):
                reply = get_ai_response(user_input.strip(), context)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("Clear conversation"):
                st.session_state.chat_history = []
                st.rerun()

# =========================================================
# Footer
# =========================================================
st.markdown("""
<hr class="divider">
<p style="font-size:0.72rem;color:#3f3f46;text-align:center;letter-spacing:0.08em;">
    StockPort &mdash; Data via Yahoo Finance &mdash; For informational purposes only. Not financial advice.
</p>
""", unsafe_allow_html=True)
