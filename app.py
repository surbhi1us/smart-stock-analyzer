import streamlit as st # Web app UI framework
import yfinance as yf # Fetches real stock market data from Yahoo Finance
import pandas as pd # Handles data tables and calculations
import numpy as np # Math & statistics for returns, volatility, simulations
import plotly.graph_objects as go # For creating detailed interactive charts
from datetime import datetime, timedelta # Handle date inputs and calculations
import plotly.express as px # Simple plotting (heatmaps, correlations)
import requests # Make HTTP requests for ticker search API

# =========================================================
# Page Config: title, icon, layout of the Streamlit web app
# =========================================================
st.set_page_config(page_title="Smart Stock Picker", page_icon="📈", layout="wide")

# =========================================================
# Custom CSS styling for app visuals
# =========================================================
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}  /* Hide menu, header, footer */
.stApp {background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); color: white;} /* Background gradient */
.hero {padding: 3rem 2.5rem; border-radius: 24px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
       box-shadow: 0 18px 36px rgba(0,0,0,0.32); margin: 2rem auto; max-width: 1100px;} /* Hero section styling */
.stock-chip {padding: 0.5rem 1rem; border-radius: 25px; background: rgba(255,255,255,0.9); color: #0f172a; font-weight: 600;} /* Stock labels */
</style>
""", unsafe_allow_html=True)

# =========================================================
# Session state to store selected stocks persistently
# =========================================================
if "selected_stocks" not in st.session_state:
    st.session_state.selected_stocks = []  # Initialize empty list

# =========================================================
# Hero section: App title and tagline
# =========================================================
st.markdown("""
<div class="hero">
    <h1>Smart Stock Analyzer</h1>
    <p>🔍 Live search → 🎯 Goal-based pick → 📊 Full analysis</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Function: Search for stock tickers
# =========================================================
def search_ticker(keyword):
    """
    Search for stock tickers using multiple methods
    Returns: list of tuples (display_name, ticker)
    """
    if not keyword:
        return []
    
    results = []
    keyword = keyword.strip()  # Remove extra spaces
    
    # ------------------------------
    # Method 1: Direct ticker check
    # ------------------------------
    try:
        ticker_test = yf.Ticker(keyword.upper())
        info = ticker_test.info #Fetches company information
        if info and 'symbol' in info and info.get('regularMarketPrice'):  # Checks 3 things:Data exists,Stock symbol exists,Stock has a current price .. If all true → valid stock
            name = info.get('longName') or info.get('shortName') or keyword.upper()
            results.append((f"{name} ({keyword.upper()})", keyword.upper()))
    except:
        pass  # Ignore errors if ticker invalid
    
    # ------------------------------------------
    # Method 2: Yahoo Finance search API
    # ------------------------------------------
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={keyword}&quotesCount=10&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)  # Get API response
        
        if response.status_code == 200: #200 = success
            data = response.json() #Converts response to Python dictionary
            quotes = data.get('quotes', []) #Extracts stock list
            
            for quote in quotes:
                symbol = quote.get('symbol', '') #ticker symbol
                name = quote.get('longname') or quote.get('shortname') or symbol
                quote_type = quote.get('quoteType', '')
                
                # Only consider stocks & ETFs
                if quote_type in ['EQUITY', 'ETF'] and symbol:
                    display = f"{name} ({symbol})"
                    # Avoid duplicates
                    if not any(symbol == r[1] for r in results):
                        results.append((display, symbol))
    except:
        pass  # Ignore API errors
    
    # ------------------------------------------
    # Method 3: If search fails, try common ticker formats and pick the first valid stock.
    # ------------------------------------------
    if not results:
        variations = [
            keyword.upper(),
            keyword.lower(),
            keyword.upper() + ".US",
        ]
        
        for variant in variations:
            try:
                ticker = yf.Ticker(variant)
                info = ticker.info
                if info and 'symbol' in info and info.get('regularMarketPrice'):
                    name = info.get('longName') or info.get('shortName') or variant
                    results.append((f"{name} ({variant})", variant)) #Add first working result
                    break
            except:
                continue
    
    return results[:10]  # Return top 10 results only

# =========================================================
# Sidebar: Stock search and selection
# =========================================================
with st.sidebar:
    st.markdown("## 🔍 Live Stock Search")
    
    # Text input for company name or ticker
    keyword = st.text_input("Company name or ticker", placeholder="e.g., Apple, ORCL, Tesla")
    
    # If keyword length is at least 2 characters
    if keyword and len(keyword) >= 2: #Prevents searching when input is too small or empty.
        with st.spinner("Searching..."):
            search_results = search_ticker(keyword)  # Call search function
        
        if search_results:
            st.markdown("### 📋 Results:")
            
            # Map display name to ticker symbol
            display_to_ticker = {display: ticker for display, ticker in search_results} #Creates a dictionary,
            
            # User selects stock from dropdown
            choice = st.selectbox("Pick one:", list(display_to_ticker.keys()), key="stock_selector")
            
            # Button to add selected stock
            if st.button("➕ Add Stock", type="primary"):
                selected_ticker = display_to_ticker[choice] #Converts selected company name into its ticker symbol.
                if selected_ticker not in st.session_state.selected_stocks:
                    st.session_state.selected_stocks.append(selected_ticker)
                    st.success(f"Added {selected_ticker}!") #green success message
                    st.rerun()  # Refresh app to show added stock
                else:
                    st.warning("Already added!")
        else:
            st.warning("No results found. Try:")
            st.info("• Full company names: Apple, Microsoft, Tesla\n• Ticker symbols: AAPL, MSFT, TSLA\n• Popular stocks: Google, Amazon, Netflix")

    st.markdown("---")
    st.markdown("### 📌 Your Stocks")
    
    # Show added stocks in sidebar
    if st.session_state.selected_stocks:
        for i, s in enumerate(st.session_state.selected_stocks):
            c1, c2 = st.columns([3,1])
            c1.markdown(f"<span class='stock-chip'>{s}</span>", unsafe_allow_html=True)  # Stock label
            if c2.button("🗑️", key=f"del_{i}"):  # Delete stock
                st.session_state.selected_stocks.pop(i)
                st.rerun()
    else:
        st.info("No stocks added yet")

# =========================================================
# If no stocks added, show default suggestions
# =========================================================
if not st.session_state.selected_stocks:
    st.info("👈 Search and add stocks to begin analysis")
    st.markdown("### 💡 Try these:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("+ Apple"):
            st.session_state.selected_stocks.append("AAPL")
            st.rerun()
    with col2:
        if st.button("+ Microsoft"):
            st.session_state.selected_stocks.append("MSFT")
            st.rerun()
    with col3:
        if st.button("+ Tesla"):
            st.session_state.selected_stocks.append("TSLA")
            st.rerun()
    st.stop()  # Stop app until stocks are added

# =========================================================
# Investment settings: money, start date, goal
# =========================================================
col1, col2, col3 = st.columns(3)
with col1: 
    money = st.number_input("💰 Investment per stock ($)", 1000, 100000, 10000, step=1000) # Investment amount(min,max,default,step)
with col2: 
    start = st.date_input("📅 Analysis start date", datetime.now()-timedelta(days=365)) # Default 1 year ago
with col3: 
    goal = st.selectbox("🎯 Investment goal", ["Balanced (Risk-Adjusted)", "Max Return", "Min Risk"]) # Goal for selecting best stock

# =========================================================
# Analysis Button: Run all calculations
# =========================================================
if st.button("🚀 ANALYZE & PICK BEST", type="primary", use_container_width=True):
    results = []  # Store stock metrics
    curves = {}   # Store portfolio growth over time
    returns_df = pd.DataFrame()  # Daily returns for correlation
    
    progress_bar = st.progress(0)  # Show progress
    status_text = st.empty()       # Display status messages
    
    # ------------------------------
    # Loop through selected stocks
    # ------------------------------
    for idx, ticker in enumerate(st.session_state.selected_stocks): #Loop through each selected stock while also knowing which number it is
        status_text.text(f"Analyzing {ticker}... ({idx+1}/{len(st.session_state.selected_stocks)})")
        progress_bar.progress((idx + 1) / len(st.session_state.selected_stocks))
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start)  # Fetch historical prices
            
            if len(hist) < 30:  # Skip if insufficient data
                st.warning(f"⚠️ {ticker}: Insufficient data (less than 30 days)")
                continue
            
            price = hist["Close"]  # Closing prices
            shares = money / price.iloc[0]  # Compute how many shares you can buy on the first day
            value = price * shares  # Portfolio value over each day
            curves[ticker] = value #Stores each stock’s portfolio value separately
            
            daily_returns = value.pct_change().dropna()  # Daily % change
            returns_df[ticker] = daily_returns
            
            # Calculate metrics
            total_return = (value.iloc[-1]/money - 1) * 100  # % profit
            annual_vol = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
            sharpe = (daily_returns.mean() - 0.02/252) / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0  # Sharpe ratio
            
            # Append to results
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
            st.error(f"❌ {ticker}: Error - {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    df = pd.DataFrame(results)  # Convert results to dataframe
    
    if df.empty:
        st.error("No valid data found for any stocks.")
        st.stop()
    
    # =========================================================
    # Goal-based stock selection
    # =========================================================
    if goal == "Max Return":
        best = df.loc[df['Profit %'].idxmax()]
        metric_name = "Highest Return"
    elif goal == "Min Risk":
        best = df.loc[df['Risk %'].idxmin()]
        metric_name = "Lowest Risk"
    else:
        best = df.loc[df['Sharpe'].idxmax()]
        metric_name = "Best Sharpe Ratio"
    
    # =========================================================
    # Show Best Stock Metrics
    # =========================================================
    st.markdown("### 🏆 YOUR #1 PICK")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🥇 BEST STOCK", best['Stock'], f"{best['Profit %']:+.1f}%")
    with col2:
        st.metric("💰 Final Value", f"${best['Final $']:,.0f}", f"${best['Gain $']:+,.0f}")
    with col3:
        st.metric("⚠️ Risk (Annual)", f"{best['Risk %']:.1f}%")
    with col4:
        st.metric("⭐ Sharpe Score", f"{best['Sharpe']:.2f}")
    
    st.success(f"✅ **{best['Stock']}** selected based on **{metric_name}** | ${money:,} → ${best['Final $']:,.0f}")
    
    # =========================================================
    # Tabs for visualization
    # =========================================================
    tabs = st.tabs(["📈 Growth Curves", "📊 All Stocks", "🔗 Correlations", "🎲 Monte Carlo"])
    
    # ------------------------------
    # 1. Growth Curves
    # ------------------------------
    with tabs[0]:
        st.markdown("### Portfolio Growth Over Time")
        fig = go.Figure()
        for t, v in curves.items(): #t → stock ticker name, v → portfolio value over time
            is_best = (t == best['Stock'])
            fig.add_trace(go.Scatter(
                x=v.index, 
                y=v.values, 
                name=t,
                line=dict(width=3 if is_best else 2),
                opacity=1 if is_best else 0.7
            ))
        fig.update_layout(
            template="plotly_dark", 
            height=500,
            hovermode='x unified',
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Date"
        )
        st.plotly_chart(fig, width='stretch')
    
    # ------------------------------
    # 2. All Stocks Comparison Table
    # ------------------------------
    with tabs[1]:
        st.markdown("### Complete Stock Comparison")
        display_df = df.copy().sort_values('Profit %', ascending=False) #False → highest profit at the top
        st.dataframe(
            display_df.style.format({
                'Profit %': '{:+.1f}%',
                'Risk %': '{:.1f}%',
                'Sharpe': '{:.2f}',
                'Final $': '${:,.0f}',
                'Start $': '${:,.0f}',
                'Gain $': '${:+,.0f}' #+ sign makes profit/loss obvious
            }).background_gradient(subset=['Profit %'], cmap='RdYlGn'),
            use_container_width=True
        )
    
    
    # ------------------------------
    # 3. Correlation Heatmap
    # ------------------------------
    with tabs[2]:
        st.markdown("### Return Correlations")
        if len(returns_df.columns) > 1:
            corr = returns_df.corr()
            fig = px.imshow(corr, color_continuous_scale='RdYlGn', aspect='auto', labels=dict(color="Correlation"))
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, width='stretch')
            st.info("💡 High correlation (red) means stocks move together. Low/negative (blue) means diversification.")
        else:
            st.info("Need 2+ stocks for correlation analysis")
    
    # ------------------------------    
    # 4. Monte Carlo Simulation: To model uncertainty in future prices
    # ------------------------------
    with tabs[3]:
        st.markdown(f"### Monte Carlo Simulation: {best['Stock']}")
        st.caption("Simulating 300 possible price paths for the next year")
        ticker = best['Stock']
        returns = curves[ticker].pct_change().dropna() 
        mu, sigma = returns.mean(), returns.std()
        last_price = curves[ticker].iloc[-1]
        
        days = np.arange(252)  # Trading days in a year
        np.random.seed(42)
        sims = np.random.normal(mu, sigma, (252, 300))  #Generate random daily returns
        prices = last_price * np.cumprod(1 + sims, axis=0)  # Compute price paths
        
        p10, p50, p90 = np.percentile(prices[-1], [10, 50, 90])  # Percentiles:final day values only
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("📉 10th Percentile", f"${p10:,.0f}", "Worst case")
            st.metric("📊 50th Percentile", f"${p50:,.0f}", "Median")
            st.metric("📈 90th Percentile", f"${p90:,.0f}", "Best case")
        
        # Plot sample Monte Carlo paths
        with col2:
            fig = go.Figure()
            for i in range(50):
                fig.add_trace(go.Scatter(x=days, y=prices[:, i], opacity=0.15, line=dict(color='lightblue', width=1), showlegend=False, hoverinfo='skip'))
            
            # Add percentile lines
            fig.add_hline(y=p10, line_dash="dash", line_color="red", annotation_text="10%")
            fig.add_hline(y=p50, line_dash="dash", line_color="yellow", annotation_text="50%")
            fig.add_hline(y=p90, line_dash="dash", line_color="green", annotation_text="90%")
            
            fig.update_layout(template="plotly_dark", height=450, title=f"{ticker} - 1 Year Price Forecast", yaxis_title="Portfolio Value ($)", xaxis_title="Trading Days")
            st.plotly_chart(fig, width='stretch')

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption("Yahoo Finance | Real-time ticker search & goal-based stock selection")
