
# 📈 Smart Stock Analyzer

**Smart Stock Analyzer** is an interactive Streamlit web application that enables investors to **search, analyze, compare, and rank stocks** using live market data and goal-driven investment logic.
It combines performance metrics, risk analysis, visual insights, and Monte Carlo simulations to support **data-driven investment decisions** through a clean, user-friendly interface.

---

## 🚀 Features

### 🔍 Live Stock Discovery

* Search stocks by **company name or ticker symbol**
* Powered by **Yahoo Finance** with fallback search logic

### 📂 Portfolio Management

* Add and remove multiple stocks dynamically
* Persistent session-based stock selection

### 🎯 Goal-Based Analysis

Choose the best stock based on:

* **Balanced (Risk-Adjusted)** – Sharpe Ratio
* **Maximum Return**
* **Minimum Risk**

### 📊 Performance & Risk Metrics

* Total Return (%)
* Annualized Volatility (%)
* Sharpe Ratio
* Final Portfolio Value
* Absolute Gain / Loss

### 📈 Interactive Visualizations

* Portfolio growth curves over time
* Ranked comparison table of all stocks
* Correlation heatmap for diversification insights
* Monte Carlo simulations for future price uncertainty

### 🎨 User-Friendly UI

* Dark gradient theme
* Real-time progress feedback
* Clean metric cards and visual hierarchy

---

## 📌 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/surbhi1us/smart-stock-analyzer.git
cd smart-stock-analyzer
```

### 2️⃣ Create & activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## 🖥️ Usage

Run the app using:

```bash
streamlit run app.py
```

### Workflow

1. Search stocks via sidebar
2. Add multiple stocks to compare
3. Set investment amount, date range, and goal
4. Click **“ANALYZE & PICK BEST”**
5. Explore rankings, metrics, and simulations via tabs

---

## 📊 Analysis Tabs

| Tab              | Description                                |
| ---------------- | ------------------------------------------ |
| 📈 Growth Curves | Portfolio value trends over time           |
| 📊 All Stocks    | Ranked comparison by return, risk & Sharpe |
| 🔗 Correlations  | Heatmap showing stock return correlations  |
| 🎲 Monte Carlo   | 1-year simulated price scenarios           |

---

## ⚡ Tech Stack

* **Python**
* **Streamlit** – Interactive web UI
* **yfinance** – Real-time & historical stock data
* **Plotly** – Interactive visualizations
* **Pandas & NumPy** – Financial calculations
* **Requests** – Live ticker search

---

## ⚠️ Important Notes

* Internet connection required for live data
* Yahoo Finance may throttle excessive requests
* Monte Carlo results are **probabilistic**, not financial advice

---

## 🧠 Planned Enhancements (Roadmap)

* 🔢 **Stock Ranking System** (Top-N ranking instead of only best pick)
* 🤖 **AI Chatbot** for:

  * Stock explanations
  * Metric interpretation
  * Beginner guidance
* 🖱️ Enhanced hover tooltips & UX micro-interactions
* 📊 Portfolio-level optimization (multi-asset allocation)
* 🔐 User authentication & saved portfolios
* 🌍 International market support

---

## 📌 License

MIT License © 2026

---

## 🔗 References

* Yahoo Finance (via `yfinance`)
* Streamlit Documentation
* Plotly Python Docs

---
