# StockPort Stock Analyzer

StockPort is a Streamlit-based stock analysis app built to help users compare multiple stocks in one place. It lets the user search stocks, create a small portfolio, analyze performance based on a chosen goal, and explore charts, sentiment, correlations, Monte Carlo scenarios, and simple AI-based explanations. The main idea behind this project was to make stock analysis easier to understand for beginners instead of showing only technical finance metrics.

**Live App:** [StockPort Stock Analyzer · Streamlit](https://stockport-analyzer.streamlit.app/)

---

## Project Overview

This project compares selected stocks using historical Yahoo Finance data and ranks them based on one of three goals:

- Balanced (risk-adjusted performance using Sharpe ratio)
- Maximum return
- Minimum risk

The app also includes a sentiment and news section, a correlations section, a Monte Carlo simulation tab, and an AI assistant that explains metrics in simple words.

---

## Features

- Search stocks by company name or ticker
- Add and remove stocks from a portfolio in the sidebar
- Analyze stocks based on a selected investment goal
- View the top-ranked stock in a summary card
- Compare all selected stocks in a ranked table
- Explore portfolio growth curves
- See recent news sentiment using VADER sentiment analysis
- Check stock return correlations
- Run a 1-year Monte Carlo simulation
- Ask the AI assistant beginner-friendly finance questions

---

## Tech Stack

- Python
- Streamlit
- yfinance
- pandas
- numpy
- plotly
- requests
- nltk
- openai-compatible client for Groq API

---

## Project Structure

```bash
.
├── app.py
├── style.css
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/surbhi1us/smart-stock-analyzer.git
cd smart-stock-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

On macOS / Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## requirements.txt

Use the following dependencies:

```txt
streamlit>=1.28.0
yfinance>=0.2.32
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
openai>=1.0.0
requests>=2.31.0
nltk>=3.8.1
```

---

## Running the App

Start the app with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```bash
http://localhost:8501
```

---

## How to Use

1. Search a company name or ticker from the sidebar
2. Add one or more stocks to the portfolio
3. Choose the investment amount
4. Select the analysis start date
5. Choose the investment goal
6. Click **Run Analysis**
7. Explore the results using the tabs

---

## Tabs in the App

### Overview
Shows the top stock, key summary metrics, market mood, and one short insight.

### Growth Curves
Shows how the investment would have grown over time for each selected stock.

### Sentiment & News
Shows overall news sentiment, stock-wise sentiment scores, and recent headlines.

### Correlations
Shows how strongly the selected stocks move together.

### Monte Carlo Simulation
Shows possible future paths for the best-ranked stock using historical return behavior.

### AI Assistant
Lets the user ask basic questions about metrics, results, and investment concepts in simple language.

---

## Notes

- This app uses Yahoo Finance data through `yfinance`
- News sentiment is calculated using NLTK VADER
- The AI assistant needs a valid `GROQ_API_KEY`
- Monte Carlo simulation is only a probabilistic scenario tool, not a prediction
- This project is for learning and analysis purposes only, not financial advice

---

## API Key Setup

To use the AI assistant, set your Groq API key as an environment variable.

On Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_api_key_here"
streamlit run app.py
```

On macOS / Linux:

```bash
export GROQ_API_KEY="your_api_key_here"
streamlit run app.py
```

---

## What I Learned

While building this project, I worked on:

- using Streamlit for interactive dashboards
- fetching and cleaning stock data
- comparing return and risk metrics
- using sentiment analysis on financial news
- designing a cleaner custom UI with CSS
- adding an AI assistant with external API support

---

## Future Improvements

- Better portfolio-level analysis
- Export report as PDF or CSV
- More detailed stock explanations
- Better mobile responsiveness
- Save portfolio choices between sessions
- Add Indian stock market specific enhancements

---

## Disclaimer

This project is made for educational purposes only. It should not be used as direct investment advice.

---

## License

MIT License
