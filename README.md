# 🧠 Crypto Trading Engine

A Python-based crypto trading engine designed to analyze market data, generate trade signals using multiple advanced strategies, and provide coin and strategy recommendations. This tool is built with extensibility in mind and serves as the backend for a broader public-facing web application.

## 🚀 Features

- 🔁 Strategy Framework with Modular Design
- 📈 Technical Indicators:
  - MACD
  - RSI
  - Moving Average Crossovers
  - Bollinger Bands
  - ROC
  - Fibonacci Retracement
  - Elliott Wave Visualization
  - Ichimoku Cloud
  - OBV (On-Balance Volume)
  - ADX (Average Directional Index)
  - Volume-Weighted Average Price (VWAP)
  - Candlestick Pattern Recognition
  - Plus More ...
- 📦 Real-time data pulled via the **Coinbase API**
- 🧪 Strategy testing on historical candle data
- 🧠 Trade recommendations based on selected strategy logic
- 🛠 Clean architecture and easily extendable for more strategies

## 🧰 Tech Stack

- **Python 3.11+**
- **Coinbase Advanced Trade API**
- **Pandas** for data manipulation
- **Matplotlib / Plotly** (for visualization)
- (Optional) **Flask API** to connect with Next.js frontend

## 🗂 Project Structure
crypto-trading-engine/
│
├── strategies/ # All trading strategies live here
│ ├── macd.py
│ ├── rsi.py
│ ├── bollinger_bands.py
│ └── ...
│
├── utils/
│ ├── engine.py # Main engine to run selected strategy
│ ├── data_fetcher.py # Handles API requests to Coinbase
│ └── utils.py # Helper functions
│
├── data/ # All logs/exports as well as integration and backtests
│
├── env/ # API keys (ignored by Git)
│ └── keys.env
│
├── requirements.txt
├── README.md
└── main.py # Entrypoint script

## 🧪 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/crypto-trading-engine.git
cd crypto-trading-engine
