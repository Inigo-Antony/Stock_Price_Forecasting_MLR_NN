📈 Stock Price Prediction using Neural Networks & Linear Regression

Predicting stock prices using lag features, interpolation-driven weekly lags, and a comparison of Neural Network vs Multiple Linear Regression models.

🔍 Project Overview

This project explores short-term stock price prediction using historical price data.
The workflow includes:

Pulling 10 years of stock data from Yahoo Finance

Engineering lag features (1-day lag, 1-week lag with holiday gap estimation)

Training two models:

Neural Network (Keras)

Multiple Linear Regression (sklearn)

Comparing forecasting accuracy using MAPE

Visualizing prediction performance

This project serves as an introductory but complete pipeline for time-series ML modeling.

🧠 Key Features

Automated data download via yfinance

Holiday-aware lag creation using business-day interpolation

Neural Network model with ReLU-activated hidden layers

Multiple Linear Regression baseline

MAPE comparison for both models

Price prediction plotting

🗂️ Project Structure
├── stock_prediction.ipynb  # Notebook version (optional)
├── main.py                 # Python script 
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies

🛠️ Tech Stack
Component	Library / Tool
Data Source	yfinance
Data Processing	pandas, numpy
Visualization	matplotlib, seaborn
Models	Keras (Sequential), sklearn.LinearRegression
📥 Data Download

The script automatically downloads Stock data (SYM=AAPL, etc):

data = yf.download("SYM", start="2014-01-01", end="2024-01-01", auto_adjust=False)
data.to_csv("SYM.csv")

🧩 Feature Engineering
1. Lag-1 Feature

Price from the previous day.

data_adj['lag_1'] = data_adj['price'].shift(1)

2. Lag-5-Week Feature (same weekday previous week)

Since markets close during weekends & holidays, a business-day index is generated and interpolated:

all_business_days = pd.date_range(start=data_adj.index.min(), end=data_adj.index.max(), freq='B')
data_filled = data_adj.reindex(all_business_days)
data_filled['price'] = data_filled['price'].interpolate(method='time')
data_filled['lag_5_week'] = data_filled['price'].shift(5)

3. Day-of-Week Feature

Basic cyclical pattern enhancer.

🧪 Modeling Approach
🔹 Multiple Linear Regression (baseline)

No intercept (explicit constant term included)

Features:

Constant

Lag-1

Lag-5-Week

🔹 Neural Network (main model)

Architecture:

Dense(8, relu)

Dense(8, relu)

Dense(1, linear)

Optimizer: Adam
Loss: MSE
Epochs: 20

📊 Performance Metrics

The project uses MAPE (Mean Absolute Percentage Error):

NN Training MAPE: xx.xx%
NN Testing MAPE:  xx.xx%
MLR Training MAPE: xx.xx%
MLR Testing MAPE: xx.xx%


(Your actual printed values will appear after code execution.)

📉 Prediction Plot

The script generates:

Actual test prices

Neural Network predictions

Linear Regression predictions

A clear visual comparison of forecast performance.

▶️ How to Run
1. Clone the Repository
git clone https://github.com/Inigo-Antony/Stock_Price_Forecasting_MLR_NN.git
cd Stock_Price_Forecasting_MLR_NN

2. Install Dependencies
pip install -r requirements.txt

3. Run Script
python main.py


Or open the notebook:

jupyter notebook

📦 Requirements

Your requirements.txt should include:

yfinance
pandas
numpy
matplotlib
seaborn
scikit-learn
keras
tensorflow

🚀 Future Improvements

Here are possible upgrades for version-2:

Add LSTM/GRU models for deeper time-series understanding

Add hyperparameter tuning (Keras Tuner / Optuna)

Include more technical indicators (RSI, MACD, EMA…)

Implement walk-forward validation

Integrate real-time prediction API

📜 License

MIT License (or your preferred license)

🤝 Contributing

PRs and suggestions are welcome.
If you use this repo for research or learning, consider starring ⭐ the project.
