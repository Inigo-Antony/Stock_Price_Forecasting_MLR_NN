import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense


def download_data():
    """Download AAPL stock data and save as CSV."""
    print("Downloading data...")
    data = yf.download("AAPL", start="2014-01-01", end="2024-01-01", auto_adjust=False)
    data.to_csv("AAPL.csv")
    print("Data downloaded and saved to AAPL.csv")
    return data


def prepare_data(data):
    """Prepare dataset with lag features and return processed DataFrame."""
    data_adj = data[['Adj Close']].copy()
    data_adj.columns = ['price']

    # Lag-1 feature
    data_adj['lag_1'] = data_adj['price'].shift(1)
    data_adj.dropna(inplace=True)

    # Create business-day index and interpolate to estimate holiday prices
    all_business_days = pd.date_range(start=data_adj.index.min(),
                                      end=data_adj.index.max(),
                                      freq='B')
    data_filled = data_adj.reindex(all_business_days)
    data_filled['price'] = data_filled['price'].interpolate(method='time')

    # Lag-5 week feature
    data_filled['lag_5_week'] = data_filled['price'].shift(5)

    # Merge back to original trading-day-only dataset
    data_adj = data_adj.join(data_filled[['lag_5_week']], how='left')
    data_adj.dropna(inplace=True)

    # Day-of-week feature
    data_adj['day_of_the_week'] = data_adj.index.dayofweek

    return data_adj


def split_data(data_adj):
    """Split dataset into training and testing sets."""
    split_index = int(len(data_adj) * 0.75)
    train_data = data_adj.iloc[:split_index]
    test_data = data_adj.iloc[split_index:]

    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")

    # Create features matrix
    features = np.zeros((len(data_adj), 3))
    for i in range(len(data_adj)):
        features[i, 0] = 1  # constant term
        features[i, 1] = data_adj['lag_1'].iloc[i]
        features[i, 2] = data_adj['lag_5_week'].iloc[i]

    X_train = features[:split_index]
    y_train = data_adj['price'].iloc[:split_index].values
    X_test = features[split_index:]
    y_test = data_adj['price'].iloc[split_index:].values

    return X_train, y_train, X_test, y_test, test_data


def build_nn_model():
    """Build and compile neural network model."""
    model = Sequential()
    model.add(Dense(8, input_dim=3, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    return 100 * np.mean(np.abs((y_pred.flatten() - y_true.flatten()) / y_true.flatten()))


def main():
    # Step 1: Download and prepare data
    data = download_data()
    data_adj = prepare_data(data)

    # Step 2: Split data
    X_train, y_train, X_test, y_test, test_data = split_data(data_adj)

    # Step 3: Train Neural Network
    print("Training Neural Network...")
    model_nn = build_nn_model()
    model_nn.fit(X_train, y_train, epochs=20, batch_size=5, verbose=1)

    # Step 4: Train MLR
    print("Training Multiple Linear Regression...")
    MLR = linear_model.LinearRegression(fit_intercept=False)
    MLR.fit(X_train, y_train)

    # Step 5: Predictions
    y_pred_nn = model_nn.predict(X_test)
    y_pred_mlr = MLR.predict(X_test)
    y_train_pred_nn = model_nn.predict(X_train)
    y_train_pred_mlr = MLR.predict(X_train)

    # Step 6: Evaluation
    print("\n==== MAPE Scores ====")
    print(f"NN Training MAPE: {calculate_mape(y_train, y_train_pred_nn):.2f}%")
    print(f"NN Testing MAPE:  {calculate_mape(y_test, y_pred_nn):.2f}%")
    print(f"MLR Training MAPE: {calculate_mape(y_train, y_train_pred_mlr):.2f}%")
    print(f"MLR Testing MAPE:  {calculate_mape(y_test, y_pred_mlr):.2f}%")

    # Step 7: Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, y_test, label='Actual Price')
    plt.plot(test_data.index, y_pred_nn, label='NN Predicted Price')
    plt.plot(test_data.index, y_pred_mlr, label='MLR Predicted Price', linestyle='--')
    plt.title('AAPL Price Prediction (NN vs MLR)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
