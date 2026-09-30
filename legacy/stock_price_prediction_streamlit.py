import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import math

st.title("📈 Stock Price Prediction with Forecasting (LSTM)")

ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL, GOOGL):", "GOOGL")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

def create_dataset(dataset, time_step=100):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

if st.button("Predict"):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        st.error("No data found. Please try a different ticker or date range.")
    else:
        st.subheader("📊 Raw Stock Data")
        st.write(stock_data.tail())

        # Plot all columns
        for column in stock_data.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(stock_data[column])
            ax.set_title(f"{column} of {ticker}")
            ax.set_xlabel("Date")
            ax.set_ylabel(column)
            st.pyplot(fig)

        # Preprocessing
        df = stock_data.reset_index()['Close']
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = scaler.fit_transform(np.array(df).reshape(-1, 1))

        train_size = int(len(df_scaled) * 0.7)
        test_size = len(df_scaled) - train_size
        train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]

        time_step = 100
        X_train, Y_train = create_dataset(train_data, time_step)
        X_test, Y_test = create_dataset(test_data, time_step)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        if X_test.size == 0:
            st.error("X_test is empty. Not enough test data. Try a wider date range.")
            st.stop()
        else:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # LSTM Model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=1)

        # Prediction
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        Y_train_rescaled = scaler.inverse_transform(Y_train.reshape(-1, 1))
        Y_test_rescaled = scaler.inverse_transform(Y_test.reshape(-1, 1))

        # Model Evaluation
        st.subheader("📊 Model Evaluation")
        train_rmse = math.sqrt(mean_squared_error(Y_train_rescaled, train_predict))
        test_rmse = math.sqrt(mean_squared_error(Y_test_rescaled, test_predict))
        r2 = r2_score(Y_test_rescaled, test_predict)

        st.write(f"Train RMSE: {train_rmse:.2f}")
        st.write(f"Test RMSE: {test_rmse:.2f}")
        st.write(f"R² Score: {r2:.2f}")

        # Combined prediction plot (actual, train, test)
        train_plot = np.empty_like(df_scaled)
        train_plot[:, :] = np.nan
        train_plot[time_step:len(train_predict)+time_step, :] = train_predict

        test_plot = np.empty_like(df_scaled)
        test_plot[:, :] = np.nan
        test_plot[len(train_predict)+(time_step*2)+1:len(df_scaled)-1, :] = test_predict

        full_data = scaler.inverse_transform(df_scaled)

        st.subheader("📈 Actual vs Train/Test Predictions")
        fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
        ax_pred.plot(full_data, label='Actual Price')
        ax_pred.plot(train_plot, label='Train Prediction', color='orange')
        ax_pred.plot(test_plot, label='Test Prediction', color='green')
        ax_pred.set_title('Stock Price Prediction')
        ax_pred.set_xlabel('Time')
        ax_pred.set_ylabel('Price')
        ax_pred.legend()
        st.pyplot(fig_pred)

        # Forecasting next 7 days
        x_input = test_data[-time_step:].reshape(1, -1)
        temp_input = list(x_input[0])
        lst_output = []
        n_steps = time_step
        i = 0
        while i < 7:
            if len(temp_input) > n_steps:
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, n_steps, 1)
                yhat = model.predict(x_input, verbose=0)
                temp_input.append(yhat[0][0])
                temp_input = temp_input[1:]
                lst_output.append(yhat[0][0])
                i += 1
            else:
                x_input = np.array(temp_input).reshape(1, n_steps, 1)
                yhat = model.predict(x_input, verbose=0)
                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])
                i += 1

        lst_output = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

        # Plot forecast
        st.subheader("📈 Forecast for Next 7 Days")
        fig2 = plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(df)), scaler.inverse_transform(df_scaled), label='Original')
        plt.plot(np.arange(len(df), len(df) + 7), lst_output, label='Forecast')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.title(f"{ticker} - Forecasting Next 7 Days")
        plt.legend()
        st.pyplot(fig2)
