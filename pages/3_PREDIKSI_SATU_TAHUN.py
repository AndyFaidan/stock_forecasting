import streamlit as st
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import random

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

if "symbols_list" not in st.session_state:
    st.session_state.symbols_list = None

st.set_page_config(
    layout='wide',
    page_title='LSTM FORECAST'
)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.experimental_memo
def load_data(ticker, period='4y'):
    return yf.download(ticker, period=period)

@st.experimental_singleton
def create_model(n_lookback, n_forecast):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)),
        LSTM(units=50),
        Dense(n_forecast)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def generate_sequences(data, n_lookback, n_forecast):
    X = []
    Y = []

    for i in range(n_lookback, len(data) - n_forecast + 1):
        X.append(data[i - n_lookback: i])
        Y.append(data[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def evaluate_model(model, X_train, Y_train, scaler):
    train_predictions = model.predict(X_train)
    train_predictions = scaler.inverse_transform(train_predictions.reshape(-1, 1))
    Y_train_inv = scaler.inverse_transform(Y_train.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(Y_train_inv, train_predictions))
    mae = mean_absolute_error(Y_train_inv, train_predictions)
    mape = np.mean(np.abs((Y_train_inv - train_predictions) / Y_train_inv)) * 100
    mse = mean_squared_error(Y_train_inv, train_predictions)

    return rmse, mae, mape, mse, train_predictions

with st.form(key='params_form'):
    st.markdown('<p class="params_text">Forcasting Dengan LSTM</p>', unsafe_allow_html=True)
    st.divider()

    optimizers = ['adam', 'adamax', 'sgd', 'rmsprop'] 
    optimizer = st.selectbox('Optimizer', optimizers, key='symbol_selectbox')

    n_lookback, n_forecast = st.columns(2)
    with n_lookback:
        n_lookback = st.number_input('Lookback', min_value=1, max_value=500, value=164, step=1)
    with n_forecast:
        n_forecast = st.number_input('Forecast', min_value=10, max_value=730, value=365, step=1, key='period_no_input')

    epochs, batch_size = st.columns(2)
    with epochs:
        epochs = st.number_input('Epochs', min_value=1, value=200)
    with batch_size:
        batch_size = st.number_input('Batch Size', min_value=1, value=32)

    st.markdown('')
    train_button = st.form_submit_button('Train Model')
    st.markdown('')

if train_button:
    data = load_data('KKGI.JK')
    scaler = MinMaxScaler(feature_range=(0, 1))
    y = data['Close'].values.reshape(-1, 1)
    y_scaled = scaler.fit_transform(y)

    X, Y = generate_sequences(y_scaled, n_lookback, n_forecast)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    model = create_model(n_lookback, n_forecast)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    last_sequence = y_scaled[-n_lookback:].reshape(1, n_lookback, 1)
    Y_ = model.predict(last_sequence)
    Y_ = scaler.inverse_transform(Y_.flatten().reshape(-1, 1)).flatten()

    df_past = data[['Close']].reset_index()
    df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_
    df_future['Actual'] = np.nan

    future_results = pd.concat([df_past, df_future]).set_index('Date')
    future_results = future_results[['Forecast']].dropna()

    st.success('Model training completed!')

    # Line chart of Actual vs Forecast
    fig = px.line(future_results.reset_index(), x='Date', y=['Forecast'], title='Forecast')
    st.plotly_chart(fig, use_container_width=True)

    st.write(future_results)

    st.write(future_results.describe())
