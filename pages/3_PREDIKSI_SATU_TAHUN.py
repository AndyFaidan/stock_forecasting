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
from datetime import datetime, timedelta

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

def visualize_training_data(data_train):
    st.subheader('Training data Visualization')
    st.line_chart(data_train)

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

# Define the custom CSS for larger text and center alignment
st.markdown("""
    <style>
    .params_text {
        font-size: 30px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Create the form
st.sidebar.markdown('<p class="params_text">Prediksi Saham Satu Tahun Berdasarkan Data Historis</p>', unsafe_allow_html=True)
st.sidebar.divider()

optimizers = ['adam', 'adamax', 'sgd', 'rmsprop'] 
optimizer = st.sidebar.selectbox('Optimizer', optimizers, key='symbol_selectbox')

n_lookback = st.sidebar.number_input('Lookback', min_value=1, max_value=500, value=164, step=1)
n_forecast = st.sidebar.number_input('Forecast', min_value=10, max_value=730, value=365, step=1, key='period_no_input')

epochs = st.sidebar.number_input('Epochs', min_value=1, value=100)
batch_size = st.sidebar.number_input('Batch Size', min_value=1, value=32)

train_button = st.sidebar.button('Train Model')

if train_button:
    ticker = "KKGI.JK"
    start_date = "2021-01-01"
    end_date = "2024-06-24"  #
    data = yf.download(tickers=ticker, start=start_date, end=end_date)
                       
    scaler = MinMaxScaler(feature_range=(0, 1))
    y = data['Close'].values.reshape(-1, 1)
    y_scaled = scaler.fit_transform(y)

    X, Y = generate_sequences(y_scaled, n_lookback, n_forecast)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(n_lookback, 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, verbose=0)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    real_data = scaler.inverse_transform(y_scaled).flatten()
    train_data = pd.Series(index=data.index, data=[float('NaN')] * len(data))
    train_data[n_lookback:len(train_predict) + n_lookback] = train_predict.flatten()
    test_data = pd.Series(index=data.index, data=[float('NaN')] * len(data))
    test_data[len(train_predict) + (n_lookback * 2):len(y_scaled)] = test_predict.flatten()

    df = pd.DataFrame({
        'Tanggal': data.index,
        'Data Real': real_data,
        'Prediksi Latih': train_data.values,
        'Prediksi Uji': test_data.values,
        'Open': data['Open'],
        'High': data['High'],
        'Low': data['Low'],
        'Close': data['Close']
    })

    df['Kesalahan Latih'] = df.apply(lambda row: abs(row['Data Real'] - row['Prediksi Latih']) if pd.notnull(row['Prediksi Latih']) else None, axis=1)
    df['Kesalahan Uji'] = df.apply(lambda row: abs(row['Data Real'] - row['Prediksi Uji']) if pd.notnull(row['Prediksi Uji']) else None, axis=1)

    rata_kesalahan_latih = df['Kesalahan Latih'].mean()
    rata_kesalahan_uji = df['Kesalahan Uji'].mean()

    st.success('Pelatihan model selesai!')

    z1, z2 = st.columns((8, 2.5))
    with z1:
        fig = px.line(df, x='Tanggal', y=['Data Real', 'Prediksi Latih', 'Prediksi Uji'], title='Data Real dan Hasil Prediksi')
        fig.update_xaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig)

        fig_combined = go.Figure()
        fig_combined.add_trace(go.Scatter(x=df['Tanggal'], y=df['Data Real'], mode='lines', name='Data Real', fill='tozeroy'))
        fig_combined.add_trace(go.Scatter(x=df['Tanggal'], y=df['Prediksi Latih'], mode='lines', name='Prediksi Latih', fill='tonexty'))
        fig_combined.add_trace(go.Scatter(x=df['Tanggal'], y=df['Prediksi Uji'], mode='lines', name='Prediksi Uji', fill='tonexty'))
        fig_combined.update_layout(
            title='Perbandingan Data Real dan Prediksi',
            xaxis_title='Tanggal',
            yaxis_title='Harga',
            legend_title='Legenda',
            template='plotly_white'
        )
        fig_combined.update_xaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig_combined)

    with z2:
        st.metric(label="Rata-rata Kesalahan Latih", value=f"{rata_kesalahan_latih:.2f}")
        st.metric(label="Rata-rata Kesalahan Uji", value=f"{rata_kesalahan_uji:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Forecast Data")
        st.write(df[df['Prediksi Uji'].notna()])

    with col2:
        st.subheader("Description of Results")
        st.write(df.describe())

    st.subheader("Forecast Characteristics")

    mean_value = df['Data Real'].mean()
    df['Characteristic'] = np.where(df['Data Real'].fillna(df['Prediksi Uji']) >= mean_value, 'high', 'low')

    fig_characteristics = go.Figure()
    high_forecasts = df[df['Characteristic'] == 'high']
    low_forecasts = df[df['Characteristic'] == 'low']

    combined_forecasts = pd.concat([high_forecasts, low_forecasts], axis=0).sort_index()

    if not combined_forecasts.empty:
        fig_characteristics.add_trace(go.Scatter(x=combined_forecasts['Tanggal'], y=combined_forecasts['Prediksi Uji'],
                                                 mode='lines', line=dict(color='blue'), name='Forcast'))

    fig_characteristics.add_trace(go.Scatter(x=df['Tanggal'], y=[mean_value]*len(df['Tanggal']),
                                             mode='lines', name='Mean', line=dict(color='green', width=2)))

    fig_characteristics.update_layout(title='Forecast Grafik chart', xaxis_title='Date',
                                      yaxis_title='Forecast', showlegend=True,
                                      xaxis=dict(range=['2024-06-01', '2025-07-01']))

    st.plotly_chart(fig_characteristics, use_container_width=True)
