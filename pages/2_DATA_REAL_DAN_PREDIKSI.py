import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import streamlit as st
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    layout='wide',
    page_title='LSTM FORECAST'
)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function to create LSTM model
def create_LSTM_model(time_step, epochs, batch_size, optimizer):
    # Step 1: Load the Data from Yahoo Finance
    data = yf.download('KKGI.JK', start='2020-01-01', end=datetime.now())

    # Step 2: Preprocess the Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['Close']])

    # Splitting data into training and testing sets (80% train, 20% test)
    train_size = int(len(data_scaled) * 0.8)
    train, test = data_scaled[:train_size], data_scaled[train_size:]

    # Step 3: Create Dataset
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return pd.DataFrame(dataX), pd.Series(dataY)

    X_train, y_train = create_dataset(train, time_step)
    X_test, y_test = create_dataset(test, time_step)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Step 4: Build the LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Step 5: Train the Model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0)

    # Step 6: Make Predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform to get actual values
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Prepare data for saving to CSV
    real_data = scaler.inverse_transform(data_scaled).flatten()
    train_data = pd.Series(index=data.index, data=[float('NaN')] * len(data))
    train_data[time_step:len(train_predict) + time_step] = train_predict.flatten()
    test_data = pd.Series(index=data.index, data=[float('NaN')] * len(data))
    test_data[len(train_predict) + (time_step * 2):len(data_scaled)] = test_predict.flatten()

    df = pd.DataFrame({
        'Date': data.index,
        'Real': real_data,
        'Train Predict': train_data.values,
        'Test Predict': test_data.values,
        'Open': data['Open'],
        'High': data['High'],
        'Low': data['Low'],
        'Close': data['Close']
    })

    return model, df

with st.form(key='params_form'):
    st.markdown("""
        <div style="display: flex; justify-content: center;">
            <h2>Perbandingan Data Real dan Prediksi LSTM</h2>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    optimizers = ['adam', 'adamax', 'sgd', 'rmsprop']
    optimizer = st.selectbox('Optimizer', optimizers, key='optimizer_selectbox')

    time_step = st.number_input('Lookback', min_value=1, max_value=500, value=164, step=1)

    epochs, batch_size = st.columns(2)
    with epochs:
        epochs_value = st.number_input('Epochs', min_value=1, value=100)
    with batch_size:
        batch_size_value = st.number_input('Batch Size', min_value=1, value=32)

    st.markdown('')
    train_button = st.form_submit_button('Train Model')
    st.markdown('')

if train_button:
    # Create Model
    model, df = create_LSTM_model(time_step, epochs_value, batch_size_value, optimizer)

    # Hitung jarak nilai antara real dan prediksi
    df['Train Error'] = df.apply(lambda row: abs(row['Real'] - row['Train Predict']) if pd.notnull(row['Train Predict']) else None, axis=1)
    df['Test Error'] = df.apply(lambda row: abs(row['Real'] - row['Test Predict']) if pd.notnull(row['Test Predict']) else None, axis=1)

    # Hitung rata-rata kesalahan
    avg_train_error = df['Train Error'].mean()
    avg_test_error = df['Test Error'].mean()

    st.success('Model training completed!')

    z1, z2 = st.columns((8, 2.5))
    with z1:
        # Line chart for data comparison
        fig = px.line(df, x='Date', y=['Real', 'Train Predict', 'Test Predict'], title='Data Real dan Hasil Prediksi')
        st.plotly_chart(fig)
        
        # Line chart with scatter plot and area
        fig_combined = go.Figure()
        fig_combined.add_trace(go.Scatter(x=df['Date'], y=df['Real'], mode='lines', name='Real Data', fill='tozeroy'))
        fig_combined.add_trace(go.Scatter(x=df['Date'], y=df['Train Predict'], mode='lines', name='Train Predict', fill='tonexty'))
        fig_combined.add_trace(go.Scatter(x=df['Date'], y=df['Test Predict'], mode='lines', name='Test Predict', fill='tonexty'))
        fig_combined.update_layout(
            title='Comparison of Real Data and Predictions',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            template='plotly_white'
        )
        st.plotly_chart(fig_combined)

        
