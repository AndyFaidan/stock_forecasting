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

# Streamlit App
st.title('Stock Price Prediction with LSTM')

with st.form(key='params_form'):
    st.markdown('<p class="params_text">Prediksi Data Real LSTM Saham KKGI.JK </p>', unsafe_allow_html=True)
    st.divider()

    optimizers = ['adam', 'adamax', 'sgd', 'rmsprop']
    optimizer = st.selectbox('Optimizer', optimizers, key='optimizer_selectbox')

    time_step = st.number_input('Time Step', min_value=1, max_value=500, value=30, step=1)

    epochs, batch_size = st.columns(2)
    with epochs:
        epochs_value = st.number_input('Epochs', min_value=1, value=10)
    with batch_size:
        batch_size_value = st.number_input('Batch Size', min_value=1, value=32)

    st.markdown('')
    train_button = st.form_submit_button('Train Model')
    st.markdown('')

if train_button:
    # Create Model
    model, df = create_LSTM_model(time_step, epochs_value, batch_size_value, optimizer)

    z1, z2 = st.columns((8, 2.5))
    with z1:
        fig = px.line(df, x='Date', y=['Real', 'Train Predict', 'Test Predict'], title='Stock Price Prediction')
        st.plotly_chart(fig)
        
        # Area chart
        fig_area = px.area(df, x='Date', y=['Real', 'Train Predict', 'Test Predict'], title='Area Chart of Stock Price Prediction')
        st.plotly_chart(fig_area)

        # Line chart with scatter plot
        fig_combined = go.Figure()
        fig_combined.add_trace(go.Scatter(x=df['Date'], y=df['Real'], mode='lines', name='Real'))
        fig_combined.add_trace(go.Scatter(x=df['Date'], y=df['Train Predict'], mode='markers', name='Train Predict', marker=dict(color='blue')))
        fig_combined.add_trace(go.Scatter(x=df['Date'], y=df['Test Predict'], mode='markers', name='Test Predict', marker=dict(color='red')))
        fig_combined.update_layout(title='Stock Price Prediction with Train and Test Points', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_combined)

    with z2:
        st.write(df[['Date', 'Real', 'Train Predict', 'Test Predict']])

    st.success('Model training completed!')
