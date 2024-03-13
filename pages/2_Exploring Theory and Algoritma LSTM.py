import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
from datetime import datetime

# Setting configuration options
st.set_option('deprecation.showPyplotGlobalUse', False)

# Page layout
st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")

# Styling with custom CSS
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header and introduction
st.header(" UNDERSTANDING LSTM ALGORITHM ")
st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2")
st.markdown(
    """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <hr>
    <div class="card mb-3">
        <div class="card">
            <<div class="card-body">
                <h3 class="card-title" style="color:#007710;"><strong>⏱ PEMAHAMAN ALGORITMA LSTM UNTUK PREDIKSI SAHAM</strong></h3>
                <p class="card-text">Long Short-Term Memory (LSTM) adalah jenis arsitektur recurrent neural network (RNN) yang sangat cocok untuk masalah prediksi urutan. Dalam konteks prediksi saham, LSTM dapat mempelajari pola dan ketergantungan pada harga saham historis untuk membuat prediksi di masa depan.</p>
                <p class="card-text">Model LSTM sangat efektif dalam menangkap ketergantungan jangka panjang, sehingga cocok untuk tugas peramalan time series. Model ini telah banyak digunakan di berbagai domain, termasuk keuangan, untuk memprediksi harga saham berdasarkan data historis.</p>
                <p class="card-text">Dasbor ini menggunakan LSTM untuk memprediksi harga saham untuk simbol saham dan rentang waktu yang ditentukan. Mean Squared Error (MSE) digunakan sebagai metrik untuk mengevaluasi kinerja model.</p>
            </div>
        </div>
    </div>
    <style>
        [data-testid=stSidebar] {
             color: white;
             text-size: 24px;
        }
    </style>
    """, unsafe_allow_html=True
)


# Date range selector for stock data in the sidebar
start_date = st.sidebar.date_input('Start Date', datetime(2020, 1, 1))
end_date = st.sidebar.date_input('End Date', datetime.today())

# Ticker symbol definition
ticker_symbol = 'KKGI.JK'

# Load stock data using yfinance with the specified ticker symbol
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Select features for LSTM
features_lstm = stock_data[['Adj Close']]

c1, c2 = st.columns(2)
with c1:
# Display stock data expander
    with st.expander("⬇ STOCK DATA FOR LSTM SUMMARY"):

        st.write(features_lstm.describe())

# Feature engineering: Extract year from index
stock_data['Year'] = stock_data.index.year
with c2:
    # Display stock data for LSTM with year-wise summary statistics
    with st.expander("⬇ STOCK DATA FOR LSTM WITH YEAR-WISE SUMMARY STATISTICS"):
        # Summary statistics by year
        yearwise_stats = stock_data.groupby('Year')['Adj Close'].describe()
        st.write("### Year-wise Summary Statistics:")
        st.write(yearwise_stats)

with st.expander("⬇ STOCK DATA FOR LSTM WITH YEAR-WISE SUMMARY STATISTICS LINE CHART"):
    # Plot stock data by year using Plotly Express
    fig_stock_data_yearwise = px.line(stock_data, x=stock_data.index, y='Adj Close', color='Year',
                                    title='Stock Prices Over Time (Year-wise)', labels={'Adj Close': 'Stock Price'})
    fig_stock_data_yearwise.update_layout(xaxis_title='Date', yaxis_title='Adj Close Price', width=1240, height=500)
    st.plotly_chart(fig_stock_data_yearwise)


# Feature scaling for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features_lstm)


