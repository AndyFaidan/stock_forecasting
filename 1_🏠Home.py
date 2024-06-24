import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
from datetime import datetime, date
from PIL import Image
from streamlit_lightweight_charts import renderLightweightCharts
import plotly.graph_objects as go

def main():
    # Set page configuration
    st.set_page_config(layout="wide", page_title="KKGI.JK DashBoard For LSTM")

    # Load custom styles
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Aplikasi Streamlit
    st.title('PREDIKSI ANALISIS LSTM PADA SAHAM KKGI.JK')

    # Sidebar untuk memilih tahun mulai
    start_year = st.sidebar.selectbox("Periode Forecast", options=range(2021, 2025), index=0)

    # Tentukan tanggal mulai dan akhir
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2024, 6, 23)

    # Fetch data from Yahoo Finance for KKGI.JK from 2021
    ticker = "KKGI.JK"
    data = yf.download(tickers=ticker, start=start_date, end=end_date)

    def add_range_selector(fig):
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label='1m', step='month', stepmode='backward'),
                        dict(count=6, label='6m', step='month', stepmode='backward'),
                        dict(count=1, label='YTD', step='year', stepmode='todate'),
                        dict(count=1, label='1y', step='year', stepmode='backward'),
                        dict(step='all')
                    ]
                )
            ),
            xaxis_type='date'
        )

    # Generate sparkline data
    np.random.seed(1)
    y = data['Close'].values[-24:]  # Use the last 24 closing prices for sparkline
    x = np.arange(len(y))
    fig = px.line(x=x, y=y, width=400, height=100)

    xmin = x[0]
    xmax = x[-1]
    ymin = round(y[0], 1)
    ymax = round(y[-1], 1)

    layout = {
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "yaxis": {"visible": False},
        "xaxis": {
            "nticks": 2,
            "tickmode": "array",
            "tickvals": [xmin, xmax],
            "ticktext": [f"{ymin} <br> {xmin}", f"{ymax} <br> {xmax}"],
            "title_text": None
        },
        "showlegend": False,
        "margin": {"l":4,"r":4,"t":0, "b":0, "pad": 4}
    }
    config = {'displayModeBar': False}

    fig.update_layout(layout)

    # Row A: Logo and basic metrics
    a1, a2, a3 = st.columns(3)

    # Calculate changes
    highest_open_price_change = data['Open'].max() - data['Open'].iloc[-2]
    highest_high_price_change = data['High'].min() - data['High'].iloc[-2]
    highest_volume_change = data['Volume'].min() - data['Volume'].iloc[-2]

    # Sparkline data untuk Open
    sparkline_data_open = data['Open'].iloc[-24:]  # Mengambil 24 harga open terakhir
    x_sparkline_open = np.arange(len(sparkline_data_open))

    # Sparkline data untuk Close
    sparkline_data_high = data['High'].iloc[-24:]  # Mengambil 24 harga penutupan terakhir
    x_sparkline_high = np.arange(len(sparkline_data_high))

    # Sparkline data untuk Volume
    sparkline_data_volume = data['Volume'].iloc[-24:]  # Mengambil 24 volume terakhir
    x_sparkline_volume = np.arange(len(sparkline_data_volume))

    # Metrik untuk Open, Close, dan Volume
    with a1:
        if highest_open_price_change >= 2021:
            st.metric("Highest Open Price", f"${data['Open'].max():,.2f}", delta=f"+{highest_open_price_change:.2f}")
        else:
            st.metric("Highest Open Price", f"${data['Open'].max():,.2f}", delta=f"{highest_open_price_change:.2f}")
        # Generate sparkline untuk Open
        fig_sparkline_open = px.line(x=x_sparkline_open, y=sparkline_data_open, width=150, height=50)
        fig_sparkline_open.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
                "yaxis": {"visible": False},
                "xaxis": {"visible": False},
                "showlegend": False,
                "margin": {"l":4,"r":4,"t":0, "b":0, "pad": 4}
            }
        )
        st.plotly_chart(fig_sparkline_open, use_container_width=True)
        st.markdown("<div style='text-align:center; color:green;'>OPEN KKGI.JK</div>", unsafe_allow_html=True)

    with a2:
        if highest_high_price_change >= 2021:
            st.metric("Highest High Price", f"${data['High'].max():,.2f}", delta=f"+{highest_high_price_change:.2f}")
        else:
            st.metric("Highest High Price", f"${data['High'].max():,.2f}", delta=f"{highest_high_price_change:.2f}")
        # Generate sparkline untuk High
        fig_sparkline_high = px.line(x=x_sparkline_high, y=sparkline_data_high, width=150, height=50)
        fig_sparkline_high.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
                "yaxis": {"visible": False},
                "xaxis": {"visible": False},
                "showlegend": False,
                "margin": {"l":4,"r":4,"t":0, "b":0, "pad": 4}
            }
        )
        st.plotly_chart(fig_sparkline_high, use_container_width=True)
        st.markdown("<div style='text-align:center; color:green;'>HIGH KKGI.JK</div>", unsafe_allow_html=True)

    with a3:
        if highest_volume_change >= 2021:
            st.metric("Highest Volume", f"{data['Volume'].max():,.2f}", delta=f"+{highest_volume_change:.2f}")
        else:
            st.metric("Highest Volume", f"{data['Volume'].max():,.2f}", delta=f"{highest_volume_change:.2f}")
        # Generate sparkline untuk Volume
        fig_sparkline_volume = px.line(x=x_sparkline_volume, y=sparkline_data_volume, width=150, height=50)
        fig_sparkline_volume.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
                "yaxis": {"visible": False},
                "xaxis": {"visible": False},
                "showlegend": False,
                "margin": {"l":4,"r":4,"t":0, "b":0, "pad": 4}
            }
        )
        st.plotly_chart(fig_sparkline_volume, use_container_width=True)
        st.markdown("<div style='text-align:center; color:green;'>VOLUME KKGI.JK</div>", unsafe_allow_html=True)

    # Calculate Year-over-Year (YoY) Change
    data_filtered = data[data.index.year >= start_year]
    latest_close_price = data_filtered['Close'].iloc[-1]
    earliest_close_price = data_filtered['Close'].iloc[0]
    yearly_change = ((latest_close_price - earliest_close_price) / earliest_close_price) * 100

    # Row B: Financial metrics and charts
    b1, b2, b3, b4 = st.columns(4)

    # Calculate changes
    highest_close_price_change = data['Close'].max() - data['Close'].iloc[-2]
    lowest_close_price_change = data['Close'].min() - data['Close'].iloc[-2]
    average_daily_volume_change = data['Volume'].mean() - data['Volume'].iloc[-2]

    # Sparkline data untuk perubahan harga tertinggi
    sparkline_data_b1 = data['Close'].iloc[-24:]  # Mengambil 24 harga penutupan terakhir
    x_sparkline_b1 = np.arange(len(sparkline_data_b1))

    # Sparkline data untuk perubahan harga terendah
    sparkline_data_b2 = data['Close'].iloc[-24:]  # Mengambil 24 harga penutupan terakhir
    x_sparkline_b2 = np.arange(len(sparkline_data_b2))

    # Sparkline data untuk rata-rata volume harian
    sparkline_data_b3 = data['Volume'].iloc[-24:]  # Mengambil 24 volume terakhir
    x_sparkline_b3 = np.arange(len(sparkline_data_b3))

    # Metrik untuk perubahan harga tertinggi, terendah, dan rata-rata volume harian
    with b1:
        if highest_close_price_change >= 2021:
            st.metric("Highest Close Price", f"${data['Close'].max():,.2f}", delta=f"+{highest_close_price_change:.2f}")
        else:
            st.metric("Highest Close Price", f"${data['Close'].max():,.2f}", delta=f"{highest_close_price_change:.2f}")
        # Generate sparkline untuk perubahan harga tertinggi
        fig_sparkline_b1 = px.line(x=x_sparkline_b1, y=sparkline_data_b1, width=150, height=50)
        fig_sparkline_b1.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
                "yaxis": {"visible": False},
                "xaxis": {"visible": False},
                "showlegend": False,
                "margin": {"l":4,"r":4,"t":0, "b":0, "pad": 4}
            }
        )
        st.plotly_chart(fig_sparkline_b1, use_container_width=True)

    with b2:
        if lowest_close_price_change >= 2021:
            st.metric("Lowest Close Price", f"${data['Close'].min():,.2f}", delta=f"+{lowest_close_price_change:.2f}")
        else:
            st.metric("Lowest Close Price", f"${data['Close'].min():,.2f}", delta=f"{lowest_close_price_change:.2f}")
        # Generate sparkline untuk perubahan harga terendah
        fig_sparkline_b2 = px.line(x=x_sparkline_b2, y=sparkline_data_b2, width=150, height=50)
        fig_sparkline_b2.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
                "yaxis": {"visible": False},
                "xaxis": {"visible": False},
                "showlegend": False,
                "margin": {"l":4,"r":4,"t":0, "b":0, "pad": 4}
            }
        )
        st.plotly_chart(fig_sparkline_b2, use_container_width=True)

    with b3:
        if average_daily_volume_change >= 2021:
            st.metric("Average Daily Volume", f"{data['Volume'].mean():,.2f}", delta=f"+{average_daily_volume_change:.2f}")
        else:
            st.metric("Average Daily Volume", f"{data['Volume'].mean():,.2f}", delta=f"{average_daily_volume_change:.2f}")
        # Generate sparkline untuk rata-rata volume harian
        fig_sparkline_b3 = px.line(x=x_sparkline_b3, y=sparkline_data_b3, width=150, height=50)
        fig_sparkline_b3.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
                "yaxis": {"visible": False},
                "xaxis": {"visible": False},
                "showlegend": False,
                "margin": {"l":4,"r":4,"t":0, "b":0, "pad": 4}
            }
        )
        st.plotly_chart(fig_sparkline_b3, use_container_width=True)

    with b4:
        st.metric("Year-over-Year Change", f"{yearly_change:.2f}%")
        # Generate sparkline untuk perubahan Year-over-Year
        fig_sparkline_b4 = px.line(x=x, y=data['Close'].pct_change().dropna().values[-24:], width=150, height=50)
        fig_sparkline_b4.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
                "yaxis": {"visible": False},
                "xaxis": {"visible": False},
                "showlegend": False,
                "margin": {"l":4,"r":4,"t":0, "b":0, "pad": 4}
            }
        )
        st.plotly_chart(fig_sparkline_b4, use_container_width=True)

    # Row C: Main chart
    c1, c2 = st.columns((7, 3))

    with c1:
        # LightWeight Chart untuk data saham
        ohlc = data.reset_index()[['Date', 'Open', 'High', 'Low', 'Close']]
        ohlc.columns = ['time', 'open', 'high', 'low', 'close']
        ohlc['time'] = ohlc['time'].astype('int64') // 10**9  # convert to Unix time in seconds

        price_line_series = [
            {
                "time": row['time'],
                "value": row['close']
            } for _, row in ohlc.iterrows()
        ]

        line_chart_options = {
            "priceLineVisible": True,
            "priceLineSource": 0,
            "priceLineWidth": 2,
            "priceLineColor": 'rgba(255, 0, 0, 1)',
            "crossHairMarkerVisible": True,
            "crossHairMarkerRadius": 3,
            "crossHairMarkerBorderColor": '#fff',
            "crossHairMarkerBackgroundColor": '#3f51b5'
        }

        renderLightweightCharts(price_line_series, line_chart_options)

    with c2:
        # Pie chart untuk distribusi volume
        volume_data = data[['Volume']].reset_index()
        volume_data['Date'] = volume_data['Date'].astype(str)
        fig_pie = px.pie(volume_data, names='Date', values='Volume', title='Distribusi Volume Perdagangan')
        st.plotly_chart(fig_pie)

if __name__ == "__main__":
    main()
