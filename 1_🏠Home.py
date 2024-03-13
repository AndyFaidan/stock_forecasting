import streamlit as st
import pandas as pd
import folium
import yfinance as yf
from datetime import datetime
from streamlit_folium import folium_static
from folium.plugins import HeatMap

#navicon and header
st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")  

# Load data from CSV
df = pd.read_csv('lokasi.csv')

# Create a map
m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=10, width='100%')

# Add a marker for each company
for i, row in df.iterrows():
    popup_content = f"""
    <div>
        <h3>{row['Company']}</h3>
        <p><strong>Address:</strong></p>
        <p>PT Resource Alam Indonesia Tbk</p>
        <p>Bumi Raya Group Building</p>
        <p>Jl. Pembangunan I No. 3</p>
        <p>Jakarta Pusat 10130</p>
        <p>Indonesia</p>
        <p><strong>Contact:</strong></p>
        <p>Phone: 62 21 633 3036</p>
        <p>Website: <a href="https://www.raintbk.com" target="_blank">www.raintbk.com</a></p>
        <p><strong>Location:</strong></p>
        <p>Latitude: {row['Latitude']}</p>
        <p>Longitude: {row['Longitude']}</p>
    </div>
    """

    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        tooltip=row['Company'],
        icon=folium.Icon(color='red', icon='building', prefix='fa'),
    ).add_to(m).add_child(folium.Popup(popup_content, max_width=300))

def add_google_maps(m):
    tiles = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    attr = "Google Digital Satellite"
    folium.TileLayer(tiles=tiles, attr=attr, name=attr, overlay=True, control=True).add_to(m)
    
    # Add labels for streets and objects
    label_tiles = "https://mt1.google.com/vt/lyrs=h&x={x}&y={y}&z={z}"
    label_attr = "Google Labels"
    folium.TileLayer(tiles=label_tiles, attr=label_attr, name=label_attr, overlay=True, control=True).add_to(m)

    return m

# Heatmap Layer
heat_data = [[row['Latitude'], row['Longitude']] for _, row in df.iterrows()]
HeatMap(heat_data).add_to(m)

# Fullscreen Control
folium.plugins.Fullscreen(position='topright', title='Fullscreen', title_cancel='Exit Fullscreen').add_to(m)

# Display the map using Streamlit Folium
st.markdown('### Company Location')


with st.expander("OFFICE MAP VIEW & ANALYTICS"):
    m = add_google_maps(m)
    m.add_child(folium.LayerControl(collapsed=False))
    folium_static(m, width=1240, height=600)
    folium.LayerControl().add_to(m)


st.info(
    """
    PT Resource Alam Indonesia Tbk, bersama dengan anak perusahaannya, terlibat dalam penambangan batu bara dan metana di Indonesia.
    Perusahaan juga menjual laminasi tekan tinggi dan papan partikel dilaminasi melamin; dan menyediakan layanan manajemen pelabuhan, pasokan listrik, serta dukungan penambangan dan penggalian lainnya.
    Selain itu, perusahaan terlibat dalam grosir logam dan bijih logam, industri pembangkit listrik tenaga air, real estate, dan kegiatan perdagangan.
    Perusahaan sebelumnya dikenal sebagai PT Kurnia Kapuas Utama Tbk dan mengubah namanya menjadi PT Resource Alam Indonesia Tbk pada November 2003.
    PT Resource Alam Indonesia Tbk didirikan pada tahun 1981 dan berkantor pusat di Jakarta Pusat, Indonesia.
    """
)

# Date range selector for stock data in the sidebar
start_date = st.sidebar.date_input('Start Date', datetime(2020, 1, 1))
end_date = st.sidebar.date_input('End Date', datetime.today())

# Ticker symbol definition
ticker_symbol = 'KKGI.JK'

# Display stock data for KKGI.JK from Yahoo Finance
with st.expander(f'### Stock Data for {ticker_symbol} from Yahoo Finance ({start_date} to {end_date})'):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    st.line_chart(stock_data['Close'])

# Display additional statistical information
# Create columns for expanders
c1, c2, c3, c4 = st.columns(4)

with c1:
    with st.expander(f"**Mean Closing Price:** {stock_data['Close'].mean():.2f}"):
        st.progress(stock_data['Close'].mean() / stock_data['Close'].max())

with c2:
    with st.expander(f"**Standard Deviation:** {stock_data['Close'].std():.2f}"):
        st.progress(stock_data['Close'].std() / stock_data['Close'].max())

with c3:
    with st.expander(f"**Minimum Closing Price:** {stock_data['Close'].min():.2f}"):
        st.progress(stock_data['Close'].min() / stock_data['Close'].max())

with c4:
    with st.expander(f"**Maximum Closing Price:** {stock_data['Close'].max():.2f}"):
        st.progress(1.0)  # Full progress as it represents the maximum value

# Conclusion
average_close = stock_data['Close'].mean()
min_close = stock_data['Close'].min()
max_close = stock_data['Close'].max()

st.success(f"Setelah melihat data saham, dapat disimpulkan bahwa harga penutup saham {ticker_symbol} memiliki fluktuasi dalam periode {start_date} hingga {end_date}. Rata-rata harga penutup adalah {average_close:.2f}, dengan harga minimum mencapai {min_close:.2f} dan harga maksimum mencapai {max_close:.2f}")

