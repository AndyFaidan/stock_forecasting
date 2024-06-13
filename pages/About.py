import streamlit as st

# Menambahkan CSS untuk memposisikan gambar di tengah dan mengatur ukuran teks
st.markdown("""
    <style>
    .center-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }
    .custom-title {
        font-size: 24px;
        text-align: center;
    }
    .custom-text {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)
# URL gambar yang ingin ditampilkan
image_url = "https://miro.medium.com/v2/resize:fit:720/format:webp/1*ARetFl4fJOVh1VWF8GuDoA.jpeg"
st.markdown(
    f'<div style="display: flex; justify-content: center;"><img src="{image_url}" style="width: 720px;"></div>',
    unsafe_allow_html=True
)


# Menampilkan gambar dari URL
st.markdown('<h1 class="custom-title">Tentang Aplikasi Ini</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="custom-text">
## Gambaran Aplikasi
Aplikasi ini menyediakan peramalan deret waktu menggunakan jaringan Long Short-Term Memory (LSTM) khusus untuk saham KKGI.JK.

## Tujuan
Tujuan dari aplikasi ini adalah untuk membantu pengguna membuat prediksi yang akurat berdasarkan data historis saham KKGI.JK.

## Metodologi
**Algoritma**: Aplikasi ini menggunakan LSTM, sebuah jenis jaringan saraf berulang (RNN) yang sangat cocok untuk peramalan deret waktu.
**Data**: Aplikasi ini secara otomatis mengambil data historis saham KKGI.JK dari Yahoo Finance.

## Petunjuk Penggunaan
1. **Ambil Data**: Aplikasi secara otomatis mengambil data historis saham KKGI.JK dari Yahoo Finance.
2. **Setel Parameter**: Sesuaikan parameter untuk model LSTM (misalnya, jumlah epoch, ukuran batch).
3. **Jalankan Peramalan**: Klik tombol 'Peramalan' untuk menghasilkan prediksi.
4. **Lihat Hasil**: Nilai yang diprediksi dan plot yang sesuai akan ditampilkan.

## Detail Teknis
- **Model**: Model LSTM terdiri dari beberapa lapisan dengan konfigurasi tertentu.
- **Perpustakaan**: Perpustakaan utama yang digunakan termasuk  Keras, Pandas, dan yfinance.

## Informasi Pengembang
**Penulis**: Firhan Abdillah Mahbubi
**Kontak**: firhanabdillahmahbubi@gmail.com

## Penghargaan
Terima kasih khusus kepada komunitas open-source yang telah menyediakan sumber daya dan alat yang berharga.

## Disclaimer
- Prediksi yang diberikan oleh aplikasi ini didasarkan pada data historis dan asumsi model. Akurasi tidak dijamin.
- Gunakan prediksi ini dengan risiko dan kebijakan Anda sendiri.
</div>
""", unsafe_allow_html=True)

