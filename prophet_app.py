import streamlit as st
import pandas as pd
from prophet import Prophet


st.title("Time Series Forecasting dengan Prophet")
st.caption("Memprediksi datetime series data berdasarkan target data menggunakan Prophet")

uploaded_file = st.file_uploader("Upload file CSV", type="csv")

#Menampilkan opsi selanjutnya jika file sudah diupload
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataframe yang diupload:")
    st.write(df.head()) #Menampilkan 5 data teratas pada dataframe 

    st.write("Pilih kolom untuk 'ds' (tanggal/waktu) dan 'y' (target/variable untuk forecasting)")
    #Select box untuk memilih variabel date dan target
    col_date = st.selectbox("Pilih kolom untuk tanggal (ds)", options=df.columns)
    col_target = st.selectbox("Pilih kolom untuk target (y)", options=df.columns)

    if col_date == col_target:
        #Memberikan warning ketika menginputkan data yang sama karena model tidak dapat menganalisis variabel yang sama
        st.warning("Kolom tanggal dan target tidak boleh sama.")
    else:
        df_prophet = df[[col_date, col_target]].rename(columns={col_date: 'ds', col_target: 'y'}) #Rename kolom
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds']) #Convert kolom ds menjadi datetime

        st.write("Data yang siap untuk Prophet:")
        st.write(df_prophet.head()) #Menampilkan 5 data teratas dari data yang ingin dianalisis

        period = st.slider("Pilih berapa hari untuk forecasting ke depan:", min_value=1, max_value=365) # Slider untuk menentukan jumlah hari forecasting

        if st.button("Jalankan Forecasting"):
            model = Prophet()
            model.fit(df_prophet) #Memasukkan data pada model
            future = model.make_future_dataframe(periods=period)
            forecast = model.predict(future) #Menentukan Predection Model berdasarkan future

            #Menampilkan hasil forecasting
            st.write("Hasil forecasting:")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(period))

            #Visualisasi hasil forecasting
            st.subheader('Grafik Forecasting')
            fig1 = model.plot(forecast)
            st.pyplot(fig1)
            
            #Visualisasi grafik forecasting components
            st.subheader('Grafik Forecasting Components')
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)
