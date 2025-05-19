import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


st.title("Time Series Forecasting dengan Prophet")
st.caption("Memprediksi datetime series data berdasarkan target data menggunakan Prophet")

uploaded_file = "DailyDelhiClimateTrain.csv"

#Menampilkan opsi selanjutnya jika file sudah diupload
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataframe yang diupload:")
    st.write(df) #Menampilkan 5 data teratas pada dataframe 

    st.write("Pilih keadaan lingkungan yang ingin anda ketahui")
    #Select box untuk memilih variabel date dan target
    col_date = "date"
    available_targets = [col for col in df.columns if col != col_date]
    col_target = st.selectbox("Pilih kolom untuk target (y)", options=available_targets)

    if col_date == col_target:
        #Memberikan warning ketika menginputkan data yang sama karena model tidak dapat menganalisis variabel yang sama
        st.warning("Kolom tanggal dan target tidak boleh sama.")
    else:
        df_prophet = df[[col_date, col_target]].rename(columns={col_date: 'ds', col_target: 'y'}) #Rename kolom
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds']) #Convert kolom ds menjadi datetime

        st.write("Data yang siap untuk Prophet:")
        st.write(df_prophet) #Menampilkan 5 data teratas dari data yang ingin dianalisis

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

            # Menggabungkan nilai aktual dan hasil prediksi pada periode yang tersedia
            forecast_eval = forecast.set_index('ds').join(df_prophet.set_index('ds'), how='left')
            forecast_eval = forecast_eval.dropna(subset=['y'])  # Pastikan hanya membandingkan data historis (bukan masa depan)

            # Hitung metrik evaluasi
            mae = mean_absolute_error(forecast_eval['y'], forecast_eval['yhat'])
            mse = mean_squared_error(forecast_eval['y'], forecast_eval['yhat'])
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((forecast_eval['y'] - forecast_eval['yhat']) / forecast_eval['y'])) * 100

            # Tampilkan hasil evaluasi
            st.subheader("Evaluasi Model (Data Historis)")
            st.write(f"**MAE (Mean Absolute Error):** {mae:.2f}")
            st.write(f"**MSE (Mean Squared Error):** {mse:.2f}")
            st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.2f}")
            st.write(f"**MAPE (Mean Absolute Percentage Error):** {mape:.2f}%")
