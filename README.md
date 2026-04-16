# 📊 Dashboard Analisis Dinamika Pembangunan Manusia (IPM) Jawa Timur

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/) 

## 📝 Deskripsi Proyek
Dashboard ini merupakan platform analitik interaktif yang dirancang untuk membedah profil **Indeks Pembangunan Manusia (IPM)** di 38 Kabupaten/Kota Provinsi Jawa Timur (Data 2019-2024). 

Proyek ini tidak hanya menyajikan visualisasi data, tetapi juga mengintegrasikan algoritma **Machine Learning (K-Means Clustering)** untuk memberikan insight strategis bagi pengambil kebijakan dalam memetakan kerentanan wilayah.

## 🚀 Fitur Utama
1. **Executive Summary:** Ringkasan IKU (Indikator Kinerja Utama) secara real-time berdasarkan filter wilayah dan tahun.
2. **Indeks Konvergensi (Novelty):** Metrik khusus untuk mengukur kecepatan suatu daerah dalam mengejar ketertinggalan menuju target IPM ideal (Benchmark: 85).
3. **Dekonstruksi Komponen:** Analisis korelasi dinamis antar komponen pendidikan (HLS, RLS, APS) dan kesehatan (UHH).
4. **Economic Anomaly Detection:** Deteksi otomatis fenomena *Resource Curse* (Kutukan Sumber Daya) pada daerah dengan PDRB tinggi namun ketimpangan tinggi.
5. **AI Clustering (K-Means):** Pengelompokan wilayah secara otomatis ke dalam klaster Prioritas, Berkemban, dan Mandiri menggunakan evaluasi *Silhouette Score*.

## 📂 Struktur Data & Variabel
Data yang digunakan bersumber dari BPS Jawa Timur dengan variabel sebagai berikut:
* **IPM:** Indeks Pembangunan Manusia.
* **UHH:** Umur Harapan Hidup (Dimensi Kesehatan).
* **HLS & RLS:** Harapan Lama Sekolah & Rata-rata Lama Sekolah (Dimensi Pendidikan).
* **PDRB per Kapita:** Ukuran produktivitas ekonomi wilayah.
* **Persentase Miskin:** Tingkat kemiskinan penduduk.
* **Gini Ratio:** Ukuran ketimpangan pengeluaran.
* **TPT:** Tingkat Pengangguran Terbuka.

## 🛠️ Teknologi yang Digunakan
* **Bahasa Pemrograman:** Python 3.x
* **Framework Dashboard:** [Streamlit](https://streamlit.io/)
* **Visualisasi Data:** Plotly Express (Interactive Charts)
* **Analisis Statistik & ML:** Pandas, Scikit-Learn (StandardScaler, KMeans, Silhouette Score)
* **Styling:** Custom CSS & HTML Injection untuk UI/UX yang modern.

## ⚙️ Cara Menjalankan Secara Lokal
Jika Anda ingin mencoba project ini di komputer Anda:

1. Clone repository ini:
   ```git clone [https://github.com/username-kamu/nama-repo-kamu.git](https://github.com/username-kamu/nama-repo-kamu.git)```

2. Install library yang dibutuhkan:
    ```pip install -r requirements.txt```'

3. Jalankan aplikasi:
    ```streamlit run app.py```