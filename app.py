import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard E-Commerce", layout="wide")

# ========================
# LOAD ALL CSV (DARI ROOT / FOLDER SAMA)
# ========================
@st.cache_data
def load_all_data(folder="."):
    """Memuat semua file CSV yang ada di folder yang sama dengan app.py"""
    data = {}
    if not os.path.exists(folder):
        return data
        
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            try:
                # Membaca file CSV
                data[file] = pd.read_csv(os.path.join(folder, file))
            except Exception as e:
                st.error(f"Gagal membaca {file}: {e}")
    return data

# Mengambil data dari direktori saat ini
data = load_all_data()

# ========================
# SIDEBAR
# ========================
st.sidebar.title("⚙️ Pengaturan")

if data:
    selected_file = st.sidebar.selectbox(
        "Pilih Dataset untuk Eksplorasi",
        list(data.keys())
    )
    df = data[selected_file]
else:
    st.sidebar.error("Tidak ada file CSV ditemukan!")
    st.info("Letakkan file CSV Anda (misal: customers_dataset.csv) di folder yang sama dengan app.py")
    st.stop()

menu = st.sidebar.radio(
    "Menu Utama",
    [
        "Dashboard",
        "Dataset",
        "Statistik",
        "Visualisasi",
        "Clustering",
        "Data Mining",
        "Geoanalysis",
        "Kesimpulan"
    ]
)

# ========================
# 1. DASHBOARD
# ========================
if menu == "Dashboard":
    st.title("📊 Dashboard Analisis E-Commerce")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Baris", df.shape[0])
    col2.metric("Jumlah Kolom", df.shape[1])
    col3.metric("Total Missing Value", df.isnull().sum().sum())
    
    st.info(f"File aktif: **{selected_file}**")
    st.subheader("Cuplikan Data (10 Baris Pertama)")
    st.write(df.head(10))

# ========================
# 2. DATASET
# ========================
elif menu == "Dataset":
    st.title("📁 Dataset Viewer")
    st.write(f"Menampilkan data dari: **{selected_file}**")
    st.dataframe(df)

# ========================
# 3. STATISTIK
# ========================
elif menu == "Statistik":
    st.title("📑 Statistik Deskriptif")
    st.write("Analisis statistik otomatis untuk semua kolom:")
    st.dataframe(df.describe(include="all"))

# ========================
# 4. VISUALISASI
# ========================
elif menu == "Visualisasi":
    st.title("📈 Visualisasi Data Numerik")

    num_cols = df.select_dtypes(include="number").columns

    if len(num_cols) > 0:
        col = st.selectbox("Pilih Kolom", num_cols)
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            sns.histplot(df[col], bins=30, ax=ax, kde=True, color="#3498db")
            ax.set_title(f"Distribusi {col}")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax, color="#e74c3c")
            ax.set_title(f"Boxplot {col}")
            st.pyplot(fig)
    else:
        st.warning("Tidak ada data numerik di file ini untuk divisualisasikan.")

# ========================
# 5. CLUSTERING (SIMPLE)
# ========================
elif menu == "Clustering":
    st.title("🤖 Clustering Sederhana")

    if "payment_value" in df.columns:
        X = df[["payment_value"]].dropna()
        k = st.slider("Jumlah Cluster (K)", 2, 6, 3)

        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        df_cluster = X.copy()
        df_cluster["cluster"] = model.fit_predict(X)

        fig, ax = plt.subplots()
        sns.scatterplot(x=df_cluster.index, y="payment_value", hue="cluster", data=df_cluster, palette="Set1", ax=ax)
        ax.set_title(f"Hasil KMeans (K={k})")
        st.pyplot(fig)
    else:
        st.warning("Menu ini memerlukan kolom 'payment_value'. Silakan pilih file payments.")

# ========================
# 6. DATA MINING
# ========================
elif menu == "Data Mining":
    st.title("🧠 Data Mining - Segmentasi Pelanggan")
    
    needed = ["orders_dataset.csv", "order_payments_dataset.csv", "customers_dataset.csv"]
    if all(f in data for f in needed):
        try:
            # Menggabungkan data secara otomatis
            df_merge = data["orders_dataset.csv"].merge(data["order_payments_dataset.csv"], on="order_id")
            df_merge = df_merge.merge(data["customers_dataset.csv"], on="customer_id")

            customer_spending = (
                df_merge.groupby("customer_unique_id")["payment_value"]
                .sum()
                .reset_index()
                .rename(columns={"payment_value": "total_spending"})
            )

            k = st.slider("Jumlah Cluster", 2, 6, 3)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            customer_spending["cluster"] = kmeans.fit_predict(customer_spending[["total_spending"]])

            st.write("Hasil Segmentasi Pelanggan (Grup):")
            st.dataframe(customer_spending.head())

            fig, ax = plt.subplots()
            sns.scatterplot(data=customer_spending, x=customer_spending.index, y="total_spending", hue="cluster", palette="viridis", ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error saat penggabungan data mining: {e}")
    else:
        st.warning(f"File yang dibutuhkan tidak lengkap: {needed}")

# ========================
# 7. GEOANALYSIS (PERBAIKAN)
# ========================
elif menu == "Geoanalysis":
    st.title("🌍 Geoanalysis - Lokasi Pelanggan")

    f1 = "customers_dataset.csv"
    f2 = "geolocation_dataset.csv"

    if f1 in data and f2 in data:
        try:
            cust = data[f1]
            geo = data[f2]

            # Melakukan Merge berdasarkan Zip Code
            customer_geo = cust.merge(
                geo,
                left_on="customer_zip_code_prefix",
                right_on="geolocation_zip_code_prefix"
            )

            # Optimasi: Ambil kolom utama dan hapus duplikat koordinat agar ringan
            customer_geo = customer_geo[[
                "customer_state",
                "geolocation_lat",
                "geolocation_lng"
            ]].drop_duplicates().head(10000) # Ambil 10rb titik agar cepat

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Customer per State")
                state_count = customer_geo["customer_state"].value_counts().reset_index()
                state_count.columns = ["State", "Total"]
                fig, ax = plt.subplots()
                sns.barplot(data=state_count, x="State", y="Total", palette="rocket", ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

            with col2:
                st.subheader("Sebaran Koordinat (Lat/Lng)")
                fig, ax = plt.subplots()
                ax.scatter(customer_geo["geolocation_lng"], customer_geo["geolocation_lat"], alpha=0.3, s=1, color="purple")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            st.info("Pastikan kolom 'customer_zip_code_prefix' dan 'geolocation_zip_code_prefix' tersedia.")
    else:
        st.warning(f"Membutuhkan file '{f1}' dan '{f2}' di dalam folder.")

# ========================
# 8. KESIMPULAN
# ========================
elif menu == "Kesimpulan":
    st.title("📌 Kesimpulan")
    st.markdown("""
    1. **Sistem File**: Aplikasi sekarang otomatis mendeteksi file CSV di folder root.
    2. **Geoanalysis**: Peta koordinat dapat ditampilkan jika data lokasi tersedia (dibatasi 10rb baris untuk performa).
    3. **Insight**: Anda dapat melakukan segmentasi pelanggan (Data Mining) dan eksplorasi data mentah dalam satu tempat.
    """)