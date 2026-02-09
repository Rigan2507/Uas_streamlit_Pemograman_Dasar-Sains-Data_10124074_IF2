import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard E-Commerce", layout="wide")

# ========================
# LOAD ALL CSV (ROOT FOLDER)
# ========================
@st.cache_data
def load_all_data(folder="."): # Mengubah ke "." agar membaca folder tempat app.py berada
    data = {}
    if not os.path.exists(folder):
        st.error(f"Direktori '{folder}' tidak ditemukan!")
        return data
        
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            try:
                data[file] = pd.read_csv(os.path.join(folder, file))
            except Exception as e:
                st.error(f"Gagal memuat {file}: {e}")
    return data

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
    st.sidebar.error("Tidak ada file CSV ditemukan di direktori utama.")
    st.info("Pastikan file CSV Anda sudah di-upload ke folder yang sama dengan app.py")
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
    col1.metric("Jumlah Data", df.shape[0])
    col2.metric("Jumlah Kolom", df.shape[1])
    col3.metric("Missing Value", df.isnull().sum().sum())
    
    st.info(f"File aktif saat ini: **{selected_file}**")
    st.subheader("Cuplikan Data")
    st.dataframe(df.head(10))

# ========================
# 2. DATASET
# ========================
elif menu == "Dataset":
    st.title("📁 Dataset Viewer")
    st.write(f"Menampilkan isi dari: **{selected_file}**")
    st.dataframe(df)

# ========================
# 3. STATISTIK
# ========================
elif menu == "Statistik":
    st.title("📑 Statistik Deskriptif")
    st.dataframe(df.describe(include="all"))

# ========================
# 4. VISUALISASI
# ========================
elif menu == "Visualisasi":
    st.title("📈 Visualisasi Data")

    num_cols = df.select_dtypes(include="number").columns

    if len(num_cols) > 0:
        col = st.selectbox("Pilih Kolom Numerik", num_cols)
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            sns.histplot(df[col], bins=30, ax=ax, kde=True, color="skyblue")
            ax.set_title(f"Distribusi {col}")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax, color="salmon")
            ax.set_title(f"Boxplot {col}")
            st.pyplot(fig)
    else:
        st.warning("Tidak ada kolom numerik di dataset ini.")

# ========================
# 5. CLUSTERING (SIMPLE)
# ========================
elif menu == "Clustering":
    st.title("🤖 Clustering Sederhana")

    if "payment_value" in df.columns:
        X = df[["payment_value"]].dropna()
        k = st.slider("Jumlah Cluster", 2, 6, 3)

        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        df_cluster = X.copy()
        df_cluster["cluster"] = model.fit_predict(X)

        fig, ax = plt.subplots()
        sns.scatterplot(x=df_cluster.index, y="payment_value", hue="cluster", data=df_cluster, palette="viridis", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Kolom 'payment_value' tidak ditemukan. Gunakan file order_payments_dataset.csv")

# ========================
# 6. DATA MINING
# ========================
elif menu == "Data Mining":
    st.title("🧠 Data Mining - Customer Segmentation")
    
    needed = ["orders_dataset.csv", "order_payments_dataset.csv", "customers_dataset.csv"]
    if all(f in data for f in needed):
        try:
            orders = data["orders_dataset.csv"]
            payments = data["order_payments_dataset.csv"]
            customers = data["customers_dataset.csv"]

            df_merge = orders.merge(payments, on="order_id").merge(customers, on="customer_id")
            
            customer_spending = (
                df_merge.groupby("customer_id")["payment_value"]
                .sum().reset_index().rename(columns={"payment_value": "total_spending"})
            )

            k = st.slider("Jumlah Cluster", 2, 6, 3)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            customer_spending["cluster"] = kmeans.fit_predict(customer_spending[["total_spending"]])

            st.write("Hasil Segmentasi:")
            st.dataframe(customer_spending.head())

            fig, ax = plt.subplots()
            sns.scatterplot(data=customer_spending, x=customer_spending.index, y="total_spending", hue="cluster", palette="Set2", ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error pada Data Mining: {e}")
    else:
        st.warning(f"Membutuhkan file: {', '.join(needed)}")

# ========================
# 7. GEOANALYSIS (FIXED)
# ========================
elif menu == "Geoanalysis":
    st.title("🌍 Geoanalysis - Persebaran Pelanggan")

    # Nama file harus persis sama dengan yang di-upload ke GitHub
    file_cust = "customers_dataset.csv"
    file_geo = "geolocation_dataset.csv"

    if file_cust in data and file_geo in data:
        try:
            customers = data[file_cust]
            geo = data[file_geo]

            # Merge data berdasarkan kode pos
            customer_geo = customers.merge(
                geo,
                left_on="customer_zip_code_prefix",
                right_on="geolocation_zip_code_prefix"
            )

            # Optimasi agar tidak berat: ambil 10.000 data unik saja
            customer_geo = customer_geo[[
                "customer_state",
                "geolocation_lat",
                "geolocation_lng"
            ]].drop_duplicates().head(10000)

            col1, col2 = st.columns(2)

            with col1:
                state_count = customer_geo["customer_state"].value_counts().reset_index()
                state_count.columns = ["State", "Total Customer"]
                fig, ax = plt.subplots()
                sns.barplot(data=state_count, x="State", y="Total Customer", palette="viridis", ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots()
                ax.scatter(customer_geo["geolocation_lng"], customer_geo["geolocation_lat"], alpha=0.3, s=1, color="blue")
                ax.set_title("Peta Koordinat Pelanggan")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Gagal memproses Geoanalysis: {e}")
    else:
        # Pesan jika file tidak ditemukan di folder root
        st.error(f"File '{file_cust}' atau '{file_geo}' tidak ditemukan di folder utama.")
        st.info("Pastikan Anda sudah mengunggah kedua file tersebut ke GitHub di folder yang sama dengan app.py.")

# ========================
# 8. KESIMPULAN
# ========================
elif menu == "Kesimpulan":
    st.title("📌 Kesimpulan")
    st.markdown("""
    - **Akses Data**: File sekarang dibaca langsung dari root directory.
    - **Geoanalysis**: Memerlukan file `customers_dataset.csv` dan `geolocation_dataset.csv`.
    - **Optimasi**: Visualisasi geografis dibatasi 10.000 titik agar aplikasi tetap ringan.
    """)