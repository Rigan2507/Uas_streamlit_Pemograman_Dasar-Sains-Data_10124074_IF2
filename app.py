import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard E-Commerce", layout="wide")

# ========================
# LOAD ALL CSV (DARI ROOT)
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
                data[file] = pd.read_csv(os.path.join(folder, file))
            except Exception as e:
                st.error(f"Gagal membaca {file}: {e}")
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
    st.sidebar.error("Tidak ada file CSV di folder ini!")
    st.info("Silakan letakkan file .csv Anda di folder yang sama dengan file app.py ini.")
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
    
    st.info(f"File yang sedang dianalisis: **{selected_file}**")
    
    st.subheader("Cuplikan Data")
    st.write(df.head(10))

# ========================
# 2. DATASET
# ========================
elif menu == "Dataset":
    st.title("📁 Dataset")
    st.write(f"Menampilkan seluruh data dari: **{selected_file}**")
    st.dataframe(df)

# ========================
# 3. STATISTIK
# ========================
elif menu == "Statistik":
    st.title("📑 Statistik Deskriptif")
    st.write("Ringkasan statistik untuk kolom numerik dan kategorikal:")
    st.dataframe(df.describe(include="all"))

# ========================
# 4. VISUALISASI
# ========================
elif menu == "Visualisasi":
    st.title("📈 Visualisasi Data")

    num_cols = df.select_dtypes(include="number").columns

    if len(num_cols) > 0:
        col = st.selectbox("Pilih Kolom untuk Visualisasi", num_cols)
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
        st.warning("Dataset ini tidak memiliki kolom numerik untuk divisualisasikan.")

# ========================
# 5. CLUSTERING (SIMPLE)
# ========================
elif menu == "Clustering":
    st.title("🤖 Clustering Sederhana")
    st.write("Clustering berbasis kolom 'payment_value' (jika tersedia).")

    if "payment_value" in df.columns:
        X = df[["payment_value"]].dropna()
        k = st.slider("Tentukan Jumlah Cluster (K)", 2, 6, 3)

        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        df_cluster = X.copy()
        df_cluster["cluster"] = model.fit_predict(X)

        fig, ax = plt.subplots()
        sns.scatterplot(
            x=df_cluster.index,
            y="payment_value",
            hue="cluster",
            data=df_cluster,
            palette="viridis",
            ax=ax
        )
        ax.set_title(f"Hasil KMeans Clustering (K={k})")
        st.pyplot(fig)
    else:
        st.warning("Kolom 'payment_value' tidak ditemukan. Coba pilih dataset payments.")

# ========================
# 6. DATA MINING
# ========================
elif menu == "Data Mining":
    st.title("🧠 Data Mining - Customer Segmentation")
    
    # Syarat file harus ada di folder utama
    needed = ["orders_dataset.csv", "order_payments_dataset.csv", "customers_dataset.csv"]
    check = all(item in data for item in needed)
    
    if check:
        try:
            orders = data["orders_dataset.csv"]
            payments = data["order_payments_dataset.csv"]
            customers = data["customers_dataset.csv"]

            # Proses Join
            df_merge = orders.merge(payments, on="order_id")
            df_merge = df_merge.merge(customers, on="customer_id")

            customer_spending = (
                df_merge.groupby("customer_id")["payment_value"]
                .sum()
                .reset_index()
                .rename(columns={"payment_value": "total_spending"})
            )

            k = st.slider("Jumlah Cluster", 2, 6, 3)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            customer_spending["cluster"] = kmeans.fit_predict(customer_spending[["total_spending"]])

            st.subheader("Hasil Segmentasi Pelanggan")
            st.write(customer_spending.head())

            fig, ax = plt.subplots()
            sns.scatterplot(
                data=customer_spending,
                x=customer_spending.index,
                y="total_spending",
                hue="cluster",
                palette="Set2",
                ax=ax
            )
            ax.set_title("Distribusi Spending per Cluster")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Gagal memproses data mining: {e}")
    else:
        st.warning(f"Untuk menu ini, pastikan file berikut ada di folder: {', '.join(needed)}")

# ========================
# 7. GEOANALYSIS
# ========================
elif menu == "Geoanalysis":
    st.title("🌍 Geoanalysis - Lokasi Pelanggan")

    needed_geo = ["customers_dataset.csv", "geolocation_dataset.csv"]
    check_geo = all(item in data for item in needed_geo)

    if check_geo:
        try:
            customers = data["customers_dataset.csv"]
            geo = data["geolocation_dataset.csv"]

            customer_geo = customers.merge(
                geo,
                left_on="customer_zip_code_prefix",
                right_on="geolocation_zip_code_prefix"
            )

            customer_geo = customer_geo[[
                "customer_state",
                "geolocation_lat",
                "geolocation_lng"
            ]].drop_duplicates().head(5000) # Limit data agar ringan

            state_count = customer_geo["customer_state"].value_counts().reset_index()
            state_count.columns = ["State", "Total Customer"]

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots()
                sns.barplot(data=state_count, x="State", y="Total Customer", palette="magma", ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots()
                ax.scatter(customer_geo["geolocation_lng"], customer_geo["geolocation_lat"], alpha=0.2, s=2, color="green")
                ax.set_title("Peta Titik Koordinat")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Gagal memproses analisis geografi: {e}")
    else:
        st.warning("Menu ini membutuhkan file 'customers_dataset.csv' dan 'geolocation_dataset.csv'.")

# ========================
# 8. KESIMPULAN
# ========================
elif menu == "Kesimpulan":
    st.title("📌 Kesimpulan & Insight")
    st.markdown("""
    ### Ringkasan Analisis:
    1.  **Aksesibilitas**: Kode telah dimodifikasi untuk membaca dataset langsung di folder utama (root) tanpa sub-folder `data/`.
    2.  **Segmentasi**: Melalui menu Data Mining, kita bisa melihat kelompok pelanggan berdasarkan daya beli (Spending).
    3.  **Geografis**: Kita dapat memetakan wilayah mana yang paling banyak memiliki basis pelanggan.
    4.  **Statistik**: Dashboard menyediakan pandangan cepat mengenai kualitas data (seperti missing values).
    
    ---
    *Saran: Pastikan nama file CSV sesuai dengan standar dataset (misal: `customers_dataset.csv`) agar fitur otomatis berjalan lancar.*
    """)