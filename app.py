import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard E-Commerce", layout="wide")

# ========================
# LOAD ALL CSV
# ========================
@st.cache_data
def load_all_data(folder="data"):
    data = {}
    # Pastikan folder 'data' ada agar tidak error
    if not os.path.exists(folder):
        st.error(f"Folder '{folder}' tidak ditemukan!")
        return data
        
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            data[file] = pd.read_csv(os.path.join(folder, file))
    return data

data = load_all_data()

# ========================
# SIDEBAR
# ========================
st.sidebar.title("⚙️ Pengaturan")

if data:
    selected_file = st.sidebar.selectbox(
        "Pilih Dataset CSV",
        list(data.keys())
    )
    df = data[selected_file]
else:
    st.sidebar.error("Tidak ada file CSV di folder data.")
    st.stop()

menu = st.sidebar.radio(
    "Menu",
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
# LOGIK DASHBOARD
# ========================
if menu == "Dashboard":
    st.title("📊 Dashboard Analisis E-Commerce")

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Data", df.shape[0])
    col2.metric("Jumlah Kolom", df.shape[1])
    col3.metric("Missing Value", df.isnull().sum().sum())
    
    st.info(f"File aktif saat ini: **{selected_file}**")

# ========================
# DATASET
# ========================
elif menu == "Dataset":
    st.title("📁 Dataset")
    st.write(f"Menampilkan isi dari: **{selected_file}**")
    st.dataframe(df)

# ========================
# STATISTIK
# ========================
elif menu == "Statistik":
    st.title("📑 Statistik Deskriptif")
    st.dataframe(df.describe(include="all"))

# ========================
# VISUALISASI
# ========================
elif menu == "Visualisasi":
    st.title("📈 Visualisasi")

    num_cols = df.select_dtypes(include="number").columns

    if len(num_cols) > 0:
        col = st.selectbox("Pilih Kolom Numerik", num_cols)
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            sns.histplot(df[col], bins=30, ax=ax, kde=True)
            ax.set_title(f"Distribusi {col}")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Boxplot {col}")
            st.pyplot(fig)
    else:
        st.warning("Tidak ada kolom numerik di dataset ini.")

# ========================
# CLUSTERING (SIMPLE)
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
        sns.scatterplot(
            x=df_cluster.index,
            y="payment_value",
            hue="cluster",
            data=df_cluster,
            palette="viridis",
            ax=ax
        )
        st.pyplot(fig)
    else:
        st.warning("Kolom 'payment_value' tidak ditemukan pada dataset terpilih.")

# ========================
# DATA MINING (VALIDASI KHUSUS)
# ========================
elif menu == "Data Mining":
    st.title("🧠 Data Mining - Customer Segmentation")
    
    # Validasi File
    allowed_mining = ["orders_dataset.csv", "order_payments_dataset.csv", "customers_dataset.csv"]
    
    if selected_file in allowed_mining:
        try:
            orders = data["orders_dataset.csv"]
            payments = data["order_payments_dataset.csv"]
            customers = data["customers_dataset.csv"]

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

            st.subheader("Hasil Clustering")
            st.dataframe(customer_spending.head())

            fig, ax = plt.subplots()
            sns.scatterplot(
                data=customer_spending,
                x=customer_spending.index,
                y="total_spending",
                hue="cluster",
                palette="Set2",
                ax=ax
            )
            ax.set_title("Segmentasi Pelanggan Berdasarkan Total Spending")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat menggabungkan data: {e}")
    else:
        st.warning("⚠️ Menu Data Mining hanya dapat diakses melalui file: orders_dataset.csv, order_payments_dataset.csv, atau customers_dataset.csv")

# ========================
# GEOANALYSIS (VALIDASI KHUSUS)
# ========================
elif menu == "Geoanalysis":
    st.title("🌍 Geoanalysis - Persebaran Pelanggan")

    # Validasi File
    if selected_file == "geolocation_dataset.csv":
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
            ]].drop_duplicates()

            state_count = customer_geo["customer_state"].value_counts().reset_index()
            state_count.columns = ["State", "Total Customer"]

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots()
                sns.barplot(data=state_count, x="State", y="Total Customer", palette="viridis", ax=ax)
                ax.set_title("Jumlah Pelanggan per State")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots()
                ax.scatter(customer_geo["geolocation_lng"], customer_geo["geolocation_lat"], alpha=0.1, s=1)
                ax.set_title("Persebaran Lokasi Pelanggan")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Gagal memuat Geoanalysis: {e}")
    else:
        st.warning("⚠️ Menu Geoanalysis hanya tersedia jika Anda memilih file **geolocation_dataset.csv**.")

# ========================
# KESIMPULAN
# ========================
elif menu == "Kesimpulan":
    st.title("📌 Kesimpulan")
    st.markdown("""
    - **Data Mining**: Segmentasi pelanggan berhasil dihitung hanya jika data Orders, Payments, dan Customers tersedia.
    - **Geoanalysis**: Peta persebaran lokasi hanya ditampilkan pada dataset Geolocation untuk efisiensi performa.
    - **Dashboard**: Memungkinkan navigasi cepat antar dataset untuk eksplorasi data mentah dan statistik dasar.
    """)