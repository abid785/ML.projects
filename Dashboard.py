import Selection_courses as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st

st.set_page_config(
    page_title="📊 Dataset Dashboard",
    layout="wide"
)
st.title("Visualization Dashboard")

# --- Custom CSS (dark layout) ---
st.markdown("""
    <style>
        .card {
            background-color: #1f2b3d;
            padding: 15px;
            border-radius: 10px;
            color: white;
            text-align: center;
            font-size: 16px;
            margin-bottom: 10px;
        }
        body {
            background-color: #0f1220;
            color: white;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 8px 8px 0 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f2b3d;
        }
    </style>
""", unsafe_allow_html=True)

# === Upload File ===
st.sidebar.header("📁 Upload Your Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.title("📈 Dataset Dashboard")

    # === Basic Info Cards ===
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='card'>Rows: <strong>{df.shape[0]}</strong></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'>Columns: <strong>{df.shape[1]}</strong></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'>Missing Values: <strong>{df.isnull().sum().sum()}</strong></div>", unsafe_allow_html=True)

    # === Dataset Metadata in Expander ===
    with st.expander("📄 Dataset Info"):
        st.write("### Column Information")
        st.write(df.dtypes)
        st.write("### Null Values per Column")
        st.write(df.isnull().sum())

    # === Target Column Selection ===
    all_cols = df.columns.tolist()
    target_col = st.sidebar.selectbox("🎯 Select Target Column", all_cols)

    # === Main Content Tabs ===
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Target Analysis", "🔥 Correlations"])

    with tab1:
      st.subheader("📊 Dataset Overview")
    
    with st.expander("🔍 Dataset Preview (Top 10 Rows)"):
        st.dataframe(df.head(10), use_container_width=True, height=300)

    with st.expander("📌 Quick Statistics"):
        st.dataframe(df.describe(), use_container_width=True)

    with tab2:
      st.subheader("🎯 Target Column Analysis")
    
    if df[target_col].nunique() < 30:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🔹 Histogram**")
            fig1 = px.histogram(df, x=target_col, color=target_col)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("**🔹 Pie Chart**")
            pie_df = df[target_col].value_counts().reset_index()
            pie_df.columns = [target_col, 'count']
            fig2 = px.pie(pie_df, names=target_col, values='count', hole=0.3)
            st.plotly_chart(fig2, use_container_width=True)

        with col3:
            st.markdown("**🔹 Bar Chart**")
            bar_df = df[target_col].value_counts().reset_index()
            bar_df.columns = [target_col, 'count']
            fig3 = px.bar(bar_df, x=target_col, y='count', color=target_col, text='count')
            st.plotly_chart(fig3, use_container_width=True)

    else:
        st.subheader("Numerical Target Analysis")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🔹 Histogram**")
            fig = px.histogram(df, x=target_col)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**🔹 Box Plot**")
            fig = px.box(df, y=target_col)
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.markdown("**🔹 Violin Plot**")
            fig = px.violin(df, y=target_col, box=True, points="all")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("🔗 Feature Correlation Heatmap")
        numeric_cols = df.select_dtypes(include='number')
        if not numeric_cols.empty:
            corr = numeric_cols.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            with st.expander("📈 Top Correlations with Target"):
                if target_col in numeric_cols.columns:
                    target_corr = corr[target_col].sort_values(ascending=False)
                    st.dataframe(target_corr, use_container_width=True)
        else:
            st.info("No numeric columns found for correlation heatmap.")

else:
    st.info("📤 Upload a CSV file.")
