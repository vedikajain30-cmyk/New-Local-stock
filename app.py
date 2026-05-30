import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="LocalStock Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 14px;
        padding: 6px 0;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    h2 { font-size: 1.5rem; font-weight: 600; }
    h3 { font-size: 1.15rem; font-weight: 600; }
    .stMetric label { font-size: 12px; color: #64748b; }
    .stMetric [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stHorizontalBlock"] > div { min-width: 0; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    try:
        df_raw = pd.read_csv("localstock_survey_raw.csv")
        df_enc = pd.read_csv("localstock_survey_encoded.csv")
    except FileNotFoundError:
        st.error("Data files not found. Ensure localstock_survey_raw.csv and localstock_survey_encoded.csv are in the same directory.")
        st.stop()
    # Ensure target column exists in encoded df
    if 'target_enc' not in df_enc.columns:
        df_enc['target_enc'] = (df_raw['target_interested'] == 'Interested').astype(int)
    return df_raw, df_enc


def main():
    df_raw, df_enc = load_data()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 🛒 LocalStock")
        st.markdown("**Intelligence Dashboard**")
        st.markdown("*Data-Driven Decision Engine*")
        st.markdown("---")

        pages = {
            "📊 Executive Overview": "overview",
            "👥 Customer Segmentation": "clustering",
            "🔗 Product Associations": "association",
            "🎯 Conversion Predictor": "classification",
            "💰 Spending Power Model": "regression",
            "🗺️ City & Market Targeting": "city",
            "🔮 New Customer Predictor": "upload"
        }

        selected_label = st.radio("Navigation", list(pages.keys()))
        selected_page = pages[selected_label]

        st.markdown("---")
        st.markdown("**Dataset Summary**")
        st.markdown(f"- Respondents: **{len(df_raw):,}**")
        st.markdown(f"- Features: **25**")
        st.markdown(f"- Cities: **{df_raw['q3_city'].nunique()}**")
        interested_pct = (df_raw['target_interested'] == 'Interested').mean() * 100
        st.markdown(f"- Interest rate: **{interested_pct:.1f}%**")
        st.markdown("---")
        st.caption("LocalStock v1.0 · Built with Streamlit")

    # ── Page routing ──
    if selected_page == "overview":
        import page_overview
        page_overview.render(df_raw, df_enc)

    elif selected_page == "clustering":
        import page_clustering
        page_clustering.render(df_raw, df_enc)

    elif selected_page == "association":
        import page_association
        page_association.render(df_raw, df_enc)

    elif selected_page == "classification":
        import page_classification
        page_classification.render(df_raw, df_enc)

    elif selected_page == "regression":
        import page_regression
        page_regression.render(df_raw, df_enc)

    elif selected_page == "city":
        import page_city_targeting
        page_city_targeting.render(df_raw, df_enc)

    elif selected_page == "upload":
        import page_upload_predict
        page_upload_predict.render(df_raw, df_enc)


if __name__ == "__main__":
    main()
