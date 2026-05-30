import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, os, io
from data_utils import encode_dataframe, get_feature_matrix, FEATURE_COLS, spend_band_label
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CLUSTER_NAMES_MAP = {
    0: 'Loyal Rural Buyer', 1: 'Price-Sensitive Urban',
    2: 'Aspirational Tier-2', 3: 'Homemaker Budget Manager',
    4: 'Young Tech-Forward'
}

REGRESSION_FEATURES = [
    'q1_age_group_enc', 'q4_occupation_enc', 'q5_income_range_enc',
    'q6_income_pattern_enc', 'q7_household_size_enc',
    'q9_shop_frequency_enc', 'q10_num_shops_enc', 'q12_udhaar_enc',
    'q14_distance_tolerance_enc', 'q15_online_shopping_enc',
    'q21_discount_threshold_enc', 'q22_brand_preference_enc',
    'q3_city_enc', 'q2_gender_enc', 'q8_decision_maker_enc',
    'cat_Grocery', 'cat_Agri', 'cat_Personal', 'cat_Mobile', 'cat_Snacks'
]

CLUSTER_FEATURES = [
    'q11_monthly_spend_enc', 'q9_shop_frequency_enc', 'q12_udhaar_enc',
    'q14_distance_tolerance_enc', 'q15_online_shopping_enc',
    'q5_income_range_enc', 'q6_income_pattern_enc',
    'q21_discount_threshold_enc', 'q22_brand_preference_enc',
    'q24_app_willingness_enc', 'q10_num_shops_enc',
    'cat_Grocery', 'cat_Agri', 'cat_Puja', 'cat_Mobile', 'cat_Snacks'
]


def get_action_tag(prob, spend_band):
    if prob >= 0.70 and spend_band >= 4:
        return 'High Priority — Aggressive Acquisition'
    elif prob >= 0.70 and spend_band < 4:
        return 'High Interest — Standard Offer'
    elif 0.40 <= prob < 0.70 and spend_band >= 4:
        return 'High Spend — Nurture Campaign'
    elif 0.40 <= prob < 0.70:
        return 'Mid Priority — Discount Offer'
    else:
        return 'Low Priority — Awareness Only'


def action_color(action):
    if 'Aggressive' in action:
        return '🔴'
    elif 'High Interest' in action or 'High Spend' in action:
        return '🟠'
    elif 'Mid' in action:
        return '🟡'
    else:
        return '⚪'


@st.cache_resource
def load_models(df_enc):
    """Load or train models from cached pkl files"""
    clf_data, reg_data, cluster_data = None, None, None

    # Classifier
    if os.path.exists('best_classifier.pkl'):
        with open('best_classifier.pkl', 'rb') as f:
            clf_data = pickle.load(f)
    else:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        X, available = get_feature_matrix(df_enc)
        y = df_enc['target_enc'].fillna(0).astype(int)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
        clf_data = {'model': model, 'scaler': scaler, 'features': available}

    # Regressor
    if os.path.exists('best_regressor.pkl'):
        with open('best_regressor.pkl', 'rb') as f:
            reg_data = pickle.load(f)
    else:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        avail_r = [c for c in REGRESSION_FEATURES if c in df_enc.columns]
        X_r = df_enc[avail_r].fillna(0)
        y_r = df_enc['q11_monthly_spend_enc'].fillna(3)
        X_train_r, _, y_train_r, _ = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
        model_r = RandomForestRegressor(n_estimators=150, random_state=42)
        model_r.fit(X_train_r, y_train_r)
        reg_data = {'model': model_r, 'scaler': None, 'features': avail_r}

    # Clusterer
    avail_c = [c for c in CLUSTER_FEATURES if c in df_enc.columns]
    X_c = df_enc[avail_c].fillna(0)
    scaler_c = StandardScaler()
    X_c_scaled = scaler_c.fit_transform(X_c)
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    km.fit(X_c_scaled)
    cluster_data = {'model': km, 'scaler': scaler_c, 'features': avail_c}

    return clf_data, reg_data, cluster_data


def predict_new_data(df_new_raw, clf_data, reg_data, cluster_data):
    df_enc_new = encode_dataframe(df_new_raw)

    results = df_new_raw.copy()

    # Classification
    clf_feat = [f for f in clf_data['features'] if f in df_enc_new.columns]
    if clf_feat:
        X_clf = df_enc_new[clf_feat].fillna(0)
        probs = clf_data['model'].predict_proba(X_clf)[:, 1]
        results['interested_probability'] = (probs * 100).round(1)
        results['interest_prediction'] = ['Interested' if p >= 0.5 else 'Not Interested' for p in probs]
    else:
        results['interested_probability'] = 50.0
        results['interest_prediction'] = 'Unknown'

    # Regression
    reg_feat = [f for f in reg_data['features'] if f in df_enc_new.columns]
    if reg_feat:
        X_reg = df_enc_new[reg_feat].fillna(0)
        pred_spend = reg_data['model'].predict(X_reg)
        pred_spend_clipped = np.clip(np.round(pred_spend).astype(int), 1, 6)
        results['predicted_spend_band_num'] = pred_spend_clipped
        results['predicted_spend_band'] = [spend_band_label(v) for v in pred_spend_clipped]
    else:
        results['predicted_spend_band'] = '₹1.5k–3k'
        results['predicted_spend_band_num'] = 3

    # Clustering
    clust_feat = [f for f in cluster_data['features'] if f in df_enc_new.columns]
    if clust_feat:
        X_clust = df_enc_new[clust_feat].fillna(0)
        X_clust_s = cluster_data['scaler'].transform(X_clust)
        cluster_labels = cluster_data['model'].predict(X_clust_s)
        results['cluster_label'] = cluster_labels
        results['persona_segment'] = [CLUSTER_NAMES_MAP.get(l, f'Cluster {l}') for l in cluster_labels]
    else:
        results['persona_segment'] = 'Unknown'

    # Action tag
    if 'interested_probability' in results.columns and 'predicted_spend_band_num' in results.columns:
        results['recommended_action'] = [
            get_action_tag(row['interested_probability'] / 100, row['predicted_spend_band_num'])
            for _, row in results.iterrows()
        ]
    else:
        results['recommended_action'] = 'Analyse manually'

    return results


def render(df_raw, df_enc):
    st.markdown("## New Customer Predictor")
    st.markdown("*Upload new survey data → get instant predictions for interest, spend & segment*")

    clf_data, reg_data, cluster_data = load_models(df_enc)

    st.markdown("### How to use this page")
    with st.expander("Instructions — click to expand"):
        st.markdown("""
        1. Collect new survey responses using the same 25-question survey
        2. Save as a CSV with the same column names as the training data
        3. Upload below — the system will predict for each respondent:
           - **Interest probability** (0–100%)
           - **Predicted monthly spend band**
           - **Customer segment / persona**
           - **Recommended marketing action**
        4. Download the scored CSV and share with your sales team

        **Required columns (minimum):** `q3_city`, `q4_occupation`, `q5_income_range`,
        `q6_income_pattern`, `q9_shop_frequency`, `q11_monthly_spend`, `q12_udhaar`,
        `q13_payment_method`, `q15_online_shopping`, `q16_categories_bought`,
        `q21_discount_threshold`, `q24_app_willingness`
        """)

    # ── Sample template download ──
    sample_cols = [
        'respondent_id', 'q1_age_group', 'q2_gender', 'q3_city', 'q4_occupation',
        'q5_income_range', 'q6_income_pattern', 'q7_household_size', 'q8_decision_maker',
        'q9_shop_frequency', 'q10_num_shops', 'q11_monthly_spend', 'q12_udhaar',
        'q13_payment_method', 'q14_distance_tolerance', 'q15_online_shopping',
        'q16_categories_bought', 'q17_same_trip_items', 'q18_bundle_interest',
        'q19_stockout_behaviour', 'q20_product_discovery', 'q21_discount_threshold',
        'q22_brand_preference', 'q23_frustrations', 'q24_app_willingness',
        'q25_platform_interest'
    ]
    sample_row = {col: '' for col in sample_cols}
    sample_df = pd.DataFrame([sample_row])
    sample_csv = sample_df.to_csv(index=False)
    st.download_button(
        "Download Template CSV", sample_csv,
        file_name='new_customer_survey_template.csv',
        mime='text/csv'
    )

    st.markdown("---")

    # ── Demo mode: use 50 rows from training data ──
    st.markdown("### Demo Mode — Predict on Sample from Training Data")
    demo_size = st.slider("Number of sample rows to predict", 10, 200, 50)

    if st.button("Run Demo Prediction", type="primary"):
        sample = df_raw.sample(demo_size, random_state=99).reset_index(drop=True)
        with st.spinner("Running predictions..."):
            results = predict_new_data(sample, clf_data, reg_data, cluster_data)
        _show_results(results, demo_mode=True)

    st.markdown("---")

    # ── Real upload ──
    st.markdown("### Upload Real New Survey Data")
    uploaded = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded is not None:
        try:
            df_new = pd.read_csv(uploaded)
            st.success(f"Uploaded {len(df_new)} rows, {len(df_new.columns)} columns")
            st.markdown("**Preview (first 5 rows):**")
            st.dataframe(df_new.head(), use_container_width=True)

            if st.button("Run Prediction on Uploaded Data", type="primary"):
                with st.spinner("Encoding features and running models..."):
                    results = predict_new_data(df_new, clf_data, reg_data, cluster_data)
                _show_results(results, demo_mode=False)
        except Exception as e:
            st.error(f"Error reading file: {e}")


def _show_results(results, demo_mode=False):
    st.markdown("### Prediction Results")

    # Summary metrics
    if 'interested_probability' in results.columns:
        high_priority = (results['interested_probability'] >= 70).sum()
        avg_prob = results['interested_probability'].mean()
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Scored", len(results))
        with m2:
            st.metric("Avg Interest Probability", f"{avg_prob:.1f}%")
        with m3:
            st.metric("High Priority Leads", high_priority)
        with m4:
            predicted_interested = (results['interested_probability'] >= 50).sum()
            st.metric("Predicted Interested", predicted_interested)

    # Results table
    display_cols = ['respondent_id'] if 'respondent_id' in results.columns else []
    for c in ['q3_city', 'q4_occupation', 'q11_monthly_spend',
              'interested_probability', 'interest_prediction',
              'predicted_spend_band', 'persona_segment', 'recommended_action']:
        if c in results.columns:
            display_cols.append(c)

    st.dataframe(results[display_cols].head(50), use_container_width=True, hide_index=True)

    # Visualisations
    col1, col2 = st.columns(2)
    with col1:
        if 'interested_probability' in results.columns:
            fig = go.Figure(go.Histogram(
                x=results['interested_probability'], nbinsx=20,
                marker_color='#2563eb', opacity=0.8
            ))
            fig.update_layout(title='Interest Probability Distribution',
                              xaxis_title='Probability (%)', yaxis_title='Count',
                              height=300, margin=dict(l=10, r=10, t=40, b=10),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'persona_segment' in results.columns:
            seg_counts = results['persona_segment'].value_counts()
            fig2 = px.pie(values=seg_counts.values, names=seg_counts.index,
                          title='Segment Distribution',
                          color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
            fig2.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10),
                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

    if 'recommended_action' in results.columns:
        action_counts = results['recommended_action'].value_counts()
        fig3 = px.bar(x=action_counts.values, y=action_counts.index, orientation='h',
                      color=action_counts.values, color_continuous_scale='RdYlGn',
                      title='Recommended Actions Distribution')
        fig3.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10),
                           coloraxis_showscale=False,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

    # Download
    out_csv = results.to_csv(index=False)
    st.download_button(
        "Download Scored Results CSV", out_csv,
        file_name='localstock_scored_customers.csv',
        mime='text/csv',
        type='primary'
    )

    if demo_mode:
        st.caption("Note: Demo mode uses a sample from training data. Upload fresh survey data for real predictions.")
