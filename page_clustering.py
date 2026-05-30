import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from data_utils import FEATURE_COLS, FEATURE_LABELS, get_feature_matrix

CLUSTER_FEATURES = [
    'q11_monthly_spend_enc', 'q9_shop_frequency_enc', 'q12_udhaar_enc',
    'q14_distance_tolerance_enc', 'q15_online_shopping_enc',
    'q5_income_range_enc', 'q6_income_pattern_enc',
    'q21_discount_threshold_enc', 'q22_brand_preference_enc',
    'q24_app_willingness_enc', 'q10_num_shops_enc',
    'cat_Grocery', 'cat_Agri', 'cat_Puja', 'cat_Mobile', 'cat_Snacks'
]

CLUSTER_NAMES = {
    0: 'Loyal Rural Buyer', 1: 'Price-Sensitive Urban',
    2: 'Aspirational Tier-2', 3: 'Homemaker Budget Manager',
    4: 'Young Tech-Forward'
}

CLUSTER_COLORS = ['#2563eb', '#16a34a', '#d97706', '#dc2626', '#7c3aed']

CLUSTER_STRATEGY = {
    'Loyal Rural Buyer': {
        'icon': '🌾', 'color': '#16a34a',
        'discount': 'Only switch at 30%+ off — do NOT over-discount',
        'channel': 'Shopkeeper-mediated / in-person',
        'products': 'Agri inputs, Grocery staples, Puja items',
        'message': 'Trust and reliability over price. Approach through existing shopkeeper relationships.',
        'ltv': 'Medium — seasonal spikes post-harvest'
    },
    'Price-Sensitive Urban': {
        'icon': '🏙️', 'color': '#2563eb',
        'discount': '10–20% off triggers switching behaviour',
        'channel': 'WhatsApp + in-store notices',
        'products': 'Grocery, Snacks, Personal care, Cleaning',
        'message': 'Lead with value bundles and savings. Compare to online prices.',
        'ltv': 'Medium-High — steady monthly spend'
    },
    'Aspirational Tier-2': {
        'icon': '📈', 'color': '#d97706',
        'discount': 'Any saving triggers action — very deal-sensitive',
        'channel': 'App notifications + WhatsApp',
        'products': 'Branded personal care, Snacks, Mobile accessories',
        'message': 'Brand association matters. Position platform as premium service.',
        'ltv': 'High — willing to spend on quality'
    },
    'Homemaker Budget Manager': {
        'icon': '🏠', 'color': '#dc2626',
        'discount': '5–10% off is enough — very budget conscious',
        'channel': 'WhatsApp groups + word of mouth',
        'products': 'Grocery, Cleaning, Personal care, Puja, Stationery',
        'message': 'Bundle value is key. Monthly savings calculator resonates.',
        'ltv': 'Medium — daily shopper, volume driven'
    },
    'Young Tech-Forward': {
        'icon': '📱', 'color': '#7c3aed',
        'discount': 'Any saving — but convenience matters more than price',
        'channel': 'App-first, WhatsApp, Social media',
        'products': 'Mobile accessories, Snacks, Beverages, Ready-to-eat',
        'message': 'Convenience and speed. Early adopter offers. Digital-first experience.',
        'ltv': 'High potential — growing income, high loyalty if app experience is good'
    }
}


@st.cache_data
def run_clustering(df_enc, n_clusters=5):
    available = [c for c in CLUSTER_FEATURES if c in df_enc.columns]
    X = df_enc[available].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return labels, sil, X_pca, scaler, km, available


@st.cache_data
def compute_elbow(df_enc):
    available = [c for c in CLUSTER_FEATURES if c in df_enc.columns]
    X = df_enc[available].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    inertias, silhouettes = [], []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, lbl))
    return inertias, silhouettes


def render(df_raw, df_enc):
    st.markdown("## Customer Segmentation")
    st.markdown("*K-Means clustering — discovering natural customer groups for targeted marketing*")

    n_clusters = st.slider("Number of clusters (K)", 3, 7, 5, 1,
                           help="Use elbow curve below to guide your choice")

    labels, sil_score, X_pca, scaler, km_model, feat_used = run_clustering(df_enc, n_clusters)
    df_plot = df_enc.copy()
    df_plot['Cluster'] = labels
    df_plot['PCA1'] = X_pca[:, 0]
    df_plot['PCA2'] = X_pca[:, 1]

    # Map cluster to persona name based on ground truth overlap
    if 'persona_label' in df_raw.columns:
        df_plot['persona_label'] = df_raw['persona_label'].values
        cluster_persona = {}
        for c in range(n_clusters):
            mask = df_plot['Cluster'] == c
            if mask.sum() > 0 and 'persona_label' in df_plot.columns:
                most_common = df_plot.loc[mask, 'persona_label'].mode()
                cluster_persona[c] = most_common[0] if len(most_common) > 0 else f'Cluster {c}'
            else:
                cluster_persona[c] = f'Cluster {c}'
        df_plot['Cluster Name'] = df_plot['Cluster'].map(cluster_persona)
    else:
        df_plot['Cluster Name'] = df_plot['Cluster'].map(
            {i: f'Cluster {i}' for i in range(n_clusters)})

    # ── Metrics row ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Silhouette Score", f"{sil_score:.3f}",
                  "Higher = better separation (max 1.0)")
    with c2:
        sizes = pd.Series(labels).value_counts()
        st.metric("Largest Cluster", f"{sizes.max()} respondents")
    with c3:
        st.metric("Smallest Cluster", f"{sizes.min()} respondents")

    st.markdown("---")

    # ── PCA scatter plot ──
    col1, col2 = st.columns([2, 1])

    with col1:
        fig_scatter = px.scatter(
            df_plot, x='PCA1', y='PCA2', color='Cluster Name',
            title='Customer Clusters — PCA Projection (2D)',
            opacity=0.6, height=420,
            color_discrete_sequence=CLUSTER_COLORS[:n_clusters]
        )
        fig_scatter.update_traces(marker=dict(size=5))
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        cluster_sizes = df_plot.groupby('Cluster Name').size().reset_index(name='Count')
        fig_pie = px.pie(cluster_sizes, names='Cluster Name', values='Count',
                         title='Cluster Size Distribution',
                         color_discrete_sequence=CLUSTER_COLORS[:n_clusters],
                         hole=0.4)
        fig_pie.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Elbow + Silhouette ──
    st.markdown("### Choosing the Right K")
    inertias, silhouettes = compute_elbow(df_enc)
    col3, col4 = st.columns(2)

    with col3:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(range(2, 9)), y=inertias, mode='lines+markers',
                                        marker=dict(size=8, color='#2563eb'),
                                        line=dict(color='#2563eb', width=2), name='Inertia'))
        fig_elbow.update_layout(title='Elbow Curve', xaxis_title='K',
                                 yaxis_title='Inertia', height=280,
                                 margin=dict(l=10, r=10, t=40, b=10),
                                 plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col4:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(x=list(range(2, 9)), y=silhouettes, mode='lines+markers',
                                      marker=dict(size=8, color='#16a34a'),
                                      line=dict(color='#16a34a', width=2), name='Silhouette'))
        fig_sil.update_layout(title='Silhouette Score by K', xaxis_title='K',
                               yaxis_title='Silhouette Score', height=280,
                               margin=dict(l=10, r=10, t=40, b=10),
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sil, use_container_width=True)

    # ── Radar chart per cluster ──
    st.markdown("### Cluster Profiles — Radar View")
    radar_features = ['q11_monthly_spend_enc', 'q9_shop_frequency_enc', 'q12_udhaar_enc',
                      'q15_online_shopping_enc', 'q21_discount_threshold_enc',
                      'q24_app_willingness_enc', 'q5_income_range_enc']
    radar_labels = ['Monthly Spend', 'Shop Frequency', 'Udhaar Level',
                    'Online Usage', 'Discount Sensitivity', 'App Willingness', 'Income']

    fig_radar = go.Figure()
    cluster_names_list = sorted(df_plot['Cluster Name'].unique())
    for i, cname in enumerate(cluster_names_list):
        mask = df_plot['Cluster Name'] == cname
        vals = []
        for f in radar_features:
            if f in df_plot.columns:
                col_max = df_plot[f].max()
                col_min = df_plot[f].min()
                mean_val = df_plot.loc[mask, f].mean()
                norm = (mean_val - col_min) / (col_max - col_min + 1e-9)
                vals.append(round(norm, 3))
            else:
                vals.append(0)
        vals_closed = vals + [vals[0]]
        labels_closed = radar_labels + [radar_labels[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed, theta=labels_closed, fill='toself',
            name=cname, opacity=0.65,
            line=dict(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], width=2)
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=420, margin=dict(l=30, r=30, t=30, b=30),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', y=-0.1)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Strategy Cards ──
    st.markdown("### Marketing Strategy by Segment")
    unique_clusters = sorted(df_plot['Cluster Name'].unique())
    cols = st.columns(min(len(unique_clusters), 3))
    for i, cname in enumerate(unique_clusters[:5]):
        strategy = CLUSTER_STRATEGY.get(cname, {
            'icon': '👥', 'color': '#6b7280',
            'discount': 'Analyse further',
            'channel': 'Mixed', 'products': 'General',
            'message': 'Define strategy based on cluster profile.',
            'ltv': 'To be determined'
        })
        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(f"**{strategy['icon']} {cname}**")
                st.markdown(f"🎯 **Discount trigger:** {strategy['discount']}")
                st.markdown(f"📣 **Channel:** {strategy['channel']}")
                st.markdown(f"🛒 **Top products:** {strategy['products']}")
                st.markdown(f"💬 **Message:** *{strategy['message']}*")
                st.markdown(f"💰 **LTV:** {strategy['ltv']}")
                sz = (df_plot['Cluster Name'] == cname).sum()
                st.caption(f"Cluster size: {sz} respondents ({sz/len(df_plot)*100:.1f}%)")
