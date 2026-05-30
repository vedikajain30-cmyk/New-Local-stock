import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render(df_raw, df_enc):
    st.markdown("## City & Market Targeting")
    st.markdown("*Prescriptive analysis — where to launch first and how to prioritise markets*")

    # ── City scoring ──
    city_stats = df_raw.groupby('q3_city').agg(
        Total=('respondent_id', 'count'),
        Interested=('target_interested', lambda x: (x == 'Interested').sum()),
        Avg_Spend=('q11_monthly_spend', lambda x: x.map({
            'Below 500': 1, '500-1500': 2, '1500-3000': 3,
            '3000-6000': 4, '6000-10000': 5, 'Above 10000': 6}).mean()),
        UPI_Users=('q13_payment_method', lambda x: x.str.contains('UPI|digital', case=False, na=False).sum()),
        Udhaar_Users=('q12_udhaar', lambda x: (x != 'No udhaar').sum()),
        App_Willing=('q24_app_willingness', lambda x: x.isin(['Yes definitely', 'Maybe if offers are good']).sum()),
    ).reset_index()

    city_stats['Interest_Rate'] = city_stats['Interested'] / city_stats['Total']
    city_stats['UPI_Rate'] = city_stats['UPI_Users'] / city_stats['Total']
    city_stats['App_Rate'] = city_stats['App_Willing'] / city_stats['Total']
    city_stats['Udhaar_Rate'] = city_stats['Udhaar_Users'] / city_stats['Total']

    # Priority score
    city_stats['Priority_Score'] = (
        city_stats['Interest_Rate'] * 0.35 +
        city_stats['UPI_Rate'] * 0.20 +
        city_stats['App_Rate'] * 0.20 +
        (city_stats['Avg_Spend'].fillna(3) / 6) * 0.15 +
        (1 - city_stats['Udhaar_Rate']) * 0.10
    ).round(4)
    city_stats = city_stats.sort_values('Priority_Score', ascending=False)

    # ── Priority KPIs ──
    top_city = city_stats.iloc[0]
    st.markdown("### Launch Priority Ranking")
    cols = st.columns(len(city_stats))
    for i, (_, row) in enumerate(city_stats.iterrows()):
        with cols[i]:
            rank = i + 1
            color = '#2563eb' if rank == 1 else '#16a34a' if rank == 2 else '#6b7280'
            st.markdown(f"<div style='text-align:center'><span style='font-size:22px;font-weight:700;color:{color}'>"
                        f"#{rank}</span><br><b>{row['q3_city']}</b><br>"
                        f"<small>Score: {row['Priority_Score']:.3f}</small></div>",
                        unsafe_allow_html=True)
    st.markdown("---")

    # ── Bubble chart — market opportunity ──
    fig_bubble = px.scatter(
        city_stats,
        x='Interest_Rate', y='Avg_Spend',
        size='Total', color='Priority_Score',
        text='q3_city',
        color_continuous_scale='Blues',
        labels={'Interest_Rate': 'Interest Rate', 'Avg_Spend': 'Avg Spend Band',
                'Total': 'Sample Size', 'Priority_Score': 'Priority Score'},
        title='Market Opportunity Matrix — Interest Rate vs Avg Spend (bubble = sample size)'
    )
    fig_bubble.update_traces(textposition='top center', marker=dict(sizemin=10))
    fig_bubble.update_layout(
        height=420, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    # ── City stats table ──
    st.markdown("### Detailed City Metrics")
    display_df = city_stats[[
        'q3_city', 'Total', 'Interested', 'Interest_Rate',
        'UPI_Rate', 'App_Rate', 'Udhaar_Rate', 'Priority_Score'
    ]].copy()
    display_df.columns = ['City', 'Total', 'Interested', 'Interest Rate',
                           'UPI Rate', 'App Willing', 'Udhaar Rate', 'Priority Score']
    for col in ['Interest Rate', 'UPI Rate', 'App Willing', 'Udhaar Rate']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
    display_df['Priority Score'] = display_df['Priority Score'].apply(lambda x: f"{x:.3f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Persona by city heatmap ──
    st.markdown("### Persona Composition by City")
    if 'persona_label' in df_raw.columns:
        persona_city = df_raw.groupby(['q3_city', 'persona_label']).size().unstack(fill_value=0)
        persona_city_pct = persona_city.div(persona_city.sum(axis=1), axis=0).round(3) * 100

        fig_heat = go.Figure(go.Heatmap(
            z=persona_city_pct.values,
            x=persona_city_pct.columns.tolist(),
            y=persona_city_pct.index.tolist(),
            colorscale='Blues',
            text=[[f"{v:.0f}%" for v in row] for row in persona_city_pct.values],
            texttemplate='%{text}',
            textfont=dict(size=11)
        ))
        fig_heat.update_layout(
            title='Persona % by City', height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(tickangle=-25),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Category demand by city ──
    st.markdown("### Category Demand by City")
    categories = ['Grocery', 'Agri', 'Personal', 'Puja', 'Mobile', 'Snacks', 'Medicines']
    cat_city_data = []
    for city in df_raw['q3_city'].unique():
        city_df = df_raw[df_raw['q3_city'] == city]
        for cat in categories:
            cnt = city_df['q16_categories_bought'].fillna('').str.contains(cat, case=False).sum()
            pct = cnt / len(city_df) * 100
            cat_city_data.append({'City': city, 'Category': cat, 'Penetration': round(pct, 1)})

    cat_city_df = pd.DataFrame(cat_city_data)
    fig_cat = px.bar(cat_city_df, x='City', y='Penetration', color='Category',
                     barmode='group', title='Category Purchase Penetration by City (%)',
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig_cat.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                           xaxis_tickangle=-25,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_cat, use_container_width=True)

    # ── Go-to-market strategy cards ──
    st.markdown("### Go-to-Market Strategy by Tier")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**Tier 1 — Launch Now**")
            top2 = city_stats.head(2)['q3_city'].tolist()
            st.markdown(f"Cities: **{', '.join(top2)}**")
            st.markdown("- App-based pilot with early adopter discounts")
            st.markdown("- Partner with 5–10 shops in each city")
            st.markdown("- Focus: Aspirational + Tech-Forward segments")
    with c2:
        with st.container(border=True):
            st.markdown("**Tier 2 — Expand in 3–6 months**")
            mid = city_stats.iloc[2:5]['q3_city'].tolist()
            st.markdown(f"Cities: **{', '.join(mid)}**")
            st.markdown("- WhatsApp-based outreach")
            st.markdown("- Shopkeeper partnership model")
            st.markdown("- Focus: Homemaker + Price-Sensitive segments")
    with c3:
        with st.container(border=True):
            st.markdown("**Tier 3 — Build awareness**")
            low = city_stats.iloc[5:]['q3_city'].tolist()
            st.markdown(f"Cities: **{', '.join(low)}**")
            st.markdown("- Seasonal campaign only (harvest/festival)")
            st.markdown("- Agri input bundling entry point")
            st.markdown("- Focus: Loyal Rural Buyer segment")
