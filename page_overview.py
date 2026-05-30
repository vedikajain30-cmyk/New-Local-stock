import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render(df_raw, df_enc):
    st.markdown("## Executive Overview")
    st.markdown("*Descriptive analysis — understanding who your potential customers are*")

    # ── KPI Cards ──
    total = len(df_raw)
    interested = (df_raw['target_interested'] == 'Interested').sum()
    int_pct = interested / total * 100
    avg_spend_enc = df_enc['q11_monthly_spend_enc'].mean()
    spend_labels = {1: '<500', 2: '500-1.5k', 3: '1.5k-3k', 4: '3k-6k', 5: '6k-10k', 6: '>10k'}
    upi_users = (df_raw['q13_payment_method'].str.contains('UPI|digital', case=False, na=False)).sum()
    upi_pct = upi_users / total * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Respondents", f"{total:,}")
    with c2:
        st.metric("Interested in Platform", f"{interested:,}", f"{int_pct:.1f}%")
    with c3:
        st.metric("Avg Spend Band", spend_labels.get(int(round(avg_spend_enc)), "—"))
    with c4:
        st.metric("UPI / Digital Users", f"{upi_users:,}", f"{upi_pct:.1f}%")
    with c5:
        farmers = (df_raw['q4_occupation'] == 'Farmer/Agriculture').sum()
        st.metric("Farmer Respondents", f"{farmers:,}", f"{farmers/total*100:.1f}%")

    st.markdown("---")

    # ── Row 1: City distribution + Target split ──
    col1, col2 = st.columns([1.6, 1])

    with col1:
        city_counts = df_raw['q3_city'].value_counts().reset_index()
        city_counts.columns = ['City', 'Count']
        city_int = df_raw[df_raw['target_interested'] == 'Interested']['q3_city'].value_counts().reset_index()
        city_int.columns = ['City', 'Interested']
        city_merged = city_counts.merge(city_int, on='City', how='left').fillna(0)
        city_merged['Not Interested'] = city_merged['Count'] - city_merged['Interested']
        city_merged['Interest Rate'] = (city_merged['Interested'] / city_merged['Count'] * 100).round(1)
        city_merged = city_merged.sort_values('Count', ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(y=city_merged['City'], x=city_merged['Interested'],
                             name='Interested', orientation='h',
                             marker_color='#2563eb'))
        fig.add_trace(go.Bar(y=city_merged['City'], x=city_merged['Not Interested'],
                             name='Not Interested', orientation='h',
                             marker_color='#e2e8f0'))
        fig.update_layout(barmode='stack', title='Respondents by City — Interest Split',
                          height=340, margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation='h', y=-0.15),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        target_counts = df_raw['target_interested'].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=target_counts.index, values=target_counts.values,
            hole=0.55,
            marker_colors=['#2563eb', '#e2e8f0'],
            textinfo='percent+label',
            textfont_size=12
        ))
        fig2.update_layout(title='Platform Interest', height=340,
                           margin=dict(l=10, r=10, t=40, b=10),
                           showlegend=False,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Occupation + Income ──
    col3, col4 = st.columns(2)

    with col3:
        occ = df_raw['q4_occupation'].value_counts().reset_index()
        occ.columns = ['Occupation', 'Count']
        fig3 = px.bar(occ, x='Count', y='Occupation', orientation='h',
                      color='Count', color_continuous_scale='Blues',
                      title='Respondents by Occupation')
        fig3.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10),
                           coloraxis_showscale=False,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        inc = df_raw[df_raw['q5_income_range'] != 'Prefer not to say']['q5_income_range'].value_counts()
        order = ['Below 10000', '10000-20000', '20000-35000', '35000-60000', 'Above 60000']
        inc = inc.reindex([x for x in order if x in inc.index])
        fig4 = px.bar(x=inc.index, y=inc.values,
                      labels={'x': 'Income Range (₹)', 'y': 'Count'},
                      title='Income Distribution',
                      color=inc.values, color_continuous_scale='Teal')
        fig4.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10),
                           coloraxis_showscale=False,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: Category popularity + Pain points ──
    col5, col6 = st.columns(2)

    with col5:
        from data_utils import CATEGORIES_LIST
        cat_counts = {}
        for cat in CATEGORIES_LIST:
            keyword = cat.split('/')[0].split('&')[0].strip()
            cnt = df_raw['q16_categories_bought'].fillna('').str.contains(keyword, case=False).sum()
            cat_counts[cat] = cnt
        cat_df = pd.DataFrame(list(cat_counts.items()), columns=['Category', 'Count'])
        cat_df = cat_df.sort_values('Count', ascending=True)
        fig5 = px.bar(cat_df, x='Count', y='Category', orientation='h',
                      color='Count', color_continuous_scale='Viridis',
                      title='Most Purchased Categories')
        fig5.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10),
                           coloraxis_showscale=False,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        frustration_options = ['Out-of-stock products', 'Higher prices than online',
                               'Limited product variety', 'No home delivery',
                               'No digital payment', 'No personalised offers', 'No frustrations']
        frust_counts = {}
        for f in frustration_options:
            keyword = f.split()[0]
            cnt = df_raw['q23_frustrations'].fillna('').str.contains(keyword, case=False).sum()
            frust_counts[f] = cnt
        frust_df = pd.DataFrame(list(frust_counts.items()), columns=['Frustration', 'Count'])
        frust_df = frust_df[frust_df['Frustration'] != 'No frustrations']
        frust_df = frust_df.sort_values('Count', ascending=True)
        fig6 = px.bar(frust_df, x='Count', y='Frustration', orientation='h',
                      color='Count', color_continuous_scale='Reds',
                      title='Top Customer Frustrations with Local Shops')
        fig6.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10),
                           coloraxis_showscale=False,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig6, use_container_width=True)

    # ── Row 4: Age distribution + Payment methods ──
    col7, col8 = st.columns(2)

    with col7:
        age_order = ['Under 18', '18-25', '26-35', '36-45', '46-55', '56 and above']
        age_data = df_raw['q1_age_group'].value_counts().reindex(age_order).fillna(0)
        fig7 = px.bar(x=age_data.index, y=age_data.values,
                      labels={'x': 'Age Group', 'y': 'Count'},
                      color=age_data.values, color_continuous_scale='Purples',
                      title='Age Distribution')
        fig7.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10),
                           coloraxis_showscale=False,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig7, use_container_width=True)

    with col8:
        pay_data = df_raw['q13_payment_method'].value_counts()
        fig8 = px.pie(values=pay_data.values, names=pay_data.index,
                      title='Payment Method Split',
                      color_discrete_sequence=px.colors.sequential.Blues_r)
        fig8.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10),
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig8, use_container_width=True)

    # ── Udhaar & Income pattern insights ──
    col9, col10 = st.columns(2)
    with col9:
        udhaar_data = df_raw['q12_udhaar'].value_counts()
        colors_u = ['#1e3a5f', '#2563eb', '#60a5fa', '#bfdbfe']
        fig9 = px.pie(values=udhaar_data.values, names=udhaar_data.index,
                      title='Udhaar (Credit) Distribution',
                      color_discrete_sequence=colors_u)
        fig9.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10),
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig9, use_container_width=True)

    with col10:
        inc_pat = df_raw['q6_income_pattern'].value_counts()
        fig10 = px.bar(x=inc_pat.values, y=inc_pat.index, orientation='h',
                       title='Income Pattern — Steady vs Seasonal',
                       color=inc_pat.values, color_continuous_scale='Oranges')
        fig10.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10),
                            coloraxis_showscale=False,
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig10, use_container_width=True)

    # ── Persona summary table ──
    st.markdown("### Persona Distribution Summary")
    persona_summary = df_raw.groupby('persona_label').agg(
        Count=('respondent_id', 'count'),
        Interested=('target_interested', lambda x: (x == 'Interested').sum()),
    ).reset_index()
    persona_summary['Interest Rate'] = (persona_summary['Interested'] / persona_summary['Count'] * 100).round(1).astype(str) + '%'
    persona_summary['Share of Total'] = (persona_summary['Count'] / total * 100).round(1).astype(str) + '%'
    st.dataframe(persona_summary.rename(columns={'persona_label': 'Persona'}),
                 use_container_width=True, hide_index=True)
