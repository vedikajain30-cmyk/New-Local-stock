import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from data_utils import get_basket_transactions


def transactions_to_df(transactions):
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    return pd.DataFrame(te_array, columns=te.columns_)


@st.cache_data
def run_arm(df_raw, col, min_support, min_confidence, min_lift):
    transactions = get_basket_transactions(df_raw, col)
    if len(transactions) < 10:
        return pd.DataFrame(), pd.DataFrame()
    basket_df = transactions_to_df(transactions)
    try:
        freq = apriori(basket_df, min_support=min_support, use_colnames=True)
        if freq.empty:
            return freq, pd.DataFrame()
        rules = association_rules(freq, metric='lift', min_threshold=min_lift)
        rules = rules[rules['confidence'] >= min_confidence]
        rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        rules['support'] = rules['support'].round(4)
        rules['confidence'] = rules['confidence'].round(4)
        rules['lift'] = rules['lift'].round(4)
        return freq, rules
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def render(df_raw, df_enc):
    st.markdown("## Product Association Intelligence")
    st.markdown("*Apriori algorithm — what customers buy together, and what to bundle/stock together*")

    # ── Controls ──
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        min_support = st.slider("Min Support", 0.02, 0.30, 0.05, 0.01,
                                help="Fraction of customers who bought this combination")
    with col_ctrl2:
        min_confidence = st.slider("Min Confidence", 0.20, 0.90, 0.40, 0.05,
                                   help="P(B | A) — if they buy A, how likely is B?")
    with col_ctrl3:
        min_lift = st.slider("Min Lift", 1.0, 3.0, 1.2, 0.1,
                             help="Lift > 1 means A and B are positively correlated")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "Category Co-Purchase", "Same-Trip Basket", "Bundle Preferences"
    ])

    datasets = {
        "Category Co-Purchase": ('q16_categories_bought', 'What product categories are bought together?'),
        "Same-Trip Basket": ('q17_same_trip_items', 'What items are picked up alongside grocery staples?'),
        "Bundle Preferences": ('q18_bundle_interest', 'Which bundle combinations are co-preferred by customers?')
    }

    for tab, (tab_name, (col, description)) in zip([tab1, tab2, tab3], datasets.items()):
        with tab:
            st.markdown(f"*{description}*")
            freq_items, rules = run_arm(df_raw, col, min_support, min_confidence, min_lift)

            if rules.empty:
                st.warning("No rules found with current thresholds. Try lowering min support or min confidence.")
                continue

            # ── Metrics ──
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Rules Found", len(rules))
            with m2:
                st.metric("Max Lift", f"{rules['lift'].max():.2f}")
            with m3:
                st.metric("Max Confidence", f"{rules['confidence'].max():.2f}")
            with m4:
                st.metric("Frequent Itemsets", len(freq_items) if not freq_items.empty else 0)

            # ── Rules table ──
            st.markdown("#### Top Association Rules")
            display_rules = rules[['antecedents', 'consequents', 'support',
                                   'confidence', 'lift']].head(20).copy()
            display_rules.columns = ['If customer buys...', 'They also buy...', 'Support', 'Confidence', 'Lift']
            display_rules['Confidence'] = display_rules['Confidence'].apply(lambda x: f"{x:.1%}")
            display_rules['Lift'] = display_rules['Lift'].apply(lambda x: f"{x:.2f}x")
            display_rules['Support'] = display_rules['Support'].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_rules, use_container_width=True, hide_index=True)

            # ── Scatter: Support vs Confidence coloured by Lift ──
            col_a, col_b = st.columns(2)
            with col_a:
                fig_scatter = px.scatter(
                    rules.head(50), x='support', y='confidence',
                    size='lift', color='lift',
                    color_continuous_scale='YlOrRd',
                    hover_data=['antecedents', 'consequents'],
                    title='Support vs Confidence (size = Lift)',
                    labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'}
                )
                fig_scatter.update_layout(height=340,
                                           margin=dict(l=10, r=10, t=40, b=10),
                                           plot_bgcolor='rgba(0,0,0,0)',
                                           paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col_b:
                # Top rules bar chart by lift
                top_rules = rules.head(10).copy()
                top_rules['Rule'] = top_rules['antecedents'].str[:20] + ' → ' + top_rules['consequents'].str[:20]
                fig_lift = px.bar(top_rules[::-1], x='lift', y='Rule', orientation='h',
                                  color='confidence',
                                  color_continuous_scale='Blues',
                                  title='Top 10 Rules by Lift',
                                  labels={'lift': 'Lift', 'confidence': 'Confidence'})
                fig_lift.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10),
                                       plot_bgcolor='rgba(0,0,0,0)',
                                       paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_lift, use_container_width=True)

            # ── Network graph ──
            st.markdown("#### Association Network — Top Rules")
            top_net = rules.head(15)

            # Build network using plotly
            all_nodes = list(set(list(top_net['antecedents']) + list(top_net['consequents'])))
            node_idx = {n: i for i, n in enumerate(all_nodes)}

            edge_x, edge_y = [], []
            node_x = np.random.uniform(0.1, 0.9, len(all_nodes))
            node_y = np.random.uniform(0.1, 0.9, len(all_nodes))

            for _, row in top_net.iterrows():
                i1 = node_idx[row['antecedents']]
                i2 = node_idx[row['consequents']]
                edge_x += [node_x[i1], node_x[i2], None]
                edge_y += [node_y[i1], node_y[i2], None]

            fig_net = go.Figure()
            fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                          line=dict(width=1, color='#94a3b8'),
                                          hoverinfo='none'))
            fig_net.add_trace(go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                marker=dict(size=14, color='#2563eb',
                            line=dict(width=1.5, color='white')),
                text=[n[:20] for n in all_nodes],
                textposition='top center',
                textfont=dict(size=10),
                hoverinfo='text'
            ))
            fig_net.update_layout(
                height=360, showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(l=10, r=10, t=20, b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_net, use_container_width=True)

            # ── Business recommendations ──
            st.markdown("#### Shop Owner Recommendations")
            top3 = rules.head(3)
            for _, row in top3.iterrows():
                conf_pct = f"{row['confidence']:.0%}"
                lift_val = f"{row['lift']:.1f}"
                st.success(
                    f"**Stock together:** {row['antecedents']} + {row['consequents']} "
                    f"— customers who buy the first, buy the second {conf_pct} of the time "
                    f"(Lift: {lift_val}x). Consider creating a bundle offer."
                )
