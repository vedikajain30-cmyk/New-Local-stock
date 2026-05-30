import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
from data_utils import FEATURE_LABELS, get_feature_matrix

REGRESSION_FEATURES = [
    'q1_age_group_enc', 'q4_occupation_enc', 'q5_income_range_enc',
    'q6_income_pattern_enc', 'q7_household_size_enc',
    'q9_shop_frequency_enc', 'q10_num_shops_enc', 'q12_udhaar_enc',
    'q14_distance_tolerance_enc', 'q15_online_shopping_enc',
    'q21_discount_threshold_enc', 'q22_brand_preference_enc',
    'q3_city_enc', 'q2_gender_enc', 'q8_decision_maker_enc',
    'cat_Grocery', 'cat_Agri', 'cat_Personal', 'cat_Mobile', 'cat_Snacks'
]

SPEND_LABELS = {1: '<₹500', 2: '₹500–1.5k', 3: '₹1.5k–3k',
                 4: '₹3k–6k', 5: '₹6k–10k', 6: '>₹10k'}


@st.cache_resource
def train_regression_models(df_enc):
    available = [c for c in REGRESSION_FEATURES if c in df_enc.columns]
    X = df_enc[available].fillna(0)
    y = df_enc['q11_monthly_spend_enc'].fillna(3).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'Random Forest Regressor': RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0)
    }

    trained = {}
    for name, model in models.items():
        if name == 'Random Forest Regressor':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_pred = np.clip(y_pred, 1, 6)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        trained[name] = {
            'model': model, 'y_test': y_test, 'y_pred': y_pred,
            'X_test': X_test, 'X_test_s': X_test_s,
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'scaler': scaler, 'available': available
        }

    # Save RF regressor
    rf_data = {
        'model': trained['Random Forest Regressor']['model'],
        'scaler': scaler,
        'features': available
    }
    with open('best_regressor.pkl', 'wb') as f:
        pickle.dump(rf_data, f)

    return trained, available


def render(df_raw, df_enc):
    st.markdown("## Spending Power Model — Regression")
    st.markdown("*Predicting how much each customer will spend monthly — foundation for LTV calculation*")

    trained, available = train_regression_models(df_enc)

    model_choice = st.selectbox("Select Regression Model", list(trained.keys()))
    result = trained[model_choice]

    # ── Metrics ──
    st.markdown("### Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("R² Score", f"{result['r2']:.4f}",
                  help="Proportion of variance explained (1.0 = perfect)")
    with m2:
        st.metric("MAE (bands)", f"{result['mae']:.3f}",
                  help="Mean absolute error in spend band units (1 band ≈ ₹500–1.5k)")
    with m3:
        st.metric("RMSE", f"{result['rmse']:.3f}")
    with m4:
        r2_pct = max(0, result['r2']) * 100
        st.metric("Explained Variance", f"{r2_pct:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # ── Actual vs Predicted ──
    with col1:
        y_test_arr = np.array(result['y_test'])
        y_pred_arr = np.array(result['y_pred'])
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=y_test_arr + np.random.normal(0, 0.08, len(y_test_arr)),
            y=y_pred_arr,
            mode='markers', marker=dict(size=4, color='#2563eb', opacity=0.4),
            name='Predictions'
        ))
        fig_avp.add_trace(go.Scatter(x=[1, 6], y=[1, 6], mode='lines',
                                      line=dict(color='#dc2626', dash='dash', width=2),
                                      name='Perfect Prediction'))
        fig_avp.update_layout(
            title='Actual vs Predicted Spend Band',
            xaxis_title='Actual Spend Band', yaxis_title='Predicted Spend Band',
            xaxis=dict(ticktext=list(SPEND_LABELS.values()), tickvals=list(SPEND_LABELS.keys())),
            yaxis=dict(ticktext=list(SPEND_LABELS.values()), tickvals=list(SPEND_LABELS.keys())),
            height=360, margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_avp, use_container_width=True)

    # ── Residuals ──
    with col2:
        residuals = y_test_arr - y_pred_arr
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y_pred_arr, y=residuals, mode='markers',
            marker=dict(size=4, color='#16a34a', opacity=0.5),
            name='Residuals'
        ))
        fig_res.add_hline(y=0, line=dict(color='#dc2626', dash='dash', width=2))
        fig_res.update_layout(
            title='Residual Plot', xaxis_title='Predicted', yaxis_title='Residual',
            height=360, margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_res, use_container_width=True)

    # ── Feature importance ──
    st.markdown("### What Drives Customer Spending?")
    friendly_labels = [FEATURE_LABELS.get(f, f) for f in available]

    if model_choice == 'Random Forest Regressor':
        importances = result['model'].feature_importances_
        fi_df = pd.DataFrame({'Feature': friendly_labels, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=True).tail(15)
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Greens',
                        title='Feature Importance — Spend Prediction')
        fig_fi.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                              coloraxis_showscale=False,
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        coeffs = result['model'].coef_
        fi_df = pd.DataFrame({'Feature': friendly_labels, 'Coefficient': coeffs})
        fi_df = fi_df.sort_values('Coefficient', ascending=True)
        fi_df['Direction'] = fi_df['Coefficient'].apply(lambda x: 'Increases spend' if x > 0 else 'Decreases spend')
        fig_fi = px.bar(fi_df, x='Coefficient', y='Feature', orientation='h',
                        color='Direction',
                        color_discrete_map={'Increases spend': '#16a34a', 'Decreases spend': '#dc2626'},
                        title='Regression Coefficients — Spend Drivers')
        fig_fi.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Model comparison ──
    st.markdown("### Model Comparison")
    comp = []
    for name, res in trained.items():
        comp.append({'Model': name, 'R² Score': f"{res['r2']:.4f}",
                     'MAE': f"{res['mae']:.4f}", 'RMSE': f"{res['rmse']:.4f}"})
    st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

    # ── Spend distribution predicted vs actual ──
    st.markdown("### Predicted vs Actual — Spend Band Distribution")
    y_pred_rounded = np.clip(np.round(y_pred_arr).astype(int), 1, 6)
    pred_dist = pd.Series(y_pred_rounded).value_counts().sort_index()
    actual_dist = pd.Series(y_test_arr.astype(int)).value_counts().sort_index()

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        x=[SPEND_LABELS.get(k, k) for k in actual_dist.index],
        y=actual_dist.values, name='Actual', marker_color='#2563eb', opacity=0.7
    ))
    fig_dist.add_trace(go.Bar(
        x=[SPEND_LABELS.get(k, k) for k in pred_dist.index],
        y=pred_dist.values, name='Predicted', marker_color='#16a34a', opacity=0.7
    ))
    fig_dist.update_layout(
        barmode='group', xaxis_title='Spend Band', yaxis_title='Count',
        height=320, margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── LTV segments ──
    st.markdown("### Customer LTV Segments — Action Guide")
    ltv_data = {
        'LTV Tier': ['High Value (>₹6k/month)', 'Mid Value (₹3k–6k)', 'Standard (₹1.5k–3k)', 'Low (<₹1.5k)'],
        'Estimated Count': [
            int((df_enc['q11_monthly_spend_enc'] >= 5).sum()),
            int(((df_enc['q11_monthly_spend_enc'] >= 4) & (df_enc['q11_monthly_spend_enc'] < 5)).sum()),
            int(((df_enc['q11_monthly_spend_enc'] >= 3) & (df_enc['q11_monthly_spend_enc'] < 4)).sum()),
            int((df_enc['q11_monthly_spend_enc'] < 3).sum())
        ],
        'Recommended Action': [
            'Premium bundle offers, dedicated account manager for shop, early-access deals',
            'Regular discount vouchers, festival bundles, loyalty cashback',
            'Entry-level offers, awareness campaigns, volume deals',
            'Low-cost acquisition, frequency incentives, referral programs'
        ]
    }
    st.dataframe(pd.DataFrame(ltv_data), use_container_width=True, hide_index=True)
