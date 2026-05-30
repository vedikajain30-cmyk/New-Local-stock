import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_curve, auc, confusion_matrix,
                              classification_report)
from sklearn.preprocessing import StandardScaler
import pickle, os
from data_utils import FEATURE_COLS, FEATURE_LABELS, get_feature_matrix


def get_friendly_labels(available_features):
    return [FEATURE_LABELS.get(f, f) for f in available_features]


@st.cache_resource
def train_models(df_enc):
    X, available = get_feature_matrix(df_enc)
    y = df_enc['target_enc'].fillna(0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42)
    }

    trained = {}
    for name, model in models.items():
        if name == 'Random Forest' or name == 'Decision Tree':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_prob = model.predict_proba(X_test_s)[:, 1]

        trained[name] = {
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'X_test': X_test,
            'X_test_s': X_test_s,
            'scaler': scaler,
            'available': available,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
    # Save best model (RF) as pickle for upload predictor page
    best = trained['Random Forest']
    model_data = {
        'model': best['model'],
        'scaler': best['scaler'],
        'features': available
    }
    with open('best_classifier.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    return trained, X_train, X_test, y_train, y_test, scaler, available


def render(df_raw, df_enc):
    st.markdown("## Conversion Predictor — Classification")
    st.markdown("*Which customers are likely to be interested in your platform?*")

    trained, X_train, X_test, y_train, y_test, scaler, available = train_models(df_enc)

    model_choice = st.selectbox("Select Algorithm", list(trained.keys()))
    result = trained[model_choice]

    # ── Metrics Row ──
    st.markdown("### Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Accuracy", f"{result['accuracy']:.3f}", f"{result['accuracy']*100:.1f}%")
    with m2:
        st.metric("Precision", f"{result['precision']:.3f}")
    with m3:
        st.metric("Recall", f"{result['recall']:.3f}")
    with m4:
        st.metric("F1 Score", f"{result['f1']:.3f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # ── Confusion Matrix ──
    with col1:
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=['Predicted: Not Interested', 'Predicted: Interested'],
            y=['Actual: Not Interested', 'Actual: Interested'],
            colorscale='Blues', showscale=True,
            text=[[str(v) for v in row] for row in cm],
            texttemplate='%{text}', textfont=dict(size=16, color='black')
        ))
        fig_cm.update_layout(
            title=f'Confusion Matrix — {model_choice}',
            height=340, margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── ROC Curve ──
    with col2:
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_prob'])
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                      line=dict(color='#2563eb', width=2.5),
                                      name=f'ROC (AUC = {roc_auc:.3f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                      line=dict(color='gray', dash='dash', width=1),
                                      name='Random Classifier'))
        fig_roc.update_layout(
            title=f'ROC Curve — AUC: {roc_auc:.3f}',
            xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
            height=340, margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(x=0.55, y=0.1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Feature Importance ──
    st.markdown("### Feature Importance — What Drives Customer Interest?")
    friendly_labels = get_friendly_labels(available)

    if model_choice in ['Random Forest', 'Decision Tree']:
        importances = result['model'].feature_importances_
        fi_df = pd.DataFrame({'Feature': friendly_labels, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=True).tail(15)
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Blues',
                        title=f'Feature Importance — {model_choice}')
        fig_fi.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10),
                              coloraxis_showscale=False,
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_fi, use_container_width=True)

    elif model_choice == 'Logistic Regression':
        coeffs = result['model'].coef_[0]
        fi_df = pd.DataFrame({'Feature': friendly_labels, 'Coefficient': coeffs})
        fi_df['abs_coef'] = fi_df['Coefficient'].abs()
        fi_df = fi_df.sort_values('abs_coef', ascending=True).tail(15)
        fi_df['Direction'] = fi_df['Coefficient'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
        fig_fi = px.bar(fi_df, x='Coefficient', y='Feature', orientation='h',
                        color='Direction',
                        color_discrete_map={'Positive': '#2563eb', 'Negative': '#dc2626'},
                        title='Logistic Regression Coefficients')
        fig_fi.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── All 3 model comparison ──
    st.markdown("### Model Comparison")
    comparison = []
    for name, res in trained.items():
        fpr_c, tpr_c, _ = roc_curve(res['y_test'], res['y_prob'])
        auc_c = auc(fpr_c, tpr_c)
        comparison.append({
            'Model': name,
            'Accuracy': f"{res['accuracy']:.4f}",
            'Precision': f"{res['precision']:.4f}",
            'Recall': f"{res['recall']:.4f}",
            'F1 Score': f"{res['f1']:.4f}",
            'AUC-ROC': f"{auc_c:.4f}"
        })
    comp_df = pd.DataFrame(comparison)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ── ROC all models ──
    st.markdown("### ROC Curves — All Models")
    fig_all_roc = go.Figure()
    colors = ['#2563eb', '#16a34a', '#d97706']
    for i, (name, res) in enumerate(trained.items()):
        fpr_i, tpr_i, _ = roc_curve(res['y_test'], res['y_prob'])
        auc_i = auc(fpr_i, tpr_i)
        fig_all_roc.add_trace(go.Scatter(
            x=fpr_i, y=tpr_i, mode='lines',
            line=dict(color=colors[i], width=2),
            name=f'{name} (AUC={auc_i:.3f})'
        ))
    fig_all_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                      line=dict(color='gray', dash='dash', width=1),
                                      name='Random'))
    fig_all_roc.update_layout(
        xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
        height=380, margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_all_roc, use_container_width=True)

    # ── Classification report ──
    with st.expander("Detailed Classification Report"):
        report = classification_report(result['y_test'], result['y_pred'],
                                       target_names=['Not Interested', 'Interested'])
        st.code(report)

    # ── Predicted probability distribution ──
    st.markdown("### Predicted Probability Distribution")
    fig_dist = go.Figure()
    mask_int = result['y_test'] == 1
    fig_dist.add_trace(go.Histogram(
        x=result['y_prob'][mask_int], name='Actually Interested',
        opacity=0.65, marker_color='#2563eb', nbinsx=30
    ))
    fig_dist.add_trace(go.Histogram(
        x=result['y_prob'][~mask_int], name='Not Interested',
        opacity=0.65, marker_color='#dc2626', nbinsx=30
    ))
    fig_dist.update_layout(
        barmode='overlay', xaxis_title='Predicted Probability of Interest',
        yaxis_title='Count', height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_dist, use_container_width=True)

    st.info("The trained Random Forest model has been saved as **best_classifier.pkl** for use in the New Customer Predictor page.")
