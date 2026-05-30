# LocalStock Intelligence Dashboard

A data-driven analytics dashboard for kirana store intelligence — built with Streamlit.

## What it does

LocalStock helps identify which customers are most valuable, which products run out together, and which cities to expand into — turning gut-feel retail decisions into data-backed ones.

## Pages

| Page | Description |
|------|-------------|
| 📊 Executive Overview | KPIs, city breakdown, category popularity, customer frustrations |
| 👥 Customer Segmentation | K-Means clustering with radar charts and segment strategy cards |
| 🔗 Product Associations | Market basket analysis — products bought together |
| 🎯 Conversion Predictor | Classification model to predict platform interest |
| 💰 Spending Power Model | Regression model to estimate customer monthly spend |
| 🗺️ City & Market Targeting | Priority scoring for city launch decisions |
| 🔮 New Customer Predictor | Upload new customer data and get predictions |

## Setup & Run locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/localstock-dashboard.git
cd localstock-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (keep both CSV files included)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy** — it will auto-install `requirements.txt`

> **Note:** Both `localstock_survey_raw.csv` and `localstock_survey_encoded.csv` must be in the root of the repo for the app to work.

## Project structure

```
localstock-dashboard/
├── app.py                        # Main entry point & navigation
├── data_utils.py                 # Shared constants and helpers
├── page_overview.py              # Executive Overview page
├── page_clustering.py            # Customer Segmentation page
├── page_association.py           # Product Associations page
├── page_classification.py        # Conversion Predictor page
├── page_regression.py            # Spending Power Model page
├── page_city_targeting.py        # City & Market Targeting page
├── page_upload_predict.py        # New Customer Predictor page
├── localstock_survey_raw.csv     # Raw survey data
├── localstock_survey_encoded.csv # Encoded/ML-ready data
├── requirements.txt              # Python dependencies
└── .streamlit/
    └── config.toml               # Theme and server config
```

## Tech stack

- **Streamlit** — UI framework
- **Pandas / NumPy** — data wrangling
- **Scikit-learn** — ML models (KMeans, classification, regression)
- **Plotly** — interactive charts
- **mlxtend** — association rule mining

## Dataset

Survey data from kirana store customers across multiple Indian cities (Tier 1–3). Contains ~25 features including demographics, spending behaviour, payment preferences, and product categories.

---

Built for LocalStock · India's kirana intelligence platform
