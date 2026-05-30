import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

ORDINAL_MAPS = {
    'q1_age_group': {'Under 18': 0, '18-25': 1, '26-35': 2, '36-45': 3, '46-55': 4, '56 and above': 5},
    'q5_income_range': {'Below 10000': 1, '10000-20000': 2, '20000-35000': 3,
                        '35000-60000': 4, 'Above 60000': 5, 'Prefer not to say': 3},
    'q7_household_size': {'1-2': 1, '3-4': 2, '5-6': 3, '7 or more': 4},
    'q9_shop_frequency': {'Daily': 6, '3-4 times a week': 5, 'Once a week': 4,
                          'Once a fortnight': 3, 'Once a month': 2, 'Rarely': 1},
    'q10_num_shops': {'Only 1 - very loyal': 1, '2-3 shops': 2, '4-5 shops': 3, 'More than 5': 4},
    'q11_monthly_spend': {'Below 500': 1, '500-1500': 2, '1500-3000': 3,
                          '3000-6000': 4, '6000-10000': 5, 'Above 10000': 6},
    'q12_udhaar': {'No udhaar': 0, 'Up to 500/month': 1, '500-2000/month': 2, 'Above 2000/month': 3},
    'q14_distance_tolerance': {'Within 500m': 1, 'Up to 1 km': 2, 'Up to 3 km': 3,
                                'Up to 5 km': 4, 'Distance doesnt matter': 5},
    'q15_online_shopping': {'Never': 1, 'Occasionally 1-2x/month': 2,
                             'Regularly weekly': 3, 'Mostly online rarely local': 4},
    'q21_discount_threshold': {'Price not my main factor': 0, '30% or more off': 1,
                                'At least 15-20% off': 2, 'At least 5-10% off': 3,
                                'Any saving (even 5 Rs)': 4},
    'q6_income_pattern': {'Steady monthly': 1, 'Mix of both': 2,
                           'Irregular/unpredictable': 3, 'Mostly seasonal/harvest-based': 4},
    'q24_app_willingness': {'No - dont use smartphones for this': 1, 'Unlikely': 2,
                             'Maybe if offers are good': 3, 'Yes definitely': 4},
    'q25_platform_interest': {'Very unlikely': 1, 'Unlikely': 2, 'Neutral': 3,
                               'Likely': 4, 'Very likely': 5},
}

NOMINAL_COLS = ['q2_gender', 'q3_city', 'q4_occupation', 'q8_decision_maker',
                'q13_payment_method', 'q19_stockout_behaviour',
                'q20_product_discovery', 'q22_brand_preference']

CATEGORIES_LIST = [
    'Grocery & Staples', 'Snacks & Beverages', 'Personal Care/Hygiene',
    'Medicines/OTC', 'Agri Inputs/Seeds', 'Mobile Accessories',
    'Puja/Religious Items', 'Stationery/School', 'Household Cleaning',
    'Clothing/Footwear', 'Animal Feed', 'Seasonal Items'
]

SAME_TRIP_LIST = [
    'Snacks & Biscuits', 'Personal Care Items', 'Cleaning Supplies',
    'Puja Items', 'Beverages (chai/coffee)', 'OTC Medicines', 'Nothing specific'
]

BUNDLE_LIST = [
    'Atta+Dal+Oil combo', 'Soap+Shampoo+Toothpaste', 'Seeds+Fertiliser+Pesticide',
    'Snacks+Beverages pack', 'Mosquito coils+Repellent',
    'School stationery bundle', 'Puja essentials kit', 'No bundling interest'
]

FEATURE_COLS = [
    'q1_age_group_enc', 'q2_gender_enc', 'q3_city_enc', 'q4_occupation_enc',
    'q5_income_range_enc', 'q6_income_pattern_enc', 'q7_household_size_enc',
    'q8_decision_maker_enc', 'q9_shop_frequency_enc', 'q10_num_shops_enc',
    'q11_monthly_spend_enc', 'q12_udhaar_enc', 'q13_payment_method_enc',
    'q14_distance_tolerance_enc', 'q15_online_shopping_enc',
    'q19_stockout_behaviour_enc', 'q20_product_discovery_enc',
    'q21_discount_threshold_enc', 'q22_brand_preference_enc',
    'q24_app_willingness_enc',
    'cat_Grocery', 'cat_Snacks', 'cat_Personal', 'cat_Medicines',
    'cat_Agri', 'cat_Mobile', 'cat_Puja', 'cat_Stationery',
    'cat_Household', 'cat_Clothing', 'cat_Animal', 'cat_Seasonal'
]

FEATURE_LABELS = {
    'q1_age_group_enc': 'Age Group',
    'q2_gender_enc': 'Gender',
    'q3_city_enc': 'City',
    'q4_occupation_enc': 'Occupation',
    'q5_income_range_enc': 'Income Range',
    'q6_income_pattern_enc': 'Income Pattern',
    'q7_household_size_enc': 'Household Size',
    'q8_decision_maker_enc': 'Decision Maker',
    'q9_shop_frequency_enc': 'Shop Frequency',
    'q10_num_shops_enc': 'No. of Shops Visited',
    'q11_monthly_spend_enc': 'Monthly Spend',
    'q12_udhaar_enc': 'Udhaar Level',
    'q13_payment_method_enc': 'Payment Method',
    'q14_distance_tolerance_enc': 'Distance Tolerance',
    'q15_online_shopping_enc': 'Online Shopping Freq',
    'q19_stockout_behaviour_enc': 'Stockout Behaviour',
    'q20_product_discovery_enc': 'Product Discovery',
    'q21_discount_threshold_enc': 'Discount Threshold',
    'q22_brand_preference_enc': 'Brand Preference',
    'q24_app_willingness_enc': 'App Willingness',
    'cat_Grocery': 'Buys: Grocery',
    'cat_Snacks': 'Buys: Snacks',
    'cat_Personal': 'Buys: Personal Care',
    'cat_Medicines': 'Buys: Medicines',
    'cat_Agri': 'Buys: Agri Inputs',
    'cat_Mobile': 'Buys: Mobile Acc.',
    'cat_Puja': 'Buys: Puja Items',
    'cat_Stationery': 'Buys: Stationery',
    'cat_Household': 'Buys: Household',
    'cat_Clothing': 'Buys: Clothing',
    'cat_Animal': 'Buys: Animal Feed',
    'cat_Seasonal': 'Buys: Seasonal'
}


def encode_dataframe(df_raw):
    df = df_raw.copy()
    # Fill NaN in income
    if 'q5_income_range' in df.columns:
        df['q5_income_range'] = df['q5_income_range'].fillna('Prefer not to say')

    # Ordinal encoding
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            enc_col = col + '_enc'
            df[enc_col] = df[col].map(mapping).fillna(mapping.get(
                list(mapping.keys())[len(mapping)//2], 2)).astype(float)

    # Nominal encoding
    le = LabelEncoder()
    for col in NOMINAL_COLS:
        if col in df.columns:
            df[col + '_enc'] = le.fit_transform(df[col].fillna('Unknown').astype(str))

    # Multi-select binary flags
    cat_keywords = {
        'cat_Grocery': 'Grocery', 'cat_Snacks': 'Snacks',
        'cat_Personal': 'Personal', 'cat_Medicines': 'Medicines',
        'cat_Agri': 'Agri', 'cat_Mobile': 'Mobile',
        'cat_Puja': 'Puja', 'cat_Stationery': 'Stationery',
        'cat_Household': 'Household', 'cat_Clothing': 'Clothing',
        'cat_Animal': 'Animal', 'cat_Seasonal': 'Seasonal'
    }
    for flag, keyword in cat_keywords.items():
        if 'q16_categories_bought' in df.columns:
            df[flag] = df['q16_categories_bought'].fillna('').str.contains(
                keyword, case=False, na=False).astype(int)
        else:
            df[flag] = 0

    # Target
    if 'target_interested' in df.columns:
        df['target_enc'] = (df['target_interested'] == 'Interested').astype(int)

    return df


def get_feature_matrix(df_encoded, feature_cols=None):
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    available = [c for c in feature_cols if c in df_encoded.columns]
    X = df_encoded[available].fillna(0)
    return X, available


def get_basket_transactions(df_raw, col='q16_categories_bought'):
    transactions = []
    for val in df_raw[col].dropna():
        items = [i.strip() for i in str(val).split('|') if i.strip()]
        if items:
            transactions.append(items)
    return transactions


def spend_band_label(val):
    mapping = {1: '<₹500', 2: '₹500–1.5k', 3: '₹1.5k–3k',
                4: '₹3k–6k', 5: '₹6k–10k', 6: '>₹10k'}
    return mapping.get(int(round(val)), str(val))
