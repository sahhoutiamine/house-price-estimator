import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Ensure src is in the path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import build_model_input

# Configuration of the Streamlit page
st.set_page_config(page_title="House Price Estimator", layout="centered")

# Hide Streamlit default icons and menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    """
    Load models and preprocessing artifacts ONCE to save time.
    Uses @st.cache_resource to persist across user interactions.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Load the trained model
    model_path = os.path.join(base_dir, "models", "best_model.pkl")
    model = joblib.load(model_path)
    
    # Load the preprocessing artifacts
    scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
    scaler = joblib.load(scaler_path)
    
    feature_cols_path = os.path.join(base_dir, "models", "feature_columns.pkl")
    feature_columns = joblib.load(feature_cols_path)
    
    cont_cols_path = os.path.join(base_dir, "models", "continuous_columns.pkl")
    continuous_columns = joblib.load(cont_cols_path)
    
    default_row_path = os.path.join(base_dir, "models", "default_row.pkl")
    default_row = joblib.load(default_row_path)
    
    # Load raw data to extract sorted unique neighborhoods and max YrSold
    raw_data_path = os.path.join(base_dir, "data", "raw", "House_Prices.csv")
    df_raw = pd.read_csv(raw_data_path)
    neighborhoods = sorted(df_raw["Neighborhood"].dropna().unique())
    max_yr_sold = df_raw["YrSold"].max()
    
    # Load training targets to plot the historic price distribution
    y_train_path = os.path.join(base_dir, "data", "processed", "y_train.csv")
    y_train = pd.read_csv(y_train_path)["SalePrice"]
    
    return model, scaler, feature_columns, continuous_columns, default_row, neighborhoods, max_yr_sold, y_train

# Load everything using the cached function
model, scaler, feature_columns, continuous_columns, default_row, neighborhoods, max_yr_sold, y_train = load_resources()

st.title("House Price Estimator")
st.markdown("Welcome! Fill out the details below to get an estimated sale price for your house.")

# Build the form
with st.form("house_price_form"):
    st.subheader("1. General Size & Lot")
    col1, col2 = st.columns(2)
    with col1:
        gr_liv_area = st.number_input("Living Area (sq ft) - GrLivArea", min_value=100, max_value=10000, value=1500, step=50)
        lot_area = st.number_input("Lot Area (sq ft)", min_value=500, max_value=50000, value=10000, step=500)
    with col2:
        total_bsmt_sf = st.number_input("Basement Area (sq ft)", min_value=0, max_value=5000, value=1000, step=50)
        
    st.subheader("2. Rooms & Bathrooms")
    col3, col4 = st.columns(2)
    with col3:
        bedroom_abv_gr = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
        fireplaces = st.number_input("Fireplaces", min_value=0, max_value=5, value=0)
    with col4:
        full_bath = st.number_input("Full Bathrooms", min_value=0, max_value=5, value=2)
        half_bath = st.number_input("Half Bathrooms", min_value=0, max_value=5, value=0)
        
    st.subheader("3. Location & Quality")
    neighborhood = st.selectbox("Neighborhood", options=neighborhoods)
    
    # Overall Quality and Condition as sliders
    overall_qual = st.slider("Overall Quality (1=Poor, 10=Excellent)", min_value=1, max_value=10, value=6)
    overall_cond = st.slider("Overall Condition (1=Poor, 10=Excellent)", min_value=1, max_value=10, value=5)
    
    st.subheader("4. Age & Garage")
    col5, col6 = st.columns(2)
    with col5:
        year_built = st.number_input("Year Built", min_value=1800, max_value=int(max_yr_sold), value=1990)
        year_remod_add = st.number_input("Year Remodeled (default = Year Built)", min_value=1800, max_value=int(max_yr_sold), value=1990)
    with col6:
        garage_cars = st.number_input("Garage Cars Capacity", min_value=0, max_value=5, value=2)
        garage_area = st.number_input("Garage Area (sq ft)", min_value=0, max_value=2000, value=400, step=50)

    # Note about the fixed YrSold
    st.caption(f"*Note: For the calculation of the house age, the year sold is fixed to the most recent historical year in our dataset ({max_yr_sold}). All other unspecified features are filled with average historical defaults.*")

    submitted = st.form_submit_button("Estimate Price")

# When the user clicks the submit button
if submitted:
    # 1. Gather all inputs into a dictionary
    user_inputs = {
        "GrLivArea": gr_liv_area,
        "LotArea": lot_area,
        "TotalBsmtSF": total_bsmt_sf,
        "BedroomAbvGr": bedroom_abv_gr,
        "Fireplaces": fireplaces,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "Neighborhood": neighborhood,
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod_add,
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "YrSold": max_yr_sold # Keep consistent with the training dataset's max timeline
    }
    
    # 2. Preprocess the input
    try:
        X_new = build_model_input(
            user_inputs=user_inputs,
            default_row=default_row,
            scaler=scaler,
            feature_columns=feature_columns,
            continuous_columns=continuous_columns
        )
        
        # 3. Predict using the loaded model
        # Our target was modeled via TransformedTargetRegressor with log1p, 
        # but predict() automatically applies expm1 to bring it back to dollars!
        pred_price = model.predict(X_new)[0]
        
        # 4. Display result
        st.success(f"### Estimated Sale Price: **${pred_price:,.0f}**")
        
        # 5. Show useful information chart
        st.markdown("---")
        st.subheader("How does this compare?")
        st.write("The chart below shows your predicted price compared to the historical distribution of house prices in Ames.")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(y_train, bins=50, kde=True, color='skyblue', ax=ax)
        ax.axvline(x=pred_price, color='red', linestyle='--', linewidth=2, label='Your Estimate')
        ax.set_title("Distribution of Historical Sale Prices")
        ax.set_xlabel("Sale Price ($)")
        ax.set_ylabel("Number of Houses")
        
        # Adjust x-axis formatting for thousands (K)
        xticks = ax.get_xticks()
        
        # Avoid formatting errors if xticks contains floats that don't cast neatly or are negative
        # We can just let matplotlib handle it, or format cleanly:
        # ax.set_xticklabels([f"${int(x/1000)}k" for x in xticks])
        
        ax.legend()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
