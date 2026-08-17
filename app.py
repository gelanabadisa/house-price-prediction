"""
app.py
------
Streamlit dashboard: user enters house features, trained model predicts price.

Run with:
    streamlit run app.py
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join("models", "house_price_model.pkl")
METADATA_PATH = os.path.join("models", "metadata.json")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")


@st.cache_resource
def load_model_and_metadata():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return model, metadata


model, metadata = load_model_and_metadata()

st.title("🏠 House Price Prediction")
st.write(
    "Enter the property details below and the model will estimate the sale price."
)

if model is None:
    st.error(
        "No trained model found. Run `python train_model.py --data data/Housing.csv` "
        "first, then restart this app."
    )
    st.stop()

st.caption(f"Model in use: **{metadata['best_model']}**  |  Test R²: **{metadata['metrics'][metadata['best_model']]['r2']}**")

with st.form("prediction_form"):
    st.subheader("Property details")

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=5000, step=50)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
        stories = st.number_input("Stories", min_value=1, max_value=5, value=2, step=1)
        parking = st.number_input("Parking spaces", min_value=0, max_value=5, value=1, step=1)

    with col2:
        mainroad = st.radio("On main road?", ["yes", "no"], horizontal=True)
        guestroom = st.radio("Has guest room?", ["yes", "no"], horizontal=True)
        basement = st.radio("Has basement?", ["yes", "no"], horizontal=True)
        hotwaterheating = st.radio("Hot water heating?", ["yes", "no"], horizontal=True)
        airconditioning = st.radio("Air conditioning?", ["yes", "no"], horizontal=True)

    prefarea = st.radio("Preferred area?", ["yes", "no"], horizontal=True)
    furnishingstatus = st.selectbox(
        "Furnishing status", ["furnished", "semi-furnished", "unfurnished"]
    )

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

if submitted:
    input_df = pd.DataFrame(
        [
            {
                "area": area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "stories": stories,
                "parking": parking,
                "mainroad": 1 if mainroad == "yes" else 0,
                "guestroom": 1 if guestroom == "yes" else 0,
                "basement": 1 if basement == "yes" else 0,
                "hotwaterheating": 1 if hotwaterheating == "yes" else 0,
                "airconditioning": 1 if airconditioning == "yes" else 0,
                "prefarea": 1 if prefarea == "yes" else 0,
                "furnishingstatus": furnishingstatus,
            }
        ]
    )

    prediction = model.predict(input_df)[0]

    st.success(f"### Estimated Price: {prediction:,.0f}")
    st.caption(
        "This is a model estimate based on historical data, not a guaranteed valuation."
    )

    with st.expander("See input summary"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption("Built with scikit-learn + Streamlit.")
