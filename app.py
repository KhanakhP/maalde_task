import os
import tempfile

import streamlit as st

from src.demand_predictor.predictor import Predictor


st.title("Demand Prediction Engine")


@st.cache_resource
def load_predictor():
    return Predictor()


with st.form("prediction_form"):
    uploaded_file = st.file_uploader("Upload Dress Image", type=["jpg", "jpeg", "png"])
    rate = st.number_input("Enter Product Rate", min_value=0.0, value=1000.0, step=50.0)
    submitted = st.form_submit_button("Predict Demand")

if submitted:
    if not uploaded_file:
        st.warning("Please upload a product image first.")
    else:
        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name

        try:
            predictor = load_predictor()
            result = predictor.predict(path, rate)
            st.success(f"Predicted Demand: {result} units")
        except FileNotFoundError:
            st.error("Model not found. Run scripts/prepare_data.py, scripts/build_dataset.py, and scripts/train.py first.")
