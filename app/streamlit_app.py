"""
House Price Estimator — Streamlit app.

This is a placeholder for now (Day 1 of the project).
It will be built out in full on Day 5, once we have:
  - a cleaned/prepared dataset (Day 1)
  - a trained + validated model (Day 3-4)
  - a saved model artifact in models/ (joblib/pickle)

Running it now just confirms the container/infra works end to end.
"""

import streamlit as st

st.set_page_config(page_title="House Price Estimator", page_icon="🏠", layout="centered")

st.title("🏠 House Price Estimator")
st.info(
    "Infrastructure is up and running. \n\n"
    "The prediction form will be built on Day 5 once the model is trained "
    "and saved to `models/`."
)
