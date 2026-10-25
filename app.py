import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('model.sav')

st.title("Advertising Sales Prediction App")
st.write("This app predicts sales revenue based on advertising budgets across TV, Radio, and Newspaper channels.")

# Input fields for advertising budgets
tv_budget = st.number_input("TV Budget ($)", min_value=0.0, step=1.0, value=100.0)
radio_budget = st.number_input("Radio Budget ($)", min_value=0.0, step=1.0, value=20.0)
newspaper_budget = st.number_input("Newspaper Budget ($)", min_value=0.0, step=1.0, value=30.0)

# Prediction button
if st.button("Predict Sales"):
    # Reshape input to match model structure: [[TV, Radio, Newspaper]]
    features = np.array([[tv_budget, radio_budget, newspaper_budget]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Sales Revenue: ${prediction:.2f}k")
