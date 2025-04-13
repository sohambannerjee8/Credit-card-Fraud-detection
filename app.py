# app_fraud.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessors
try:
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
except FileNotFoundError:
    st.error("Model or scaler files not found. Please run train_model_fraud.py first.")
    st.stop()

# Streamlit app
st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to predict if it's fraudulent.")

# Create input form
with st.form("transaction_form"):
    st.subheader("Transaction Details")
    time = st.number_input("Time (seconds)", min_value=0.0, value=0.0)
    amount = st.number_input("Amount ($)", min_value=0.0, value=0.0)
    v_inputs = {}
    for i in range(1, 29):
        v_inputs[f'V{i}'] = st.number_input(f'V{i}', value=0.0)

    # Submit button
    submitted = st.form_submit_button("Predict Fraud")

# Process input and predict
if submitted:
    # Create input dictionary
    input_data = {'Time': time, 'Amount': amount}
    input_data.update(v_inputs)

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Debug: Verify columns
    expected_cols = feature_columns.tolist()
    missing_cols = [col for col in expected_cols if col not in input_df.columns]
    if missing_cols:
        st.error(f"Missing columns in input data: {missing_cols}")
        st.stop()

    # Scale features
    try:
        input_df = scaler.transform(input_df[feature_columns])
    except Exception as e:
        st.error(f"Error scaling features: {e}")
        st.stop()

    # Predict
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # Display results
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"Transaction is likely fraudulent with {probability*100:.2f}% probability.")
    else:
        st.success(f"Transaction is not likely fraudulent with {(1-probability)*100:.2f}% confidence.")
