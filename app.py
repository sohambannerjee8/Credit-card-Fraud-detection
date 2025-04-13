# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessors
try:
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
except FileNotFoundError:
    st.error("Model or preprocessor files not found. Please run train_model.py first.")
    st.stop()

# Streamlit app
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict if they are likely to churn.")

# Create input form
with st.form("customer_form"):
    st.subheader("Demographic Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    st.subheader("Service Details")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])  # Fixed typo
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.subheader("Billing Information")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=50.0, step=0.1)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0, step=0.1)

    # Submit button
    submitted = st.form_submit_button("Predict Churn")

# Process input and predict
if submitted:
    # Create input dictionary
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Debug: Print columns for verification
    st.write("Input DataFrame columns:", input_df.columns.tolist())
    st.write("Expected feature columns:", feature_columns.tolist())
    st.write("Label Encoder columns:", list(label_encoders.keys()))

    # Check for missing columns
    expected_cols = feature_columns.tolist()
    missing_cols = [col for col in expected_cols if col not in input_df.columns]
    if missing_cols:
        st.error(f"Missing columns in input data: {missing_cols}")
        st.stop()

    # Encode categorical variables
    for col in label_encoders:
        if col not in input_df.columns:
            st.error(f"Column {col} not found in input DataFrame.")
            st.stop()
        try:
            input_df[col] = label_encoders[col].transform(input_df[col])
        except ValueError as e:
            st.error(f"Invalid value for {col}: {e}. Ensure input matches training data options.")
            st.write(f"Valid options for {col}: {label_encoders[col].classes_.tolist()}")
            st.stop()

    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    try:
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    except Exception as e:
        st.error(f"Error scaling numerical features: {e}")
        st.stop()

    # Ensure correct feature order
    try:
        input_df = input_df[feature_columns]
    except KeyError as e:
        st.error(f"Feature order error: {e}. Ensure all expected columns are present.")
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
        st.error(f"Customer is likely to churn with {probability*100:.2f}% probability.")
    else:
        st.success(f"Customer is not likely to churn with {(1-probability)*100:.2f}% confidence.")
