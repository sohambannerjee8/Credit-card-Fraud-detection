import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Set page config for better appearance
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; }
    .stTextInput>div>input { border-radius: 5px; }
    .stSelectbox>div>div>select { border-radius: 5px; }
    .stSlider>div { color: #333; }
    .high-risk { color: #d32f2f; font-weight: bold; }
    .low-risk { color: #388e3c; font-weight: bold; }
    .sidebar .sidebar-content { background-color: #e0f7fa; }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessors
try:
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
except FileNotFoundError:
    st.error("Model or preprocessor files not found. Please ensure model files are available.")
    st.stop()

# Streamlit app
st.title("📞 Telco Customer Churn Prediction")
st.markdown("Enter customer details to assess their likelihood of churning and receive retention recommendations.")

# Sidebar for app info
with st.sidebar:
    st.header("About")
    st.info("""
        This tool predicts whether a Telco customer is likely to churn based on their profile and service usage.
        Enter accurate details to get a reliable prediction and tailored retention strategies.
    """)

# Create input form
with st.form("customer_form"):
    st.header("Customer Profile")

    # Customer ID for realism
    customer_id = st.text_input("Customer ID (optional)", value="CUST-XXXX")

    # Demographic Information
    st.subheader("Demographic Information")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], help="Select the customer's gender.")
        partner = st.selectbox("Partner", ["Yes", "No"], help="Does the customer have a partner?")
    with col2:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], help="Is the customer a senior citizen?")
        dependents = st.selectbox("Dependents", ["Yes", "No"], help="Does the customer have dependents?")

    # Service Details
    st.subheader("Service Details")
    col3, col4 = st.columns(2)
    with col3:
        tenure = st.slider("Tenure (months)", 0, 72, 12, help="How long has the customer been with Telco?")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], help="Does the customer have phone service?")
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], help="Does the customer have multiple phone lines?")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], help="Type of internet service.")
    with col4:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], help="Does the customer have online security?")
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], help="Does the customer have online backup?")
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], help="Does the customer have device protection?")
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], help="Does the customer have tech support?")

    col5, col6 = st.columns(2)
    with col5:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], help="Does the customer have streaming TV?")
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], help="Does the customer have streaming movies?")
    with col6:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], help="Type of contract.")

    # Billing Information
    st.subheader("Billing Information")
    col7, col8 = st.columns(2)
    with col7:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], help="Does the customer use paperless billing?")
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], help="Customer's payment method.")
    with col8:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=50.0, step=0.1, help="Customer's monthly bill.")
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0, step=0.1, help="Total amount billed to the customer.")

    # Submit button
    submitted = st.form_submit_button("Predict Churn", help="Click to calculate churn risk.")

# Process input and predict
if submitted:
    # Map SeniorCitizen to numeric values
    senior_citizen_numeric = 0 if senior_citizen == "No" else 1

    # Create input dictionary
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen_numeric,
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

    # Check for missing columns
    expected_cols = feature_columns.tolist()
    missing_cols = [col for col in expected_cols if col not in input_df.columns]
    if missing_cols:
        st.error(f"Error: Missing columns: {missing_cols}")
        st.stop()

    # Encode categorical variables
    for col in label_encoders:
        if col not in input_df.columns:
            st.error(f"Error: Column {col} not found.")
            st.stop()
        try:
            input_df[col] = label_encoders[col].transform(input_df[col])
        except ValueError as e:
            st.error(f"Invalid input for {col}: {e}. Valid options: {label_encoders[col].classes_.tolist()}")
            st.stop()

    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    try:
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    except Exception as e:
        st.error(f"Error processing numerical data: {e}")
        st.stop()

    # Ensure correct feature order
    try:
        input_df = input_df[feature_columns]
    except KeyError as e:
        st.error(f"Error aligning features: {e}")
        st.stop()

    # Predict
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Display results
    st.header("Prediction Result")
    customer_ref = f"Customer {customer_id}" if customer_id.strip() else "The customer"
    
    # Categorize risk
    if probability >= 0.7:
        risk_level = "High Risk"
        risk_style = "high-risk"
        recommendations = [
            "Offer a loyalty discount or promotional pricing.",
            "Provide enhanced customer support or a dedicated account manager.",
            "Highlight long-term contract benefits to encourage commitment."
        ]
    elif probability >= 0.4:
        risk_level = "Medium Risk"
        risk_style = "high-risk"
        recommendations = [
            "Send personalized retention offers, such as a free month of streaming.",
            "Engage with a satisfaction survey to identify pain points.",
            "Promote additional services to increase engagement."
        ]
    else:
        risk_level = "Low Risk"
        risk_style = "low-risk"
        recommendations = [
            "Continue providing excellent service to maintain loyalty.",
            "Invite to a referral program to leverage satisfaction.",
            "Offer minor incentives to reinforce positive experience."
        ]

    # Display prediction
    if prediction[0] == 1:
        st.markdown(f"**{customer_ref} is <span class='{risk_style}'>{risk_level}</span> of churning with a {probability*100:.1f}% probability.**", unsafe_allow_html=True)
    else:
        st.markdown(f"**{customer_ref} is <span class='{risk_style}'>{risk_level}</span> of churning with a {(1-probability)*100:.1f}% likelihood of staying.**", unsafe_allow_html=True)

    # Visualization 1: Churn Probability Bar Plot
    st.subheader("Churn Probability")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Churn', 'Stay'], [probability, 1-probability], color=['#d32f2f', '#388e3c'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Churn vs. Retention Probability')
    for i, v in enumerate([probability, 1-probability]):
        ax.text(i, v + 0.02, f'{v*100:.1f}%', ha='center')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Show input summary
    with st.expander("View Customer Details"):
        st.write("**Summary of Inputs:**")
        st.write(f"- **Customer ID**: {customer_id or 'Not provided'}")
        st.write(f"- **Gender**: {gender}")
        st.write(f"- **Senior Citizen**: {senior_citizen}")
        st.write(f"- **Partner**: {partner}")
        st.write(f"- **Dependents**: {dependents}")
        st.write(f"- **Tenure**: {tenure} months")
        st.write(f"- **Phone Service**: {phone_service}")
        st.write(f"- **Multiple Lines**: {multiple_lines}")
        st.write(f"- **Internet Service**: {internet_service}")
        st.write(f"- **Online Security**: {online_security}")
        st.write(f"- **Online Backup**: {online_backup}")
        st.write(f"- **Device Protection**: {device_protection}")
        st.write(f"- **Tech Support**: {tech_support}")
        st.write(f"- **Streaming TV**: {streaming_tv}")
        st.write(f"- **Streaming Movies**: {streaming_movies}")
        st.write(f"- **Contract**: {contract}")
        st.write(f"- **Paperless Billing**: {paperless_billing}")
        st.write(f"- **Payment Method**: {payment_method}")
        st.write(f"- **Monthly Charges**: ${monthly_charges:.2f}")
        st.write(f"- **Total Charges**: ${total_charges:.2f}")

    # Retention recommendations
    st.subheader("Retention Recommendations")
    for rec in recommendations:
        st.write(f"- {rec}")

    # Visualization 2: Feature Importance Bar Plot
    st.subheader("Key Factors Influencing Prediction")
    factors = []
    factor_scores = []
    if tenure <= 12:
        factors.append("Short Tenure")
        factor_scores.append(0.4)  # Arbitrary weight for visualization
    if contract == "Month-to-month":
        factors.append("Month-to-month Contract")
        factor_scores.append(0.3)
    if monthly_charges > 80:
        factors.append("High Monthly Charges")
        factor_scores.append(0.2)
    if not factors:
        factors.append("No Major Risks")
        factor_scores.append(0.1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(factors, factor_scores, color='#ff9800')
    ax.set_xlabel('Relative Impact')
    ax.set_title('Key Factors Driving Churn Risk')
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    for factor in factors:
        st.write(f"- {factor}")

    # Visualization 3: Tenure vs. Charges Scatter Plot
    st.subheader("Tenure vs. Monthly Charges")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(tenure, monthly_charges, color='#2196f3', s=100, label='Customer')
    ax.axvline(x=12, color='gray', linestyle='--', label='Low Tenure Threshold (12 months)')
    ax.axhline(y=80, color='orange', linestyle='--', label='High Charges Threshold ($80)')
    ax.set_xlabel('Tenure (months)')
    ax.set_ylabel('Monthly Charges ($)')
    ax.set_title('Customer Profile: Tenure vs. Monthly Charges')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
