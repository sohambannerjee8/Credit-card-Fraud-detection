# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

def on_predict_button_clicked(b):
    with output:
        output.clear_output()
        
        # Map SeniorCitizen to numeric values
        senior_citizen_numeric = 0 if senior_citizen.value == 'No' else 1

        # Create input dictionary
        input_data = {
            'gender': gender.value,
            'SeniorCitizen': senior_citizen_numeric,
            'Partner': partner.value,
            'Dependents': dependents.value,
            'tenure': tenure.value,
            'PhoneService': phone_service.value,
            'MultipleLines': multiple_lines.value,
            'InternetService': internet_service.value,
            'OnlineSecurity': online_security.value,
            'OnlineBackup': online_backup.value,
            'DeviceProtection': device_protection.value,
            'TechSupport': tech_support.value,
            'StreamingTV': streaming_tv.value,
            'StreamingMovies': streaming_movies.value,
            'Contract': contract.value,
            'PaperlessBilling': paperless_billing.value,
            'PaymentMethod': payment_method.value,
            'MonthlyCharges': monthly_charges.value,
            'TotalCharges': total_charges.value
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Check for missing columns
        expected_cols = feature_columns.tolist()
        missing_cols = [col for col in expected_cols if col not in input_df.columns]
        if missing_cols:
            display(Markdown(f"**Error**: Missing columns: {missing_cols}"))
            return

        # Encode categorical variables
        for col in label_encoders:
            if col not in input_df.columns:
                display(Markdown(f"**Error**: Column {col} not found."))
                return
            try:
                input_df[col] = label_encoders[col].transform(input_df[col])
            except ValueError as e:
                display(Markdown(f"**Error**: Invalid input for {col}: {e}. Valid options: {label_encoders[col].classes_.tolist()}"))
                return

        # Scale numerical features
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        try:
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        except Exception as e:
            display(Markdown(f"**Error**: Processing numerical data: {e}"))
            return

        # Ensure correct feature order
        try:
            input_df = input_df[feature_columns]
        except KeyError as e:
            display(Markdown(f"**Error**: Aligning features: {e}"))
            return

        # Predict
        try:
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0][1]
        except Exception as e:
            display(Markdown(f"**Error**: Prediction failed: {e}"))
            return

        # Display results
        display(Markdown("## Prediction Result"))
        customer_ref = f"Customer {customer_id.value}" if customer_id.value.strip() else "The customer"
        
        # Categorize risk
        if probability >= 0.7:
            risk_level = "High Risk"
            risk_color = "color: #d32f2f; font-weight: bold;"
            recommendations = [
                "Offer a loyalty discount or promotional pricing.",
                "Provide enhanced customer support or a dedicated account manager.",
                "Highlight long-term contract benefits to encourage commitment."
            ]
        elif probability >= 0.4:
            risk_level = "Medium Risk"
            risk_color = "color: #d32f2f; font-weight: bold;"
            recommendations = [
                "Send personalized retention offers, such as a free month of streaming.",
                "Engage with a satisfaction survey to identify pain points.",
                "Promote additional services to increase engagement."
            ]
        else:
            risk_level = "Low Risk"
            risk_color = "color: #388e3c; font-weight: bold;"
            recommendations = [
                "Continue providing excellent service to maintain loyalty.",
                "Invite to a referral program to leverage satisfaction.",
                "Offer minor incentives to reinforce positive experience."
            ]

        # Display prediction
        if prediction[0] == 1:
            display(HTML(f"<p><b>{customer_ref}</b> is <span style='{risk_color}'>{risk_level}</span> of churning with a {probability*100:.1f}% probability.</p>"))
        else:
            display(HTML(f"<p><b>{customer_ref}</b> is <span style='{risk_color}'>{risk_level}</span> of churning with a {(1-probability)*100:.1f}% likelihood of staying.</p>"))

        # Visualization 1: Churn Probability Bar Plot
        display(Markdown("### Churn Probability"))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Churn', 'Stay'], [probability, 1-probability], color=['#d32f2f', '#388e3c'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Churn vs. Retention Probability')
        for i, v in enumerate([probability, 1-probability]):
            ax.text(i, v + 0.02, f'{v*100:.1f}%', ha='center')
        plt.tight_layout()
        plt.show()

        # Show input summary
        display(Markdown("### Customer Details"))
        details = (
            f"- **Customer ID**: {customer_id.value or 'Not provided'}\n"
            f"- **Gender**: {gender.value}\n"
            f"- **Senior Citizen**: {senior_citizen.value}\n"
            f"- **Partner**: {partner.value}\n"
            f"- **Dependents**: {dependents.value}\n"
            f"- **Tenure**: {tenure.value} months\n"
            f"- **Phone Service**: {phone_service.value}\n"
            f"- **Multiple Lines**: {multiple_lines.value}\n"
            f"- **Internet Service**: {internet_service.value}\n"
            f"- **Online Security**: {online_security.value}\n"
            f"- **Online Backup**: {online_backup.value}\n"
            f"- **Device Protection**: {device_protection.value}\n"
            f"- **Tech Support**: {tech_support.value}\n"
            f"- **Streaming TV**: {streaming_tv.value}\n"
            f"- **Streaming Movies**: {streaming_movies.value}\n"
            f"- **Contract**: {contract.value}\n"
            f"- **Paperless Billing**: {paperless_billing.value}\n"
            f"- **Payment Method**: {payment_method.value}\n"
            f"- **Monthly Charges**: ${monthly_charges.value:.2f}\n"
            f"- **Total Charges**: ${total_charges.value:.2f}"
        )
        display(Markdown(details))

        # Retention recommendations
        display(Markdown("### Retention Recommendations"))
        for rec in recommendations:
            display(Markdown(f"- {rec}"))

        # Visualization 2: Feature Importance Bar Plot
        display(Markdown("### Key Factors Influencing Prediction"))
        factors = []
        factor_scores = []
        if tenure.value <= 12:
            factors.append("Short Tenure")
            factor_scores.append(0.4)  # Arbitrary weight for visualization
        if contract.value == "Month-to-month":
            factors.append("Month-to-month Contract")
            factor_scores.append(0.3)
        if monthly_charges.value > 80:
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
        plt.show()

        for factor in factors:
            display(Markdown(f"- {factor}"))

        # Visualization 3: Tenure vs. Charges Scatter Plot
        display(Markdown("### Tenure vs. Monthly Charges"))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(tenure.value, monthly_charges.value, color='#2196f3', s=100, label='Customer')
        ax.axvline(x=12, color='gray', linestyle='--', label='Low Tenure Threshold (12 months)')
        ax.axhline(y=80, color='orange', linestyle='--', label='High Charges Threshold ($80)')
        ax.set_xlabel('Tenure (months)')
        ax.set_ylabel('Monthly Charges ($)')
        ax.set_title('Customer Profile: Tenure vs. Monthly Charges')
        ax.legend()
        plt.tight_layout()
        plt.show()

# Attach the button click event
predict_button.on_click(on_predict_button_clicked)
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

    # Feature importance (simplified, based on typical model output)
    st.subheader("Key Factors Influencing Prediction")
    factors = []
    if tenure <= 12:
        factors.append("Short tenure: Customers with less than a year are at higher risk.")
    if contract == "Month-to-month":
        factors.append("Month-to-month contract: Longer contracts reduce churn risk.")
    if monthly_charges > 80:
        factors.append("High monthly charges: Cost may be a concern.")
    if not factors:
        factors.append("No standout risk factors; maintain current service quality.")
    for factor in factors:
        st.write(f"- {factor}")
