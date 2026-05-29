import streamlit as st
import pandas as pd
import pickle

# --------------------------
# Page Config (FIRST)
# --------------------------
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📞",
    layout="wide"
)

# --------------------------
# Load Models
# --------------------------
churn_model = pickle.load(open("churn_model.pkl", "rb"))
rev_model = pickle.load(open("revenue_model.pkl", "rb"))
scaler = pickle.load(open("churn_scaler.pkl", "rb"))
columns = pickle.load(open("churn_columns.pkl", "rb"))
cluster_model = pickle.load(open("cluster_model.pkl", "rb"))

# --------------------------
# Title
# --------------------------
st.title("📞 AI-Powered Telecom Customer Churn Prediction Platform")

st.write(
    """
Predict:

✅ Churn Probability

✅ Expected Monthly Revenue Loss

✅ Customer Segment

✅ Retention Recommendation
"""
)

# --------------------------
# Sidebar
# --------------------------
st.sidebar.header("Customer Information")

tenure = st.sidebar.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

contract = st.sidebar.selectbox(
    "Contract Type",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# --------------------------
# Recommendation Function
# --------------------------
def retention_recommendation(prob):

    if prob >= 80:
        return "🔥 Offer 30% discount and premium support"

    elif prob >= 50:
        return "🎁 Offer loyalty rewards and retention plan"

    else:
        return "✅ Maintain regular customer engagement"

# --------------------------
# Prediction Button
# --------------------------
if st.button("Predict Customer Risk"):

    input_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_data[f"Contract_{contract}"] = 1
    input_data[f"InternetService_{internet_service}"] = 1
    input_data[f"PaymentMethod_{payment_method}"] = 1

    df = pd.DataFrame([input_data])

    # Match training columns
    df = df.reindex(
        columns=columns,
        fill_value=0
    )

    scaled = scaler.transform(df)

    churn_probability = (
        churn_model.predict_proba(scaled)[0][1]
        * 100
    )

    revenue_loss = rev_model.predict(df)[0]

    cluster = cluster_model.predict(df)[0]

    recommendation = retention_recommendation(
        churn_probability
    )

    # --------------------------
    # Results
    # --------------------------

    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )

    with col2:
        st.metric(
            "Revenue Loss",
            f"${revenue_loss:.2f}"
        )

    with col3:
        st.metric(
            "Customer Cluster",
            int(cluster)
        )

    if churn_probability >= 80:
        st.error("⚠️ High Churn Risk")

    elif churn_probability >= 50:
        st.warning("⚠️ Medium Churn Risk")

    else:
        st.success("✅ Low Churn Risk")

    st.subheader("Retention Recommendation")

    st.success(recommendation)

    st.subheader("API Output")

    st.json({
        "churn_probability":
        round(churn_probability, 2),

        "expected_monthly_revenue_loss":
        round(float(revenue_loss), 2),

        "customer_cluster":
        int(cluster),

        "recommendation":
        recommendation
    })