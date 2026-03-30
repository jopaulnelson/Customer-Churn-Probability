import streamlit as st

st.title("Customer Churn Risk Predictor")

st.write("Enter customer details:")

age = st.slider("Age", 18, 80)
balance = st.number_input("Balance", value=0.0)
products = st.slider("Number of Products", 1, 4)
active = st.selectbox("Is Active Member", [0, 1])

if st.button("Predict"):
    # Simple demo logic (not ML yet)
    risk = (age * 0.01 + balance * 0.000001 + products * 0.1 + active * 0.2)
    
    st.subheader("Result:")
    st.write("Churn Probability:", round(risk, 2))
    
    if risk > 0.7:
        st.error("High Risk")
    elif risk > 0.3:
        st.warning("Medium Risk")
    else:
        st.success("Low Risk")