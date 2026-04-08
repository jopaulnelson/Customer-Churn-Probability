import streamlit as st
import numpy as np
import pickle
#import os

# Load model and tools
#model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open("model.pkl", "wb"))
#scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
scaler = pickle.load(open("scaler.pkl", "wb"))
#features_path = os.path.join(os.path.dirname(__file__), "features.pkl")
features = pickle.load(open("features.pkl", "wb"))

st.title("Customer Churn Risk Predictor")

st.write("Enter customer details:")

# Inputs
age = st.slider("Age", 18, 80)
balance = st.number_input("Balance", value=0.0)
credit_score = st.slider("Credit Score", 300, 900)
products = st.slider("Number of Products", 1, 4)
tenure = st.slider("Tenure", 0, 10)
active = st.selectbox("Is Active Member", [0, 1])
card = st.selectbox("Has Credit Card", [0, 1])

# Simple geography & gender inputs
geography = st.selectbox("Geography", ["Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict"):

    # Create input array
    input_data = [credit_score, age, tenure, balance, products, card, active]

    # Add encoded values
    input_dict = dict.fromkeys(features, 0)

    # Fill values
    input_dict['CreditScore'] = credit_score
    input_dict['Age'] = age
    input_dict['Tenure'] = tenure
    input_dict['Balance'] = balance
    input_dict['NumOfProducts'] = products
    input_dict['HasCrCard'] = card
    input_dict['IsActiveMember'] = active

    # Gender encoding
    if 'Gender_Male' in input_dict:
        input_dict['Gender_Male'] = 1 if gender == "Male" else 0

    # Geography encoding
    if geography == "Germany":
        input_dict['Geography_Germany'] = 1
    elif geography == "Spain":
        input_dict['Geography_Spain'] = 1

    # Convert to array
    final_input = np.array(list(input_dict.values())).reshape(1, -1)

    # Scale
    final_input = scaler.transform(final_input)

    # Predict
    prob = model.predict_proba(final_input)[0][1]

    st.subheader("Result:")
    st.write("Churn Probability:", round(prob, 2))

    if prob > 0.7:
        st.error("High Risk")
    elif prob > 0.3:
        st.warning("Medium Risk")
    else:
        st.success("Low Risk")
