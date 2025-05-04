import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.models import load_model

# Load model and preprocessing tools
model = load_model('model_NN.h5')
with open('scaler_NN.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('encoder_NN.pkl', 'rb') as f:
    encoder = pickle.load(f)


# Define categorical and numeric columns
cat_var = ["person_gender", "person_education", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]
numeric_features = ["loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score"]

# Streamlit UI
st.title("Loan Default Prediction App")
st.markdown("Please fill in the information below to predict whether a loan will default.")

# Input fields
user_input = {}
for col in numeric_features:
    user_input[col] = st.number_input(f"Enter {col.replace('_', ' ').title()}", value=0.0)

for col in cat_var:
    options = encoder.categories_[cat_var.index(col)].tolist()
    user_input[col] = st.selectbox(f"Select {col.replace('_', ' ').title()}", options)

if st.button("Predict"):
    # Prepare input
    num_df = pd.DataFrame([user_input])[numeric_features]
    num_scaled = scaler.transform(num_df)

    cat_df = pd.DataFrame([user_input])[cat_var]
    cat_encoded = encoder.transform(cat_df)

    final_input = np.concatenate([cat_encoded, num_scaled], axis=1)
    prediction = model.predict(final_input)[0][0]

    st.subheader(f"Default Probability: {prediction:.2%}")
    if prediction > 0.5:
        st.error("⚠️ Likely to Default")
    else:
        st.success("✅ Unlikely to Default")
