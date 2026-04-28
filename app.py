import streamlit as st
import pickle
import numpy as np

# Load model & encoders
model = pickle.load(open("model.pkl", "rb"))
le_edu = pickle.load(open("edu_encoder.pkl", "rb"))
le_role = pickle.load(open("role_encoder.pkl", "rb"))

st.title("💼 Employee Salary Prediction App")

# User Inputs
experience = st.slider("Experience (years)", 0, 20)
age = st.slider("Age", 18, 60)

education = st.selectbox("Education Level", le_edu.classes_)
job_role = st.selectbox("Job Role", le_role.classes_)

# Encode input
edu_encoded = le_edu.transform([education])[0]
role_encoded = le_role.transform([job_role])[0]

# Predict
if st.button("Predict Salary"):
    input_data = np.array([[experience, age, edu_encoded, role_encoded]])
    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Salary: ₹{int(prediction[0])}")