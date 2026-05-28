import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered"
)

# ================= LOAD FILES =================

model = pickle.load(open("model.pkl", "rb"))
le_edu = pickle.load(open("edu_encoder.pkl", "rb"))
le_role = pickle.load(open("role_encoder.pkl", "rb"))

# ================= SIDEBAR =================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Data Analysis",
        "Visualizations",
        "Model Info",
        "Data Cleaning",
        "About Project"
    ]
)

# ================= HOME PAGE =================

if page == "Home":

    st.title("💼 Employee Salary Prediction App")

    st.markdown("""
    This Machine Learning project predicts employee salary based on:

    - Experience
    - Age
    - Education Level
    - Job Role
    """)

    st.markdown("---")

    # User Inputs
    experience = st.slider(
        "Experience (Years)",
        0,
        20
    )

    age = st.slider(
        "Age",
        18,
        60
    )

    education = st.selectbox(
        "Education Level",
        le_edu.classes_
    )

    job_role = st.selectbox(
        "Job Role",
        le_role.classes_
    )

    # Encode Input
    edu_encoded = le_edu.transform([education])[0]
    role_encoded = le_role.transform([job_role])[0]

    # Prediction
    if st.button("Predict Salary"):

        input_data = np.array([
            [experience, age, edu_encoded, role_encoded]
        ])

        prediction = model.predict(input_data)

        salary = int(prediction[0])

        st.success(f"💰 Predicted Salary: ₹{salary}")

        # Salary Category
        if salary < 30000:
            st.warning("📉 Low Salary")

        elif salary < 70000:
            st.info("📊 Medium Salary")

        else:
            st.success("🚀 High Salary")

        st.balloons()

# ================= DATA ANALYSIS =================

elif page == "Data Analysis":

    st.title("📊 Data Analysis")

    df = pd.read_csv("employee_data.csv")

    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Columns")
    st.write(df.columns)

    st.subheader("Summary Statistics")
    st.write(df.describe())

# ================= VISUALIZATIONS =================

elif page == "Visualizations":

    st.title("📈 Data Visualizations")

    df = pd.read_csv("employee_data.csv")

    # Scatter Plot
    st.subheader("Experience vs Salary")

    fig, ax = plt.subplots()

    ax.scatter(df["Experience"], df["Salary"])

    ax.set_xlabel("Experience")
    ax.set_ylabel("Salary")

    st.pyplot(fig)

    # Bar Chart
    st.subheader("Average Salary by Job Role")

    avg_salary = df.groupby("Job_Role")["Salary"].mean()

    st.bar_chart(avg_salary)

# ================= MODEL INFO =================

elif page == "Model Info":

    st.title("🤖 Model Information")

    st.write("""
    ### Machine Learning Model Used
    - Linear Regression

    ### Features Used
    - Experience
    - Age
    - Education Level
    - Job Role

    ### Target Variable
    - Salary

    ### Libraries Used
    - Pandas
    - NumPy
    - Scikit-learn
    - Streamlit
    """)

# ================= DATA CLEANING =================

elif page == "Data Cleaning":

    st.title("🧹 Data Cleaning")

    st.write("""
    ### Data Cleaning Steps Performed

    1. Removed missing values

    2. Removed duplicate records

    3. Encoded categorical columns:
       - Education Level
       - Job Role

    4. Converted categorical data into numerical format

    5. Prepared clean dataset for Machine Learning model
    """)

# ================= ABOUT PROJECT =================

elif page == "About Project":

    st.title("ℹ️ About Project")

    st.write("""
    ## Employee Salary Prediction App

    This project predicts employee salary using Machine Learning.

    ### Developed By
    Shivanshi

    ### Technologies Used
    - Python
    - Streamlit
    - Machine Learning
    - Scikit-learn
    - Pandas
    """)

    st.markdown("---")

    st.subheader("📞 Contact Information")

    st.write("👩 Name: Shivanshi")

    st.write("📧 Email: shivanshichandel9@gmail.com")
