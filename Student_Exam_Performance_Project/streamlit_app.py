from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Exam Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

MODEL_FILE = Path(__file__).resolve().parent / "student_performance_models.joblib"

@st.cache_resource
def load_models():
    return joblib.load(MODEL_FILE)

models = load_models()

FEATURES = [
    "Gender", "Parental Education Level", "Lunch Type",
    "Test Preparation Course", "Study Time", "Absences"
]

st.title("🎓 Student Exam Performance Predictor")
st.write("Enter student attributes below to estimate Math, Reading, and Writing exam scores.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"], index=0)
        parental_education = st.selectbox(
            "Parental Education Level",
            ["High School", "College", "Bachelor", "Master"],
            index=1
        )
        lunch_type = st.selectbox("Lunch Type", ["Standard", "Free/Reduced"], index=0)

    with col2:
        test_prep = st.selectbox(
            "Test Preparation Course",
            ["Not recorded", "Completed"],
            index=0,
            help="'Not recorded' preserves the unknown status of blank preparation values in training data."
        )
        study_time = st.slider("Study Time (hours per week)", min_value=1.0, max_value=20.0, value=10.0, step=0.5)
        absences = st.slider("Absences (days)", min_value=0, max_value=14, value=5, step=1)

    submitted = st.form_submit_button("Predict Scores", type="primary", use_container_width=True)

if submitted:
    prep_value = "Unknown" if test_prep == "Not recorded" else test_prep
    row = pd.DataFrame([{
        "Gender": gender,
        "Parental Education Level": parental_education,
        "Lunch Type": lunch_type,
        "Test Preparation Course": prep_value,
        "Study Time": float(study_time),
        "Absences": int(absences),
    }], columns=FEATURES)

    math_score = round(float(np.clip(models["Math Score"].predict(row)[0], 0, 100)), 2)
    reading_score = round(float(np.clip(models["Reading Score"].predict(row)[0], 0, 100)), 2)
    writing_score = round(float(np.clip(models["Writing Score"].predict(row)[0], 0, 100)), 2)

    st.subheader("📊 Predicted Scores")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Math Score", f"{math_score} / 100")
    m_col2.metric("Reading Score", f"{reading_score} / 100")
    m_col3.metric("Writing Score", f"{writing_score} / 100")

    avg_score = round((math_score + reading_score + writing_score) / 3, 2)
    st.info(f"**Average Expected Score:** {avg_score} / 100")
