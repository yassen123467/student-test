from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import gradio as gr

MODEL_FILE = Path(__file__).resolve().parent / "student_performance_models.joblib"
models = joblib.load(MODEL_FILE)

FEATURES = [
    "Gender", "Parental Education Level", "Lunch Type",
    "Test Preparation Course", "Study Time", "Absences"
]

def predict_scores(gender, parental_education, lunch_type, test_prep, study_time, absences):
    prep_value = "Unknown" if test_prep == "Not recorded" else test_prep
    row = pd.DataFrame([{
        "Gender": gender,
        "Parental Education Level": parental_education,
        "Lunch Type": lunch_type,
        "Test Preparation Course": prep_value,
        "Study Time": float(study_time),
        "Absences": int(absences),
    }], columns=FEATURES)
    return tuple(round(float(np.clip(models[target].predict(row)[0], 0, 100)), 2) for target in [
        "Math Score", "Reading Score", "Writing Score"
    ])

app = gr.Interface(
    fn=predict_scores,
    inputs=[
        gr.Dropdown(["Female", "Male"], value="Female", label="Gender"),
        gr.Dropdown(["High School", "College", "Bachelor", "Master"], value="College", label="Parental Education Level"),
        gr.Dropdown(["Standard", "Free/Reduced"], value="Standard", label="Lunch Type"),
        gr.Dropdown(["Completed", "Not recorded"], value="Not recorded", label="Test Preparation Course"),
        gr.Slider(1, 20, value=10, step=0.5, label="Study Time (hours per week)"),
        gr.Slider(0, 14, value=5, step=1, label="Absences"),
    ],
    outputs=[
        gr.Number(label="Predicted Math Score (0–100)"),
        gr.Number(label="Predicted Reading Score (0–100)"),
        gr.Number(label="Predicted Writing Score (0–100)"),
    ],
    title="Predicting Student Exam Performance",
    description="Enter student attributes to estimate the three exam scores. 'Not recorded' preserves the unknown status of blank preparation values in the training data.",
    examples=[
        ["Female", "High School", "Free/Reduced", "Not recorded", 4, 10],
        ["Male", "Master", "Standard", "Completed", 16, 1],
    ],
)

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
