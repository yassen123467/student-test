# Predicting Student Exam Performance — AMHM-ITAI-01

This project contains the completed notebook, a CSV converted from the supplied macro-enabled workbook, trained models, a local Gradio app, chart assets, and two tested UI screenshots.

## Quick start in Google Colab
1. Upload/open `Student_Exam_Performance_Assessment.ipynb`.
2. Upload `student_performance_balanced_mission1.csv` if the notebook asks for it.
3. Run the notebook cells from top to bottom. The project zip includes the pre-rendered chart assets and stored outputs.

## Run the Gradio app locally
Keep `app.py` and `student_performance_models.joblib` in the same folder, then:

```bash
pip install -r requirements.txt
python app.py
```

Open the local address printed by Gradio (default `http://127.0.0.1:7860`). The app returns estimates for Math, Reading and Writing, rounded to two decimals and bounded to the observed exam scale 0–100.

## Data and evaluation notes
- 1,000 rows, 9 columns, no duplicate rows.
- 514 values are blank in `Test Preparation Course`; the model treats them as `Unknown`, not as confirmed non-completion.
- Scores are continuous; use MAE, RMSE and R² rather than classification accuracy.
- 80/20 train/test split (`random_state=42`); 5-fold GridSearchCV is applied only to the training partition.
- Selected models: Ridge for Math and Reading; Linear Regression for Writing (essentially tied with Ridge in cross-validation).
- Two interface examples were tested; screenshots are in `screenshots/` and embedded in the notebook.
- Associations and permutation importance are descriptive and not causal; this model should support, not replace, educator judgment.

## Project files
- `Student_Exam_Performance_Assessment.ipynb` — annotated notebook and assessment report.
- `student_performance_balanced_mission1.csv` — source data converted from the supplied XLSM workbook.
- `app.py` — Gradio predictor.
- `student_performance_models.joblib` — trained models used by the app.
- `charts/` — pre-rendered interactive visualizations used in the notebook.
- `screenshots/` — two locally tested Gradio results.
- `model_comparison.csv`, `feature_importance.csv` — concise result tables.
