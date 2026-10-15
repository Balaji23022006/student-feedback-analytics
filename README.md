# 🎓 SMART-LEARN: Student Recommendation Dashboard

An AI-powered Streamlit dashboard that analyzes student course feedback and predicts whether a student would **recommend a course**. It uses an **XGBoost** classifier and explains every prediction with **SHAP**, helping instructors identify strengths and areas for improvement in learning quality.

## ✨ Features

- **🤖 XGBoost model** – predicts the likelihood a student recommends the course.
- **🧠 Explainability** – SHAP force plot (single prediction) and SHAP summary plot (global importance).
- **✍️ Single-entry prediction** – enter ratings manually and get a satisfaction score, strong/weak areas and a radar chart.
- **📂 Batch prediction** – upload a CSV, get predictions for all students, view a pie chart and download results as CSV.
- **📈 Model insights** – accuracy, ROC-AUC, precision, recall, F1, confusion matrix and ROC curve.
- **ℹ️ Built-in help page** – user guide inside the app.

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 81.82% |
| ROC-AUC | 0.885 |
| Precision | 0.875 |
| Recall | 0.875 |
| F1 Score | 0.875 |

Dataset: **108 records**, **16 features**, 80/20 stratified train-test split (`random_state=42`).

## 🧩 Input Features

| Group | Features |
|-------|----------|
| 📚 Course Content (1–5) | `cc_rating`, `lab_relavance`, `job_usefulness`, `course_pace`, `material_quality`, `overall_c_satisfaction` |
| 👨‍🏫 Instructor Evaluation (1–5) | `i_clarity`, `i_engagement`, `i_approachability`, `i_eg_effectiveness`, `i_feedback`, `overall_i_satisfaction` |
| 💡 Skill Development (0–2) | `gained_new_skills`, `applied_concepts`, `independent_project_ability`, `confidence` |

Target column: `recommend` (0 = will not recommend, 1 = will recommend).

## 🗂️ Project Structure

```
.
├── app.py                          # Streamlit application
├── xgboost_recommend_model.pkl     # Trained XGBoost model
├── dataset faml project.csv        # Training dataset
├── requirements.txt                # Python dependencies
├── docs/
│   ├── faiml_ppt.pptx              # Project presentation
│   └── ilovepdf_merged.pdf         # Dashboard screenshots
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```
The dashboard opens at `http://localhost:8501`.

## 📂 Batch Upload Format

Your CSV must contain these 16 columns (extra columns are kept in the output):

```
cc_rating, lab_relavance, job_usefulness, course_pace, material_quality,
overall_c_satisfaction, i_clarity, i_engagement, i_approachability,
i_eg_effectiveness, i_feedback, overall_i_satisfaction, gained_new_skills,
applied_concepts, independent_project_ability, confidence
```

The output adds `Recommendation` and `Satisfaction (%)` columns and can be downloaded from the app.

## 🛠️ Tech Stack

Python · Streamlit · XGBoost · scikit-learn · SHAP · Plotly · Matplotlib · Seaborn · Pandas · Joblib

## 👨‍💻 Team

| Name | Role |
|------|------|
| Balaji | Team Leader |
| Hemanth Kumar | Model Developer |
| Narender | Technical Support |
| Guruprasanth | Data Analyst |
| Mohammed Iman | UI Designer |

## 📄 License

This project is for academic purposes. Add a license (e.g. MIT) if you want others to reuse it.
