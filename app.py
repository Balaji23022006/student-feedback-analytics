# =============================================
# 🎓 STUDENT RECOMMENDATION DASHBOARD (XGBoost + SHAP + CSV DOWNLOAD + FEEDBACK INSIGHTS)
# =============================================

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import time
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import train_test_split

# -------------------------------------------
# 🧠 SHAP Import (Safe)
# -------------------------------------------
try:
    import shap
    SHAP_AVAILABLE = True
    shap.initjs()
except Exception:
    SHAP_AVAILABLE = False

# -------------------------------------------
# ⚙️ Streamlit Config
# -------------------------------------------
st.set_page_config(page_title="Student Recommendation Dashboard", layout="wide")

# -------------------------------------------
# ⏳ Loading Animation
# -------------------------------------------
with st.spinner("Loading dashboard..."):
    time.sleep(1)

# -------------------------------------------
# 📦 Load Model and Dataset
# -------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgboost_recommend_model.pkl")

@st.cache_data
def load_dataset():
    return pd.read_csv("dataset faml project.csv")

MODEL = load_model()
df = load_dataset()

X = df.drop("recommend", axis=1)
y = df["recommend"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = MODEL.predict(X_test)
y_proba = MODEL.predict_proba(X_test)[:, 1]

# -------------------------------------------
# 🎨 CSS Styling
# -------------------------------------------
st.markdown("""
<style>
@keyframes fadeIn {
  from {opacity: 0; transform: translateY(10px);}
  to {opacity: 1; transform: translateY(0);}
}
body {font-family: "Inter", sans-serif;}
.hero {
  background: linear-gradient(135deg, #1565c0, #42a5f5);
  color: white;
  border-radius: 12px;
  text-align: center;
  padding: 45px 25px;
  margin-bottom: 30px;
  animation: fadeIn 1s ease-in-out;
}
.hero h1 {font-size: 42px; margin-bottom: 12px;}
.hero p {font-size: 18px; opacity: 0.95;}
.card {
  background: white; border-radius: 12px; padding: 18px;
  margin: 10px; text-align: center;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
  transition: transform 0.3s ease;
}
.card:hover {transform: translateY(-5px);}
.card h4 {margin-bottom: 6px; color: #1565c0;}
.card p {font-size: 14px; color: #444;}
.status-success {
  background: #e8f5e9; color: #2e7d32; padding: 12px;
  border-radius: 8px; font-weight: 600; margin-bottom: 15px;
  animation: fadeIn 1s ease-in-out;
}
.status-error {
  background: #ffebee; color: #c62828; padding: 12px;
  border-radius: 8px; font-weight: 600; margin-bottom: 15px;
  animation: fadeIn 1s ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------
# 🧩 Feature Setup
# -------------------------------------------
FEATURE_COLUMNS = [
    "cc_rating", "lab_relavance", "job_usefulness", "course_pace", "material_quality",
    "overall_c_satisfaction", "i_clarity", "i_engagement", "i_approachability",
    "i_eg_effectiveness", "i_feedback", "overall_i_satisfaction", "gained_new_skills",
    "applied_concepts", "independent_project_ability", "confidence"
]

FEATURE_SCALES = {
    "independent_project_ability": (0, 2),
    "confidence": (0, 2),
    "gained_new_skills": (0, 2),
    "applied_concepts": (0, 2),
}
def get_scale(feature):
    return FEATURE_SCALES.get(feature, (1, 5))

# -------------------------------------------
# 🧭 Navigation
# -------------------------------------------
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio("Navigate to", ["🏠 Overview", "🧠 Prediction", "📈 Insights", "ℹ️ Help"])

# -------------------------------------------
# 🏠 Overview Page
# -------------------------------------------
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero">
        <h1>🎓 SMART-LEARN: An AI-Powered Student Feedback Analytics and Recommendation Dashboard</h1>
        <p>This project analyzes student feedback using AI to predict whether a student would recommend a course, helping identify strengths and areas for improvement in learning quality.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🌟 Key Highlights")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="card">🤖 <h4>XGBoost Model</h4><p>Accurate and efficient student recommendation prediction.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">🧠 <h4>Explainability</h4><p>Understand predictions using SHAP feature importance.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card">📊 <h4>Interactive Insights</h4><p>Visualize performance with charts and metrics.</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="card">📂 <h4>Batch Predictions</h4><p>Upload CSVs to predict multiple students at once.</p></div>', unsafe_allow_html=True)

    st.markdown("### 👨‍💻 Project Team")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h4>Balaji</h4><p>Team Leader</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><h4>Guruprasanth</h4><p>Data Analyst</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h4>Hemanth Kumar</h4><p>Model Developer</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><h4>Mohammed Iman</h4><p>UI Designer</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h4>Narender</h4><p>Technical Support</p></div>', unsafe_allow_html=True)

    st.markdown("### 📈 Project Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("📦 Dataset Size", len(df))
    c2.metric("🎯 Model Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    c3.metric("💡 Features Used", len(FEATURE_COLUMNS))

# -------------------------------------------
# 🧠 Prediction Page
# -------------------------------------------
elif page == "🧠 Prediction":
    st.title("🧠 Prediction Center")

    tabs = st.tabs(["✍️ Single Entry", "📂 Batch Upload"])

    # --- Single Entry
    with tabs[0]:
        st.subheader("Single Student Prediction")

        user_input = {}
        with st.expander("📚 Course Content", expanded=True):
            for f in ["cc_rating", "lab_relavance", "job_usefulness", "course_pace", "material_quality", "overall_c_satisfaction"]:
                lo, hi = get_scale(f)
                user_input[f] = st.number_input(f.replace("_", " ").title(), min_value=lo, max_value=hi, value=3, step=1)

        with st.expander("👨‍🏫 Instructor Evaluation"):
            for f in ["i_clarity", "i_engagement", "i_approachability", "i_eg_effectiveness", "i_feedback", "overall_i_satisfaction"]:
                lo, hi = get_scale(f)
                user_input[f] = st.number_input(f.replace("_", " ").title(), min_value=lo, max_value=hi, value=3, step=1)

        with st.expander("💡 Skill Development"):
            for f in ["gained_new_skills", "applied_concepts", "independent_project_ability", "confidence"]:
                lo, hi = get_scale(f)
                user_input[f] = st.number_input(f.replace("_", " ").title(), min_value=lo, max_value=hi, value=1, step=1)

        if st.button("🔮 Predict"):
            input_df = pd.DataFrame([user_input])
            pred_proba = MODEL.predict_proba(input_df)[0][1]
            pred = 1 if pred_proba >= 0.5 else 0

            # Display main prediction
            if pred == 1:
                st.markdown(f'<div class="status-success">✅ The student is likely to recommend this course.<br>💠 Satisfaction Level: {pred_proba*100:.2f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-error">⚠️ The student is unlikely to recommend this course.<br>💠 Satisfaction Level: {pred_proba*100:.2f}%</div>', unsafe_allow_html=True)

            # --- Intelligent Insight Section ---
            strong_areas = [f for f, v in user_input.items() if v >= 4]
            weak_areas = [f for f, v in user_input.items() if v <= 2]

            if strong_areas:
                st.success("🌟 **Strong Areas:** " + ", ".join([s.replace('_', ' ').title() for s in strong_areas]))
            if weak_areas:
                st.warning("⚙️ **Needs Improvement:** " + ", ".join([w.replace('_', ' ').title() for w in weak_areas]))

            # Radar Chart
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=input_df.iloc[0].values, theta=input_df.columns, fill='toself'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,5])), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("📊 This radar chart shows how the student's ratings vary across all feedback features.")
            st.caption("It helps visualize areas of strong satisfaction and those needing improvement.")

            # SHAP Force Plot
            if SHAP_AVAILABLE:
                try:
                    explainer = shap.Explainer(MODEL, X_train)
                    shap_values = explainer(input_df)
                    exp = explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value
                    force_plot = shap.force_plot(exp, shap_values.values[0], input_df)
                    components.html(f"<head>{shap.getjs()}</head>{force_plot.html()}", height=350)
                    st.caption("💡 SHAP plot shows which features most influenced this prediction (positive or negative).")
                    st.caption("Higher values toward red increase the recommendation likelihood, blue decreases it.")
                except Exception:
                    st.warning("Unable to load SHAP force plot.")
            else:
                st.warning("SHAP not installed — skipping explainability plots.")

    # --- Batch Upload
    with tabs[1]:
        st.subheader("Batch Upload Prediction")
        uploaded_file = st.file_uploader("Upload CSV file with feedback data", type=["csv"])
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.dataframe(data.head(5))
            missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                preds = MODEL.predict(data[FEATURE_COLUMNS])
                probs = MODEL.predict_proba(data[FEATURE_COLUMNS])[:, 1]
                data["Recommendation"] = ["Will Recommend" if p == 1 else "Will Not Recommend" for p in preds]
                data["Satisfaction (%)"] = (probs*100).round(2)
                st.dataframe(data)

                pie = go.Figure(data=[go.Pie(labels=data["Recommendation"].value_counts().index,
                                             values=data["Recommendation"].value_counts().values, hole=0.3)])
                st.plotly_chart(pie)
                st.caption("📊 This pie chart shows the distribution of students likely/unlikely to recommend.")
                st.caption("A higher portion of 'Will Recommend' suggests positive course reception.")

                # CSV Download Button
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="student_recommendations.csv",
                    mime="text/csv"
                )

# -------------------------------------------
# 📈 Insights Page
# -------------------------------------------
elif page == "📈 Insights":
    st.title("📈 Model Insights")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    st.markdown("### ⚙️ Performance Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{acc*100:.2f}%")
    c2.metric("ROC–AUC", f"{roc_auc:.3f}")
    c3.metric("Precision", f"{prec:.3f}")
    c4.metric("Recall", f"{rec:.3f}")
    c5.metric("F1 Score", f"{f1:.3f}")

    st.markdown("### 📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    st.pyplot(fig)
    st.caption("📘 Confusion matrix shows model classification accuracy by comparing predictions vs actuals.")
    st.caption("Diagonal cells represent correctly predicted samples.")

    st.markdown("### 📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax2.legend()
    st.pyplot(fig2)
    st.caption("📈 ROC curve visualizes model discrimination between classes at various thresholds.")
    st.caption("A higher AUC value indicates a more reliable prediction model.")

    if SHAP_AVAILABLE:
        st.markdown("### 🌟 SHAP Summary Plot")
        explainer = shap.Explainer(MODEL, X_train)
        shap_values = explainer(X_test)
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig3)
        st.caption("🌈 SHAP summary plot highlights which features most influence overall predictions.")
        st.caption("Features with higher SHAP values have stronger impact on student recommendation outcomes.")
    else:
        st.warning("SHAP not installed — summary plot unavailable.")

# -------------------------------------------
# ℹ️ Help Page
# -------------------------------------------
elif page == "ℹ️ Help":
    st.title("ℹ️ Help & User Guide")

    st.markdown("""
    ### 🧭 How to Use This Dashboard

    **Step 1 – Overview**
    - Visit the *Overview* page to understand the project goal and explore dataset and model statistics.

    **Step 2 – Prediction**
    - *Single Entry:* Manually input student feedback to predict course recommendation likelihood.  
    - *Batch Upload:* Upload a CSV file containing multiple feedback records and download the predicted results.

    **Step 3 – Insights**
    - Explore model performance, confusion matrix, ROC curve, and SHAP explainability visualizations.

    **Step 4 – Interpretation**
    - Use this dashboard to identify factors influencing course satisfaction and overall learning experience.

    ### 💬 Tips
    - Ensure feedback scores are within correct ranges (0–2 or 1–5).  
    - SHAP explainability plots may take extra time to render.  
    - Use the CSV download feature to save and analyze batch prediction results.
    """)
