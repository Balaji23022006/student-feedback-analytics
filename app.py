# =============================================
# 🎓 STUDENT RECOMMENDATION DASHBOARD
#    (XGBoost + SHAP + CSV DOWNLOAD + FEEDBACK INSIGHTS)
# =============================================

import io
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import train_test_split

# -------------------------------------------
# ⚙️ Streamlit Config (must be the first st.* call)
# -------------------------------------------
st.set_page_config(page_title="Student Recommendation Dashboard", layout="wide")

# -------------------------------------------
# 🧠 SHAP Import (Safe) – keeps the real error so it can be shown
# NOTE: shap.initjs() is intentionally NOT called. It is meant for Jupyter
# notebooks (needs IPython) and made the import fail on Streamlit Cloud.
# -------------------------------------------
SHAP_ERROR = None
try:
    import shap
    SHAP_AVAILABLE = True
except Exception as e:
    shap = None
    SHAP_AVAILABLE = False
    SHAP_ERROR = repr(e)

# -------------------------------------------
# ⏳ Loading Animation
# -------------------------------------------
with st.spinner("Loading dashboard..."):
    time.sleep(1)

# -------------------------------------------
# 📦 Load Model and Dataset (robust file lookup)
# -------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PKL_CANDIDATES = ["xgboost_recommend_model.pkl"]
MODEL_JSON_CANDIDATES = ["xgboost_recommend_model.json"]
DATASET_CANDIDATES = ["dataset_faml_project.csv", "dataset faml project.csv"]


def _first_existing(names):
    for n in names:
        p = BASE_DIR / n
        if p.exists():
            return p
    return None


@st.cache_resource
def load_model():
    errors = []

    pkl = _first_existing(MODEL_PKL_CANDIDATES)
    if pkl is not None:
        try:
            return joblib.load(pkl)
        except Exception as e:
            errors.append(f"{pkl.name}: {e!r}")

    js = _first_existing(MODEL_JSON_CANDIDATES)
    if js is not None:
        try:
            from xgboost import XGBClassifier
            m = XGBClassifier()
            m.load_model(str(js))
            return m
        except Exception as e:
            errors.append(f"{js.name}: {e!r}")

    if not errors:
        errors.append("No model file found (expected xgboost_recommend_model.pkl or .json).")
    raise RuntimeError(" | ".join(errors))


@st.cache_data
def load_dataset():
    path = _first_existing(DATASET_CANDIDATES)
    if path is None:
        # last resort: any csv whose name contains "dataset"
        matches = sorted(BASE_DIR.glob("*dataset*.csv"))
        path = matches[0] if matches else None
    if path is None:
        raise FileNotFoundError("Dataset CSV not found next to app.py.")
    return pd.read_csv(path)


try:
    MODEL = load_model()
    df = load_dataset()
except Exception as e:
    st.error(f"❌ Could not load model/dataset: {e}")
    st.stop()

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
# 🌟 SHAP Helpers (multiple fallbacks so it always loads)
# -------------------------------------------
def _align(data):
    """Make sure columns are in the same order the model was trained on."""
    names = getattr(MODEL, "feature_names_in_", None)
    if names is not None:
        return data[list(names)]
    return data


@st.cache_resource(show_spinner=False)
def get_tree_explainer():
    """TreeExplainer if the shap library works with this model, else None."""
    if not SHAP_AVAILABLE:
        return None
    try:
        return shap.TreeExplainer(MODEL)
    except Exception:
        return None


@st.cache_data(show_spinner="Computing SHAP values...")
def compute_shap(data):
    """
    Returns (shap_values [n_rows x n_features], base_value, method_name).
    1) shap.TreeExplainer  2) XGBoost's built-in SHAP (pred_contribs) – needs no shap library.
    """
    data = _align(data)

    explainer = get_tree_explainer()
    if explainer is not None:
        try:
            sv = explainer.shap_values(data, check_additivity=False)
            if isinstance(sv, list):
                sv = sv[-1]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, -1]
            base = float(np.ravel(explainer.expected_value)[-1])
            return sv, base, "shap.TreeExplainer"
        except Exception:
            pass  # fall through to native XGBoost method

    import xgboost as xgb
    booster = MODEL.get_booster()
    contribs = booster.predict(xgb.DMatrix(data), pred_contribs=True, validate_features=False)
    contribs = np.asarray(contribs)
    return contribs[:, :-1], float(contribs[0, -1]), "XGBoost native SHAP"


def render_force_plot(values_row, base_value, feature_row, feature_names):
    """Interactive SHAP force plot rendered as self-contained HTML."""
    plot = shap.force_plot(
        base_value, np.asarray(values_row),
        features=np.asarray(feature_row), feature_names=list(feature_names)
    )
    buf = io.StringIO()
    shap.save_html(buf, plot)  # includes the SHAP JS bundle -> works on Streamlit Cloud
    html = buf.getvalue()
    # components.html is deprecated in newer Streamlit -> use st.iframe when available
    if hasattr(st, "iframe"):
        st.iframe(html, height=320)
    else:
        components.html(html, height=320, scrolling=True)


def contribution_bar(values_row, feature_names, top_n=10):
    """Always-works Plotly bar chart of the top SHAP contributions for one prediction."""
    s = pd.Series(values_row, index=feature_names)
    s = s.reindex(s.abs().sort_values(ascending=False).index).head(top_n)[::-1]
    colors = ["#ff0d57" if v > 0 else "#1e88e5" for v in s.values]
    fig = go.Figure(go.Bar(x=s.values, y=[n.replace("_", " ").title() for n in s.index],
                           orientation="h", marker_color=colors))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="SHAP value (impact on model output)")
    return fig


def importance_bar(shap_values, feature_names):
    """Fallback global-importance chart (mean |SHAP|)."""
    s = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=[n.replace("_", " ").title() for n in s.index],
                           orientation="h", marker_color="#1565c0"))
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Mean |SHAP value|")
    return fig


# -------------------------------------------
# 🧭 Navigation
# -------------------------------------------
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio("Navigate to", ["🏠 Overview", "🧠 Prediction", "📈 Insights", "ℹ️ Help"])

if SHAP_AVAILABLE:
    st.sidebar.success("SHAP library loaded ✅")
else:
    st.sidebar.warning(f"SHAP library unavailable – using XGBoost built-in SHAP.\n\n{SHAP_ERROR}")

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
            pred_proba = MODEL.predict_proba(_align(input_df))[0][1]
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
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=False)
            st.plotly_chart(fig, width="stretch")
            st.caption("📊 This radar chart shows how the student's ratings vary across all feedback features.")
            st.caption("It helps visualize areas of strong satisfaction and those needing improvement.")

            # SHAP Explanation (force plot + always-available contribution chart)
            st.markdown("### 🌟 Why this prediction?")
            try:
                aligned = _align(input_df)
                sv, base_val, method = compute_shap(aligned)
                names = list(aligned.columns)

                if SHAP_AVAILABLE:
                    try:
                        render_force_plot(sv[0], base_val, aligned.iloc[0].values, names)
                        st.caption("💡 SHAP plot shows which features most influenced this prediction (positive or negative).")
                        st.caption("Higher values toward red increase the recommendation likelihood, blue decreases it.")
                    except Exception as fe:
                        st.info(f"Interactive force plot unavailable ({fe}). Showing contribution chart instead.")

                st.plotly_chart(contribution_bar(sv[0], names), width="stretch")
                st.caption(f"📊 Top feature contributions for this student (red = pushes towards *recommend*, blue = pushes against). Method: {method}.")
            except Exception as e:
                st.error(f"Unable to compute SHAP explanation: {e}")

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
                feats = _align(data[FEATURE_COLUMNS])
                preds = MODEL.predict(feats)
                probs = MODEL.predict_proba(feats)[:, 1]
                data["Recommendation"] = ["Will Recommend" if p == 1 else "Will Not Recommend" for p in preds]
                data["Satisfaction (%)"] = (probs * 100).round(2)
                st.dataframe(data)

                counts = data["Recommendation"].value_counts()
                pie = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=0.3)])
                st.plotly_chart(pie, width="stretch")
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
    plt.close(fig)
    st.caption("📘 Confusion matrix shows model classification accuracy by comparing predictions vs actuals.")
    st.caption("Diagonal cells represent correctly predicted samples.")

    st.markdown("### 📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)
    st.caption("📈 ROC curve visualizes model discrimination between classes at various thresholds.")
    st.caption("A higher AUC value indicates a more reliable prediction model.")

    st.markdown("### 🌟 SHAP Summary Plot")
    try:
        X_test_aligned = _align(X_test)
        sv_test, _, method = compute_shap(X_test_aligned)
        names = list(X_test_aligned.columns)

        drawn = False
        if SHAP_AVAILABLE:
            try:
                fig3 = plt.figure(figsize=(8, 6))
                shap.summary_plot(sv_test, X_test_aligned, show=False)
                st.pyplot(plt.gcf())
                plt.close("all")
                drawn = True
            except Exception as pe:
                plt.close("all")
                st.info(f"SHAP beeswarm plot unavailable ({pe}). Showing feature importance chart instead.")

        if not drawn:
            st.plotly_chart(importance_bar(sv_test, names), width="stretch")

        st.caption("🌈 SHAP summary plot highlights which features most influence overall predictions.")
        st.caption("Features with higher SHAP values have stronger impact on student recommendation outcomes.")
    except Exception as e:
        st.error(f"Unable to compute SHAP summary: {e}")

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
