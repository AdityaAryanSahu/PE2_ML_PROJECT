import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Performance Predictor", page_icon="🎓", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }
div[data-testid="stButton"] > button {
    background: #1c1a17; color: #f5f0e8; border: none;
    border-radius: 10px; font-size: 15px; font-weight: 600;
    padding: 14px 0; width: 100%; margin-top: 12px;
}
div[data-testid="stButton"] > button:hover { background: #3a3628; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎓 Performance Predictor")
st.caption("Enter student details to get predictions from all trained models with confidence scores.")
st.divider()

# ── Load artefacts ────────────────────────────────────────────────────────────
MODEL_FILES = {
    "LogisticRegression": "logisticregression.pkl",
    "KNN":                "knn.pkl",
    "DecisionTree":       "decisiontree.pkl",
    "SVM":                "svm.pkl",
    "NaiveBayes":         "naivebayes.pkl",
}
ARTEFACTS = ["sc.pkl", "selector.pkl", "label_encoders.pkl", "feature_names.pkl"]
missing   = [f for f in ARTEFACTS + list(MODEL_FILES.values()) if not os.path.exists(f)]

if missing:
    st.warning(f"Missing file(s): {', '.join(missing)}. Run main.py first to generate all .pkl files.")
    st.stop()

sc            = joblib.load("sc.pkl")
selectors     = joblib.load("selector.pkl")       # dict: {model_name: selector}
le            = joblib.load("label_encoders.pkl")  # single LabelEncoder for final_result
feature_names = joblib.load("feature_names.pkl")  # all columns including final_result

# Input features = all columns except the target
input_features = [c for c in feature_names if c != "final_result"]

models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Student Information")

# Define exact bounds based on the training data to prevent Outlier Panic
# Format: "column_name": (min_value, max_value, default_value)
FEATURE_LIMITS = {
    "study_hours": (1.0, 10.0, 5.0),
    "attendance": (40.0, 100.0, 75.0),
    "sleep_hours": (4.0, 9.0, 7.0),
    "assignments_completed": (0.0, 10.0, 5.0),
    "internet_usage": (1.0, 8.0, 4.0),
    "previous_grade": (30.0, 100.0, 70.0),
    "participation": (1.0, 10.0, 5.0),
    "mock_test_score": (20.0, 100.0, 60.0),
    "class_interaction": (1.0, 10.0, 5.0),
    "extra_classes_attended": (0.0, 10.0, 2.0),
}

user_input = {}
pairs = [input_features[i:i+2] for i in range(0, len(input_features), 2)]

for pair in pairs:
    cols = st.columns(len(pair))
    for widget_col, col_name in zip(cols, pair):
        with widget_col:
            label = col_name.replace("_", " ").title()
            
            # Fetch the specific limits for this feature, or use a safe fallback
            min_val, max_val, default_val = FEATURE_LIMITS.get(col_name, (0.0, 100.0, 0.0))
            
            # Render the number input with the strict limits applied
            user_input[col_name] = st.number_input(
                label, 
                min_value=float(min_val), 
                max_value=float(max_val), 
                value=float(default_val), 
                step=1.0, 
                format="%.2f",
                help=f"Allowed range: {min_val} to {max_val}" # Adds a helpful tooltip for the user!
            )

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict Grade — All Models"):

    # Build input row (all numeric)
    row_dict = {col: float(user_input[col]) for col in input_features}
    row_df     = pd.DataFrame([row_dict], columns=input_features)
    row_scaled = sc.transform(row_df)

    # Run all models
    rows = []
    for model_name, model in models.items():
        row_selected = selectors[model_name].transform(row_scaled)
        pred_enc     = model.predict(row_selected)[0]
        # Decode label back to original class name
        grade        = le.inverse_transform([int(pred_enc)])[0]

        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(row_selected)[0].max()
        elif hasattr(model, "decision_function"):
            df_vals    = model.decision_function(row_selected)[0]
            df_arr     = np.atleast_1d(np.array(df_vals))
            exp        = np.exp(df_arr - df_arr.max())
            confidence = (exp / exp.sum()).max()
        else:
            confidence = None

        rows.append({"Model": model_name, "Grade": grade, "Confidence": confidence})

    rows.sort(key=lambda r: r["Confidence"] if r["Confidence"] is not None else -1, reverse=True)

    # ── Results ───────────────────────────────────────────────────────────────
    st.subheader("Results")

    for rank, r in enumerate(rows, 1):
        is_winner  = rank == 1
        conf       = r["Confidence"]
        conf_label = f"{conf:.1%}" if conf is not None else "N/A"

        with st.container():
            col_rank, col_model, col_grade, col_conf = st.columns([0.5, 2.5, 1, 2])

            with col_rank:
                st.markdown(f"**{rank}**")

            with col_model:
                if is_winner:
                    st.markdown(f"**{r['Model']} 👑**")
                else:
                    st.markdown(r["Model"])

            with col_grade:
                if is_winner:
                    st.markdown(f"### {r['Grade']}")
                else:
                    st.markdown(f"**{r['Grade']}**")

            with col_conf:
                if conf is not None:
                    st.progress(float(conf), text=conf_label)
                else:
                    st.markdown("—")

        if rank < len(rows):
            st.markdown("<hr style='margin:4px 0; border-color:#e8e3d8;'>", unsafe_allow_html=True)

    # ── Agreement summary (Performance-Weighted Soft Voting) ──────────────────
    st.divider()

    ALLOWED_MODELS = ["LogisticRegression", "SVM", "NaiveBayes", "DecisionTree", "KNN"]

    MODEL_WEIGHTS = {
        "LogisticRegression": 0.8,
        "KNN":                0.75,
        "DecisionTree":       0.6,
        "SVM":                0.7,
        "NaiveBayes":         0.75,
    }

    grade_scores = {}
    voting_rows  = [r for r in rows if r["Model"] in ALLOWED_MODELS]

    for r in voting_rows:
        grade          = r["Grade"]
        base_confidence = r["Confidence"] if r["Confidence"] is not None else 1.0
        trust_weight   = MODEL_WEIGHTS.get(r["Model"], 1.0)
        weighted_score = base_confidence * trust_weight
        grade_scores[grade] = grade_scores.get(grade, 0.0) + weighted_score

    final_grade   = max(grade_scores, key=grade_scores.get)
    voting_grades = [r["Grade"] for r in voting_rows]
    unique        = set(voting_grades)

    if len(unique) == 1:
        st.success(f"Final Predicted Grade: **{final_grade}** — All models agree ✅")
    else:
        st.warning(f"Final Predicted Grade: **{final_grade}** — Models disagreed; selected by weighted confidence.")

    # ── Summary dataframe ─────────────────────────────────────────────────────
    with st.expander("View full results table"):
        table_df = pd.DataFrame([
            {
                "Rank":       rank,
                "Model":      r["Model"],
                "Grade":      r["Grade"],
                "Confidence": f"{r['Confidence']:.1%}" if r["Confidence"] is not None else "N/A",
            }
            for rank, r in enumerate(rows, 1)
        ])
        st.dataframe(table_df, use_container_width=True, hide_index=True)