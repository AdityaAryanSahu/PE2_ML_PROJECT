import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Grade Predictor", page_icon="🎓", layout="centered")

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
st.title("🎓 Grade Predictor")
st.caption("Enter student details to get predictions from all trained models with confidence scores.")
st.divider()

# ── Load artefacts ────────────────────────────────────────────────────────────
MODEL_FILES = {
    "Logistic Regression":  "logisticregression.pkl",
    "K-Nearest Neighbours": "knn.pkl",
    "Decision Tree":        "decisiontree.pkl",
    "SVM":                  "svm.pkl",
    "Naïve Bayes":          "naivebayes.pkl",
}
ARTEFACTS = ["sc.pkl", "selector.pkl", "label_encoders.pkl", "feature_names.pkl"]
missing   = [f for f in ARTEFACTS + list(MODEL_FILES.values()) if not os.path.exists(f)]

if missing:
    st.warning(f"Missing file(s): {', '.join(missing)}. Run main.py first to generate all .pkl files.")
    st.stop()

sc             = joblib.load("sc.pkl")
selector       = joblib.load("selector.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_names  = joblib.load("feature_names.pkl")

grade_encoder    = label_encoders.get("Grade")
input_features   = [c for c in feature_names if c != "Grade"]

# Force Age and Study Hours to be treated as numerical even if they are in the old label_encoders
categorical_cols = [c for c in label_encoders if c not in ["Grade", "Student_Age", "Weekly_Study_Hours"]]
models           = {name: joblib.load(path) for name, path in MODEL_FILES.items()}

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Student Information")

user_input = {}
pairs = [input_features[i:i+2] for i in range(0, len(input_features), 2)]

for pair in pairs:
    cols = st.columns(len(pair))
    for widget_col, col_name in zip(cols, pair):
        with widget_col:
            label = col_name.replace("_", " ").title()
            
            # Explicitly handle Age and Study Hours as numeric inputs
            if col_name == "Student_Age":
                user_input[col_name] = st.number_input(label, min_value=10.0, max_value=100.0, value=20.0, step=1.0)
            elif col_name == "Weekly_Study_Hours":
                user_input[col_name] = st.number_input(label, min_value=0.0, max_value=168.0, value=10.0, step=1.0)
            elif col_name in categorical_cols:
                classes = list(label_encoders[col_name].classes_)
                user_input[col_name] = st.selectbox(label, classes)
            else:
                user_input[col_name] = st.number_input(label, value=0.0, step=1.0, format="%.2f")

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict Grade — All Models"):

    # Encode input
    row_dict = {}
    for col_name in input_features:
        val = user_input[col_name]
        if col_name in categorical_cols:
            row_dict[col_name] = label_encoders[col_name].transform([val])[0]
        else:
            row_dict[col_name] = float(val)

    row_df       = pd.DataFrame([row_dict], columns=input_features)
    row_scaled   = sc.transform(row_df)
    row_selected = selector.transform(row_scaled)

    # Run all models
    rows = []
    for model_name, model in models.items():
        pred_enc = model.predict(row_selected)[0]
        grade    = grade_encoder.inverse_transform([int(pred_enc)])[0] if grade_encoder else str(pred_enc)

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
        crown      = " 👑" if is_winner else f"  #{rank}"

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

    # ── Agreement summary ─────────────────────────────────────────────────────
    st.divider()
    all_grades = [r["Grade"] for r in rows]
    unique     = set(all_grades)

    if len(unique) == 1:
        st.success(f"Final Predicted Grade: **{all_grades[0]}**.")
    else:
        majority = max(unique, key=all_grades.count)
        count    = all_grades.count(majority)
        st.warning(f"Final Predicted Grade: **{majority}**.")

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