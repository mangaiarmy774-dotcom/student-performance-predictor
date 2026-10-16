import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Predictor")
st.caption("Decision Tree Classification · UCI Student Performance dataset (Maths)")

model = joblib.load("model.joblib")
metrics = json.load(open("metrics.json"))

st.subheader("Enter student details")
c1, c2 = st.columns(2)
with c1:
    studytime = st.selectbox("Weekly study time", [1, 2, 3, 4],
        format_func=lambda v: {1: "< 2 hours", 2: "2 to 5 hours", 3: "5 to 10 hours", 4: "> 10 hours"}[v], index=1)
    failures = st.number_input("Past class failures", 0, 3, 0)
    absences = st.number_input("Number of absences", 0, 93, 4)
with c2:
    g1 = st.slider("First period grade (G1)", 0, 20, 10)
    g2 = st.slider("Second period grade (G2)", 0, 20, 10)

if st.button("Predict", type="primary"):
    row = pd.DataFrame([[studytime, failures, absences, g1, g2]], columns=metrics["features"])
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][pred]
    if pred == 1:
        st.success(f"✅ Likely to Pass  (model confidence {prob:.0%})")
    else:
        st.warning(f"⚠️ Needs Support  (model confidence {prob:.0%})")
    st.caption("This is a model estimate, not a guarantee of a student's future performance.")

with st.expander("Model details"):
    st.write(f"Test accuracy: **{metrics['accuracy']:.1%}** "
             f"(trained on {metrics['train_rows']} students, tested on {metrics['test_rows']})")
    st.write("Inputs used:", ", ".join(metrics["features"]))
    st.write("Label: Likely to Pass means final grade (G3) is 10 or more out of 20.")
