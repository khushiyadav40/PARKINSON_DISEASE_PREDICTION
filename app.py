import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load("models/parkinsons_model.pkl")
scaler = joblib.load("models/scaler.pkl")
df = pd.read_csv("data/parkinsons.csv")

FEATURE_GROUPS = {
    "Voice Frequency Features": [
        "MDVP:Fo(Hz)",
        "MDVP:Fhi(Hz)",
        "MDVP:Flo(Hz)"
    ],
    "Jitter Features": [
        "MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)",
        "MDVP:RAP",
        "MDVP:PPQ",
        "Jitter:DDP"
    ],
    "Shimmer Features": [
        "MDVP:Shimmer",
        "MDVP:Shimmer(dB)",
        "Shimmer:APQ3",
        "Shimmer:APQ5",
        "MDVP:APQ",
        "Shimmer:DDA"
    ],
    "Voice Quality Features": [
        "NHR",
        "HNR"
    ],
    "Nonlinear Dynamic Features": [
        "RPDE",
        "DFA",
        "spread1",
        "spread2",
        "D2",
        "PPE"
    ]
}
st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    layout="centered"
)
if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if not st.session_state.show_result:
    st.title("Parkinson's Disease Prediction")
    st.info(
        "The fields below are pre-filled with the average values from the dataset. "
        "Modify them if needed and click Predict."
    )
    input_values = []
    for group_name, features in FEATURE_GROUPS.items():
        st.subheader(group_name)
        cols = st.columns(2)
        for i, feature in enumerate(features):
            default = float(df[feature].mean())
            if abs(default) < 0.001:
                step = 0.000001
                fmt = "%.6f"
            elif abs(default) < 1:
                step = 0.0001
                fmt = "%.4f"
            else:
                step = 0.01
                fmt = "%.2f"
            with cols[i % 2]:
                value = st.number_input(
                    feature,
                    value=default,
                    step=step,
                    format=fmt
                )
                input_values.append(value)
    if st.button("Predict", use_container_width=True):
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        st.session_state.prediction = prediction
        st.session_state.show_result = True
        st.rerun()
else:
    st.title("Prediction Result")
    st.markdown("---")
    if st.session_state.prediction == 1:
        st.error("🔴 Parkinson's Disease Detected")
    else:

        st.success("🟢 Healthy Person")
    st.markdown("---")
    if st.button("🔄 Predict Another Patient", use_container_width=True):

        st.session_state.show_result = False
        st.session_state.prediction = None

        st.rerun()