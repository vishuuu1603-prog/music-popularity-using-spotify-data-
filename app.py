import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ===============================
# LOAD MODEL PIPELINE
# ===============================
with open("model_pipeline.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
scaler = saved["scaler"]
features = saved["features"]
threshold = saved["threshold"]

st.set_page_config(page_title="Spotify Popularity Predictor", layout="wide")
st.title("🎵 Spotify Song Popularity Prediction")

st.write("Model expects these features:")
st.code(features)

# ===============================
# USER INPUTS (ONLY BASE FEATURES)
# ===============================
col1, col2 = st.columns(2)

with col1:
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    loudness = st.slider("Loudness", -60.0, 0.0, -10.0)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3)
    tempo = st.number_input("Tempo (BPM)", 40.0, 250.0, 120.0)

with col2:
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
    valence = st.slider("Valence", 0.0, 1.0, 0.5)
    duration_ms = st.number_input("Duration (ms)", 30000, 600000, 210000)
    album_type = st.selectbox("Album Type", ["album", "single"])
    explicit = st.selectbox("Explicit", ["No", "Yes"])

# ===============================
# FEATURE ENGINEERING (EXACT TRAIN LOGIC)
# ===============================
input_dict = {
    "danceability": danceability,
    "energy": energy,
    "loudness": loudness,
    "speechiness": speechiness,
    "acousticness": acousticness,
    "instrumentalness": instrumentalness,
    "liveness": liveness,
    "valence": valence,
    "tempo": tempo,
    "duration_ms": duration_ms,
    "track_duration_min": duration_ms / 60000,
    "album_type": 1 if album_type == "single" else 0,
    "explicit": 1 if explicit == "Yes" else 0
}

# ===============================
# BUILD INPUT DF SAFELY
# ===============================
input_df = pd.DataFrame([input_dict])

# 🔥 Add any missing features with 0 (prevents KeyError)
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0

# 🔥 Remove extra columns
input_df = input_df[features]

# ===============================
# SCALE + PREDICT
# ===============================
input_scaled = scaler.transform(input_df)
prob = model.predict_proba(input_scaled)[0, 1]
pred = int(prob > threshold)

# ===============================
# OUTPUT
# ===============================
st.subheader("Prediction Result")
st.metric("Popularity Probability", f"{prob:.2f}")

if pred == 1:
    st.success("🔥 Song is likely to be POPULAR")
else:
    st.warning("❄️ Song is likely to be NOT popular")
