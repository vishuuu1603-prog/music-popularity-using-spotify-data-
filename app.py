import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------
# Page config
# --------------------------------
st.set_page_config(
    page_title="Spotify Music Popularity Predictor",
    page_icon="🎵",
    layout="centered"
)

# --------------------------------
# Load model
# --------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model_pipeline.pkl")

model = load_model()

# --------------------------------
# App title
# --------------------------------
st.title("🎧 Spotify Music Popularity Prediction")
st.write("Predict how popular a song might be based on its audio features.")

st.markdown("---")

# --------------------------------
# User input section
# --------------------------------
st.header("🎶 Enter Song Features")

col1, col2 = st.columns(2)

with col1:
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    loudness = st.slider("Loudness", -60.0, 0.0, -10.0)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3)

with col2:
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
    valence = st.slider("Valence", 0.0, 1.0, 0.5)
    tempo = st.number_input("Tempo (BPM)", 40.0, 250.0, 120.0)
    duration_ms = st.number_input("Duration (ms)", 30000, 600000, 210000)

# --------------------------------
# Create input dataframe
# --------------------------------
input_data = pd.DataFrame({
    "danceability": [danceability],
    "energy": [energy],
    "loudness": [loudness],
    "speechiness": [speechiness],
    "acousticness": [acousticness],
    "instrumentalness": [instrumentalness],
    "liveness": [liveness],
    "valence": [valence],
    "tempo": [tempo],
    "duration_ms": [duration_ms]
})

# --------------------------------
# Prediction
# --------------------------------
st.markdown("---")

if st.button("🎯 Predict Popularity"):
    prediction = model.predict(input_data)[0]

    st.subheader("📊 Prediction Result")

    if prediction <= 30:
        st.error(f"Popularity Score: **{prediction:.2f}** — Low 🔻")
    elif prediction <= 70:
        st.warning(f"Popularity Score: **{prediction:.2f}** — Medium ⚠️")
    else:
        st.success(f"Popularity Score: **{prediction:.2f}** — High 🔥")

    st.caption("Popularity score is typically between 0 and 100.")

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
st.write("Keys in pickle:", model.keys())
st.stop()

st.stop()

