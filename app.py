import streamlit as st
import pandas as pd
import joblib

# ----------------------------------
# Page configuration
# ----------------------------------
st.set_page_config(
    page_title="Spotify Music Popularity Predictor",
    page_icon="🎵",
    layout="centered"
)

# ----------------------------------
# Load trained artifacts
# ----------------------------------
@st.cache_resource
def load_artifacts():
    return joblib.load("model_pipeline.pkl")

artifacts = load_artifacts()

model = artifacts["model"]
scaler = artifacts["scaler"]
features = artifacts["features"]
threshold = artifacts["threshold"]

# ----------------------------------
# Sidebar Navigation
# ----------------------------------
st.sidebar.title("🎵 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🎯 Predict Popularity", "ℹ️ About"]
)

# ----------------------------------
# HOME PAGE
# ----------------------------------
if page == "🏠 Home":
    st.title("🎧 Spotify Music Popularity Prediction")
    st.markdown(
        """
        Welcome to the **Spotify Music Popularity Predictor** 🎶  

        This app uses **Machine Learning** to predict whether a song  
        is likely to become **popular** based on its audio features.

        ### 🔍 Features Used
        - Danceability  
        - Energy  
        - Loudness  
        - Speechiness  
        - Acousticness  
        - Instrumentalness  
        - Liveness  
        - Valence  
        - Tempo  
        - Duration  

        👉 Use the **Predict Popularity** tab to get started!
        """
    )

# ----------------------------------
# PREDICTION PAGE
# ----------------------------------
elif page == "🎯 Predict Popularity":
    st.title("🎯 Predict Song Popularity")

    st.markdown("---")
    st.subheader("🎶 Enter Song Features")

    col1, col2 = st.columns(2)

    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        energy = st.slider("Energy", 0.0, 1.0, 0.5)
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3)

    with col2:
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
        valence = st.slider("Valence", 0.0, 1.0, 0.5)
        tempo = st.number_input("Tempo (BPM)", 40.0, 250.0, 120.0)
        duration_ms = st.number_input(
            "Duration (milliseconds)", 30000, 600000, 210000
        )

    # Create input dataframe with correct feature order
    input_data = pd.DataFrame(
        [[
            danceability,
            energy,
            loudness,
            speechiness,
            acousticness,
            instrumentalness,
            liveness,
            valence,
            tempo,
            duration_ms
        ]],
        columns=features
    )

    st.markdown("---")

    if st.button("🎯 Predict Popularity"):
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]

        st.subheader("📊 Prediction Result")

        if prediction >= threshold:
            st.success(f"🔥 Popular Song!\n\nPredicted Score: **{prediction:.2f}**")
        else:
            st.warning(f"⚠️ Less Popular Song\n\nPredicted Score: **{prediction:.2f}**")

# ----------------------------------
# ABOUT PAGE
# ----------------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown(
        """
        **Spotify Music Popularity Prediction App** 🎵  

        ### 🔧 Tech Stack
        - Python  
        - Scikit-learn  
        - Streamlit  

        ### 📊 Model Details
        - Trained on Spotify audio features
        - Uses scaling + ML regression
        - Threshold-based popularity classification

        ### 👨‍💻 Author
        Created by **Vishva Patel**  
        """
    )

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption("Spotify Popularity Predictor | Streamlit App")
