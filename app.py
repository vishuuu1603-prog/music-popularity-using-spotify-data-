import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px

# 1. SETUP PATHS
# This finds the folder where app.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "spotify_preprocessed_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model_pipeline.pkl")

st.set_page_config(page_title="Spotify Success Predictor", layout="wide")

# 2. LOAD DATA (with Error Handling)
@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        # This will show you exactly where the app is looking so you can debug
        st.error(f"File not found at: {CSV_PATH}")
        st.info("Ensure the CSV file is in the root of your GitHub repo.")
        return None
    return pd.read_csv(CSV_PATH)

# 3. LOAD MODEL
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# Initialize data and model
df = load_data()
pipeline = load_model()

# --- APP LAYOUT ---
st.title("🎵 Spotify Popularity Analysis & Prediction")

if df is not None and pipeline is not None:
    # Sidebar navigation
    menu = st.sidebar.radio("Menu", ["Data Dashboard", "Popularity Predictor"])

    if menu == "Data Dashboard":
        st.subheader("Dataset Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            # Distribution of Popularity
            fig = px.pie(df, names='popular', title="Percentage of Popular Songs (1=Popular)",
                         color_discrete_sequence=['#191414', '#1DB954'])
            st.plotly_chart(fig)
        
        with col2:
            # Artist Popularity vs Follower Count
            fig2 = px.scatter(df.sample(1000), x="artist_popularity", y="artist_followers", 
                              color="popular", title="Artist Popularity vs Followers (Sample)",
                              color_discrete_sequence=['#191414', '#1DB954'])
            st.plotly_chart(fig2)

    else:
        st.subheader("🔮 Predict if your song will be popular")
        st.write("Fill in the details below based on the Spotify features:")

        # Extract requirements from pipeline
        model = pipeline["model"]
        scaler = pipeline["scaler"]
        features = pipeline["features"]
        threshold = pipeline.get("threshold", 0.40)

        with st.form("predict_form"):
            col1, col2 = st.columns(2)
            with col1:
                artist_pop = st.slider("Artist Popularity Score", 0, 100, 50)
                followers = st.number_input("Artist Followers", value=5000)
                track_num = st.number_input("Track Number on Album", value=1)
                total_tracks = st.number_input("Total Tracks on Album", value=10)
            
            with col2:
                duration = st.number_input("Duration (Minutes)", value=3.2)
                year = st.number_input("Release Year", value=2024)
                is_explicit = st.checkbox("Explicit Lyrics")
                is_single = st.selectbox("Is it a Single?", ["single", "album"])

            submit = st.form_submit_button("Run Analysis")

        if submit:
            # Match the preprocessing logic from your train.ipynb
            input_dict = {
                'track_number': track_num,
                'explicit': 1 if is_explicit else 0,
                'artist_popularity': artist_pop,
                'artist_followers': followers,
                'album_total_tracks': total_tracks,
                'album_type': 1 if is_single == "single" else 0,
                'track_duration_min': duration,
                'album_release_year': year
            }
            
            # Convert to DataFrame in the exact order the model saw during training
            input_df = pd.DataFrame([input_dict])[features]
            
            # Scale and Predict
            input_scaled = scaler.transform(input_df)
            prob = model.predict_proba(input_scaled)[0, 1]
            is_popular = prob >= threshold

            st.divider()
            if is_popular:
                st.success(f"### Result: **POPULAR** (Score: {prob:.2f})")
                st.balloons()
            else:
                st.error(f"### Result: **NOT POPULAR** (Score: {prob:.2f})")
            
            st.caption(f"Note: Using a model threshold of {threshold}")

else:
    st.warning("Please check your file paths and ensure files are uploaded to GitHub.")
