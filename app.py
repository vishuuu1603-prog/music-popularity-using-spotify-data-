import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Spotify Popularity Predictor", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open("model_pipeline.pkl", "rb") as f:
        return pickle.load(f)

try:
    saved_pipeline = load_model()
    model = saved_pipeline["model"]
    scaler = saved_pipeline["scaler"]
    feature_names = saved_pipeline["features"]
    threshold = saved_pipeline.get("threshold", 0.5)
except FileNotFoundError:
    st.error("Model file 'model_pipeline.pkl' not found. Please run the training script first.")
    st.stop()

# --- LOAD DATA FOR ANALYSIS ---
@st.cache_data
def load_data():
    return pd.read_csv("spotify_preprocessed_dataset.csv")

df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard & Analysis", "Song Popularity Predictor"])

# ==========================================
# PAGE 1: DASHBOARD & ANALYSIS
# ==========================================
if page == "Dashboard & Analysis":
    st.title("🎵 Spotify Dataset Analysis")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tracks", len(df))
    col2.metric("Popular Tracks", df['popular'].sum())
    col3.metric("Avg Artist Popularity", round(df['artist_popularity'].mean(), 2))

    st.subheader("Data Overview")
    st.dataframe(df.head(10))

    st.subheader("Visual Insights")
    c1, c2 = st.columns(2)
    
    with c1:
        fig1 = px.histogram(df, x="artist_popularity", color="popular", 
                           title="Artist Popularity vs. Song Success",
                           barmode="overlay", color_discrete_sequence=['#1DB954', '#191414'])
        st.plotly_chart(fig1, use_container_width=True)
        
    with c2:
        fig2 = px.box(df, x="popular", y="track_duration_min", 
                     title="Duration Distribution by Popularity",
                     color="popular", color_discrete_sequence=['#191414', '#1DB954'])
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# PAGE 2: PREDICTOR
# ==========================================
else:
    st.title("🔮 Predict Song Popularity")
    st.write("Enter the details of a song below to see if it's likely to become **Popular**.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            artist_pop = st.slider("Artist Popularity", 0, 100, 50)
            artist_foll = st.number_input("Artist Followers", min_value=0, value=100000, step=1000)
            album_tracks = st.number_input("Total Tracks in Album", min_value=1, value=10)
            track_num = st.number_input("Track Number", min_value=1, value=1)
            
        with col2:
            duration = st.number_input("Track Duration (min)", min_value=0.1, value=3.5, step=0.1)
            release_year = st.number_input("Release Year", min_value=1950, max_value=2025, value=2024)
            album_type = st.selectbox("Album Type", ["album", "single"])
            explicit = st.checkbox("Explicit Content")

        submit = st.form_submit_button("Analyze Song")

    if submit:
        # 1. Prepare Input Data (Matching Training Order)
        # Order: ['track_number', 'explicit', 'artist_popularity', 'artist_followers', 
        #         'album_total_tracks', 'album_type', 'track_duration_min', 'album_release_year']
        
        input_data = pd.DataFrame([{
            'track_number': track_num,
            'explicit': 1 if explicit else 0,
            'artist_popularity': artist_pop,
            'artist_followers': artist_foll,
            'album_total_tracks': album_tracks,
            'album_type': 1 if album_type == "single" else 0,
            'track_duration_min': duration,
            'album_release_year': release_year
        }])
        
        # Reorder columns to match the scaler/model expectations
        input_data = input_data[feature_names]
        
        # 2. Scale
        input_scaled = scaler.transform(input_data)
        
        # 3. Predict
        prob = model.predict_proba(input_scaled)[0, 1]
        is_popular = prob > threshold
        
        # 4. Display Result
        st.divider()
        if is_popular:
            st.success(f"### Result: POPULAR! 🚀")
            st.write(f"Confidence Score: {prob:.2f} (Threshold: {threshold})")
            st.balloons()
        else:
            st.error(f"### Result: NOT POPULAR 📉")
            st.write(f"Confidence Score: {prob:.2f} (Threshold: {threshold})")

        # Feature Importance Hint
        st.info("**Tip:** Artist Popularity and Followers are usually the strongest predictors in this model.")
