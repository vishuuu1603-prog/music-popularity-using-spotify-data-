import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Spotify Popularity Predictor", layout="wide")

# Get the absolute path of the directory this script is in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Use os.path.join for cross-platform compatibility
    model_path = os.path.join(BASE_DIR, "model_pipeline.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, "spotify_preprocessed_dataset.csv")
    if not os.path.exists(csv_path):
        st.error(f"Dataset not found at {csv_path}. Please upload it to your repository.")
        return pd.DataFrame()
    return pd.read_csv(csv_path)

# Initialize
try:
    saved_pipeline = load_model()
    model = saved_pipeline["model"]
    scaler = saved_pipeline["scaler"]
    feature_names = saved_pipeline["features"]
    threshold = saved_pipeline.get("threshold", 0.40)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

df = load_data()

# --- APP UI ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predictor"])

if page == "Dashboard":
    st.title("🎵 Spotify Analysis")
    if not df.empty:
        st.metric("Total Tracks Analyzed", len(df))
        fig = px.histogram(df, x="artist_popularity", color="popular", 
                           title="Artist Popularity vs Success",
                           color_discrete_sequence=['#1DB954', '#191414'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Upload 'spotify_preprocessed_dataset.csv' to see the dashboard.")

else:
    st.title("🔮 Popularity Predictor")
    # ... (Rest of the predictor code from previous response)
    with st.form("input_form"):
        # Create inputs based on feature_names
        artist_pop = st.slider("Artist Popularity", 0, 100, 50)
        artist_foll = st.number_input("Artist Followers", value=10000)
        duration = st.number_input("Duration (min)", value=3.0)
        # Simplified for brevity - ensure all feature_names are collected here
        submit = st.form_submit_button("Predict")
        
    if submit:
        # Example processing (Ensure this matches your feature_names order)
        # input_df = pd.DataFrame(...) 
        # result = model.predict(...)
        st.write("Prediction logic goes here.")
