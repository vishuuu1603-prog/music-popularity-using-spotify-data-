import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
from streamlit_option_menu import option_menu

# --- SETUP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "spotify_preprocessed_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model_pipeline.pkl")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Spotify Analyst", layout="wide")

# --- CUSTOM CSS FOR WHITE BACKGROUND ---
st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
    }
    h1, h2, h3, p, span {
        color: #000000 !important;
    }
    .stMetric {
        background-color: #fcfcfc;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #eeeeee;
    }
    div[data-testid="stForm"] {
        background-color: #ffffff;
        border: 1px solid #dddddd;
    }
    </style>
    """, unsafe_allow_html=True) # FIXED THE PARAMETER NAME HERE

# --- DATA & MODEL LOADING ---
@st.cache_resource
def load_assets():
    df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else None
    model_data = None
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
    return df, model_data

df, pipeline = load_assets()

# --- NAVIGATION BAR ---
selected = option_menu(
    menu_title=None, 
    options=["Home", "Meaning of Model", "Use of Model", "Analysis", "Dashboard", "About"],
    icons=["house", "journal-text", "cpu", "graph-up-arrow", "grid-3x3-gap", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f8f9fa"},
        "icon": {"color": "#1DB954", "font-size": "18px"}, 
        "nav-link": {"font-size": "14px", "text-align": "center", "margin":"0px", "color": "#000000"},
        "nav-link-selected": {"background-color": "#1DB954", "color": "#FFFFFF"},
    }
)

# --- PAGE LOGIC ---

if selected == "Home":
    st.title("🎵 Spotify Popularity Predictor")
    st.image("https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_Black.png", width=200)
    st.subheader("Leveraging Machine Learning to analyze musical success.")
    st.write("Welcome! This tool uses data science to predict if a track will be popular based on Spotify features.")

elif selected == "Meaning of Model":
    st.title("🧠 What is Logistic Regression?")
    st.write("Our app uses a Logistic Regression model to classify songs.")
    col1, col2 = st.columns(2)
    with col1:
        st.info("### The Concept")
        st.write("Logistic regression estimates the probability of an event occurring (like a song being popular) based on a given dataset of independent variables.")
    with col2:
        st.success("### The Threshold")
        st.write("We use a probability threshold of **0.40**. If the model is 40% sure a song is a hit, we classify it as 'Popular'.")

elif selected == "Use of Model":
    st.title("🔮 Prediction Engine")
    if pipeline:
        features = pipeline["features"]
        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            with c1:
                artist_pop = st.slider("Artist Popularity", 0, 100, 50)
                followers = st.number_input("Followers", value=10000)
                track_num = st.number_input("Track Number", value=1)
                total_tracks = st.number_input("Total Album Tracks", value=12)
            with c2:
                duration = st.number_input("Duration (min)", value=3.0)
                year = st.number_input("Release Year", value=2024)
                is_explicit = st.checkbox("Explicit?")
                is_single = st.selectbox("Type", ["single", "album"])
            
            if st.form_submit_button("Run Analysis"):
                input_data = pd.DataFrame([{
                    'track_number': track_num, 'explicit': 1 if is_explicit else 0,
                    'artist_popularity': artist_pop, 'artist_followers': followers,
                    'album_total_tracks': total_tracks, 'album_type': 1 if is_single == "single" else 0,
                    'track_duration_min': duration, 'album_release_year': year
                }])[features]
                
                prob = pipeline["model"].predict_proba(pipeline["scaler"].transform(input_data))[0, 1]
                if prob >= 0.40:
                    st.success(f"🔥 **Result: Popular!** (Probability: {prob:.2f})")
                    st.balloons()
                else:
                    st.error(f"📉 **Result: Not Popular.** (Probability: {prob:.2f})")
    else:
        st.warning("Model file not detected.")

elif selected == "Analysis":
    st.title("📈 Feature Insights")
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            # CHART 1: Duration Histogram
            fig1 = px.histogram(df, x="track_duration_min", color="popular", 
                               title="Chart 1: Track Duration vs Popularity",
                               color_discrete_sequence=['#191414', '#1DB954'])
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            # CHART 2: Explicit Box Plot
            fig2 = px.box(df, x="popular", y="artist_popularity", color="popular",
                         title="Chart 2: Artist Popularity Distribution",
                         color_discrete_sequence=['#191414', '#1DB954'])
            st.plotly_chart(fig2, use_container_width=True)

elif selected == "Dashboard":
    st.title("📊 Global Dashboard")
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            # CHART 3: Pie Chart
            fig3 = px.pie(df, names='popular', title="Chart 3: Dataset Popularity Split",
                         color_discrete_sequence=['#191414', '#1DB954'], hole=0.5)
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            # CHART 4: Scatter Plot
            fig4 = px.scatter(df.sample(1000), x="artist_popularity", y="artist_followers", 
                             color="popular", title="Chart 4: Followers vs Artist Popularity",
                             color_discrete_sequence=['#191414', '#1DB954'])
            st.plotly_chart(fig4, use_container_width=True)

elif selected == "About":
    st.title("ℹ️ About")
    st.write("This application was built using Streamlit and Scikit-Learn to visualize Spotify metadata and predict commercial success.")
