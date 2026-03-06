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
    .main {
        background-color: #FFFFFF;
        color: #000000;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #eeeeee;
    }
    </style>
    """, unsafe_allow_密=True)

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
    st.subheader("Predicting the next big hit using Machine Learning.")
    st.write("""
    This application analyzes Spotify track features to determine the probability of a song becoming popular. 
    By looking at artist followers, track duration, and release timing, we can uncover patterns in musical success.
    """)

elif selected == "Meaning of Model":
    st.title("🧠 What is this Model?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Logistic Regression")
        st.write("""
        We use **Logistic Regression**, a classification algorithm that predicts the probability of an event.
        - **Input:** Spotify metadata (Artist info, Album tracks, etc.)
        - **Output:** A probability between 0 and 1.
        """)
    with col2:
        st.markdown("### The Threshold (0.40)")
        st.write("""
        Since popular songs are rarer than non-popular ones, we use a custom threshold. 
        If the model is at least **40% confident**, we classify it as a potential hit.
        """)

elif selected == "Use of Model":
    st.title("🔮 Predict Song Success")
    if pipeline:
        model = pipeline["model"]
        scaler = pipeline["scaler"]
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
            
            if st.form_submit_button("Predict Popularity"):
                input_data = pd.DataFrame([{
                    'track_number': track_num, 'explicit': 1 if is_explicit else 0,
                    'artist_popularity': artist_pop, 'artist_followers': followers,
                    'album_total_tracks': total_tracks, 'album_type': 1 if is_single == "single" else 0,
                    'track_duration_min': duration, 'album_release_year': year
                }])[features]
                
                prob = model.predict_proba(scaler.transform(input_data))[0, 1]
                if prob >= 0.40:
                    st.success(f"🔥 **Likely Popular!** Score: {prob:.2f}")
                    st.balloons()
                else:
                    st.error(f"📉 **Unlikely to Trend.** Score: {prob:.2f}")
    else:
        st.warning("Please upload model_pipeline.pkl")

elif selected == "Analysis":
    st.title("📈 Deep Feature Analysis")
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            # CHART 1: Duration Distribution
            fig1 = px.histogram(df, x="track_duration_min", color="popular", 
                               title="Song Duration vs. Popularity",
                               color_discrete_sequence=['#000000', '#1DB954'], barmode="overlay")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # CHART 2: Explicit Content Impact
            fig2 = px.bar(df.groupby(['explicit', 'popular']).size().reset_index(name='count'), 
                         x="explicit", y="count", color="popular", 
                         title="Impact of Explicit Lyrics", barmode="group",
                         color_discrete_sequence=['#000000', '#1DB954'])
            st.plotly_chart(fig2, use_container_width=True)

elif selected == "Dashboard":
    st.title("📊 Global Dataset Dashboard")
    if df is not None:
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Tracks in Database", len(df))
        m2.metric("Artist Avg Popularity", f"{df['artist_popularity'].mean():.1f}")
        m3.metric("Popularity %", f"{(df['popular'].mean()*100):.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            # CHART 3: Popularity Pie Chart
            fig3 = px.pie(df, names='popular', title="Dataset Popularity Split",
                         hole=0.4, color_discrete_sequence=['#000000', '#1DB954'])
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # CHART 4: Scatter Plot
            fig4 = px.scatter(df.sample(800), x="artist_popularity", y="artist_followers", 
                             color="popular", size="track_duration_min",
                             title="Artist Influence vs. Follower Count",
                             color_discrete_sequence=['#000000', '#1DB954'])
            st.plotly_chart(fig4, use_container_width=True)

elif selected == "About":
    st.title("ℹ️ Project Information")
    st.write("""
    **Developer:** AI Music Analytics Team  
    **Framework:** Streamlit, Scikit-Learn, Plotly  
    **Data Source:** Spotify API Preprocessed Dataset  
    This tool is designed for labels and independent artists to test their release strategies.
    """)
