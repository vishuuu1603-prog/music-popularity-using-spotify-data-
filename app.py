import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Spotify Popularity Predictor",
    page_icon="🎵",
    layout="centered"
)

# --------------------------------------------------
# Load model artifacts
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    return joblib.load("model_pipeline.pkl")

artifacts = load_artifacts()

model = artifacts["model"]
scaler = artifacts["scaler"]
features = artifacts["features"]
threshold = artifacts["threshold"]

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
st.sidebar.title("🎵 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🎯 Predict", "ℹ️ About"]
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "🏠 Home":
    st.title("🎧 Spotify Music Popularity Prediction")

    st.markdown("""
    This application predicts whether a song is **POPULAR or NOT**
    using a **machine learning classification model** trained on
    Spotify track, artist, and audio features.

    ### 🎯 Target
    - `popular` → 1 (Popular)
    - `popular` → 0 (Not Popular)

    ### ⚙️ Model Logic
    - Uses probability prediction
    - Threshold-based classification
    """)

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------
elif page == "🎯 Predict":
    st.title("🎯 Predict Song Popularity")

    # -------------------------------
    # Track & Artist Features
    # -------------------------------
    st.subheader("🎵 Track & Artist Details")

    col0, col1, col2 = st.columns(3)

    with col0:
        track_number = st.number_input("Track Number", min_value=1, value=1)
        track_popularity = st.slider("Track Popularity", 0, 100, 50)
        explicit = st.selectbox("Explicit", [0, 1])

    with col1:
        artist_popularity = st.slider("Artist Popularity", 0, 100, 50)
        artist_followers = st.number_input(
            "Artist Followers", min_value=0, value=10000
        )
        artist_name = st.number_input(
            "Artist Name (Encoded ID)", min_value=0, value=0
        )

    with col2:
        artist_genres = st.number_input(
            "Artist Genres (Encoded ID)", min_value=0, value=0
        )
        track_name = st.number_input(
            "Track Name (Encoded ID)", min_value=0, value=0
        )
        track_id = st.number_input(
            "Track ID (Encoded ID)", min_value=0, value=0
        )

  

    # -------------------------------
    # Build input DataFrame (SAFE)
    # -------------------------------
    input_dict = {
        "track_number": track_number,
        "track_popularity": track_popularity,
        "explicit": explicit,
        "artist_name": artist_name,
        "artist_popularity": artist_popularity,
        "artist_followers": artist_followers,
        "artist_genres": artist_genres,
        "track_name": track_name,
        "track_id": track_id,
        
    }

 input_df = pd.DataFrame([input_dict])

missing = set(features) - set(input_df.columns)
if missing:
    st.error(f"Missing features: {missing}")
    st.stop()

input_df = input_df[features]

    st.markdown("---")

    # -------------------------------
    # Prediction
    # -------------------------------
    if st.button("🎯 Predict Popularity"):
        scaled_input = scaler.transform(input_df)

        probability = model.predict_proba(scaled_input)[0][1]
        prediction = int(probability > threshold)

        st.subheader("📊 Prediction Result")

        st.write(f"**Popularity Probability:** `{probability:.2f}`")
        st.write(f"**Threshold:** `{threshold}`")

        if prediction == 1:
            st.success("🔥 POPULAR SONG")
        else:
            st.warning("⚠️ NOT POPULAR")

# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown("""
    **Spotify Music Popularity Prediction App**

    ### 🔧 Tech Stack
    - Python
    - Scikit-learn
    - Streamlit

    ### 📊 Model
    - Classification model
    - Uses scaled numerical + encoded features
    - Threshold-based decision logic

    ### 👨‍💻 Author
    Vishva Patel
    """)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Spotify Popularity Prediction | Streamlit App")
