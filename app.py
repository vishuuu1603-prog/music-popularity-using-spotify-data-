import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Spotify Song Success Intelligence",
    layout="wide"
)

# ===============================
# GREEN + 3D THEME
# ===============================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
h1, h2, h3 {
    color: #1DB954;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.4);
    backdrop-filter: blur(12px);
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL PIPELINE (CORRECT)
# ===============================
with open("model_pipeline.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
scaler = saved["scaler"]
features = saved["features"]
default_threshold = saved["threshold"]

# ===============================
# LOAD DATA (FOR CHARTS)
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("spotify_preprocessed_dataset.csv")

df = load_data()

# ===============================
# NAVIGATION
# ===============================
page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Song Popularity Prediction",
        "Artist & Genre Analysis",
        "Album Insights",
        "Model Performance"
    ]
)

# ===============================
# HOME
# ===============================
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🎧 Spotify Song Success Intelligence Platform")
    st.write("""
    A **machine‑learning based system** that predicts whether a song is likely to become popular  
    using Spotify audio features and artist metadata.

    Predictions are **probability‑based**, not rule‑based.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# PREDICTION PAGE (FIXED)
# ===============================
elif page == "Song Popularity Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🎯 Song Popularity Prediction")

    col1, col2 = st.columns(2)

    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.0)
        energy = st.slider("Energy", 0.0, 1.0, 0.0)
        loudness = st.slider("Loudness", -60.0, 0.0, -60.0)
        tempo = st.slider("Tempo", 40, 250, 40)
        track_duration_min = st.slider("Track Duration (min)", 1.0, 10.0, 1.0)

    with col2:
        artist_popularity = st.slider("Artist Popularity", 0, 100, 0)
        artist_followers = st.number_input("Artist Followers", 0, value=0)
        album_type = st.selectbox("Album Type", ["album", "single"])
        explicit = st.selectbox("Explicit Content", ["No", "Yes"])

    threshold = st.slider(
        "Decision Threshold (increase → NON‑POPULAR)",
        0.4, 0.9, 0.8
    )

    if st.button("Predict Song Success"):
        input_dict = {
            "danceability": danceability,
            "energy": energy,
            "loudness": loudness,
            "tempo": tempo,
            "track_duration_min": track_duration_min,
            "artist_popularity": artist_popularity,
            "artist_followers": artist_followers,
            "album_type": 1 if album_type == "single" else 0,
            "explicit": 1 if explicit == "Yes" else 0
        }

        input_df = pd.DataFrame([input_dict])

        # 🔑 CRITICAL: feature alignment
        for f in features:
            if f not in input_df.columns:
                input_df[f] = 0
        input_df = input_df[features]

        X_scaled = scaler.transform(input_df)
        prob = model.predict_proba(X_scaled)[0][1]
        pred = int(prob > threshold)

        st.subheader("Prediction Result")
        st.progress(int(prob * 100))
        st.write(f"**Probability:** {prob:.3f}")

        if pred == 1:
            st.success("🔥 POPULAR SONG")
        else:
            st.error("❄️ NOT POPULAR SONG")

        st.caption(f"Rule: probability > {threshold}")

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# ARTIST & GENRE ANALYSIS
# ===============================
elif page == "Artist & Genre Analysis":
    st.title("Artist & Genre Analysis")

    top_genres = df.groupby("artist_genres")["track_popularity"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(top_genres)

# ===============================
# ALBUM INSIGHTS
# ===============================
elif page == "Album Insights":
    st.title("Album Insights")

    fig, ax = plt.subplots()
    sns.boxplot(x="album_type", y="track_popularity", data=df, ax=ax)
    st.pyplot(fig)

# ===============================
# MODEL PERFORMANCE
# ===============================
elif page == "Model Performance":
    st.title("Model Performance")

    X = df[features].copy()
    X["album_type"] = (X["album_type"] == "single").astype(int)
    X["explicit"] = (X["explicit"] == True).astype(int)

    X_scaled = scaler.transform(X)
    y = df["popular"]
    y_pred = (model.predict_proba(X_scaled)[:, 1] > default_threshold).astype(int)

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
    st.pyplot(fig)

    st.write(f"Accuracy: {accuracy_score(y, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y, y_pred):.2f}")
    st.write(f"Recall: {recall_score(y, y_pred):.2f}")
    st.write(f"F1 Score: {f1_score(y, y_pred):.2f}")
