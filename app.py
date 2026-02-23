import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG + STYLING
# ===============================
st.set_page_config(page_title="Spotify Popularity Predictor", layout="wide")

st.markdown("""
<style>
body {
    background-color: #ffffff;
}
.block-container {
    padding: 2rem 3rem;
}
.card {
    background: white;
    border-radius: 12px;
    padding: 25px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.12);
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
with open("model_pipeline.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
scaler = saved["scaler"]
features = saved["features"]
threshold = saved["threshold"]

# ===============================
# NAVIGATION BAR
# ===============================
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Use Cases", "Model Insights", "Prediction", "About"]
)

# ===============================
# HOME PAGE
# ===============================
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🎵 Spotify Song Popularity Prediction System")

    st.write("""
    This project predicts whether a Spotify track is **likely to become popular**
    using **audio features and album metadata**.  
    A machine learning model has been trained on real Spotify data to assist
    artists, producers, and analysts in **data‑driven music decisions**.
    """)

    st.subheader("🎯 Project Objectives")
    st.write("""
    - Analyze song audio characteristics  
    - Predict popularity using machine learning  
    - Support music industry decision‑making  
    """)

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# USE CASES PAGE
# ===============================
elif page == "Use Cases":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("📌 Applications of This Project")

    st.markdown("""
    ### Who can use this system?
    - 🎤 **Artists** – optimize tracks before release  
    - 🎧 **Music Producers** – improve hit potential  
    - 📊 **Music Analysts** – study popularity trends  
    - 🏢 **Streaming Platforms** – recommendation insights  
    - 🎼 **Record Labels** – A&R decision support  

    ### Key Benefits
    - Data‑driven predictions  
    - Reduced guesswork in music releases  
    - Better audience targeting  
    """)

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# MODEL INSIGHTS PAGE
# ===============================
elif page == "Model Insights":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("📈 Model Prediction Insights")

    st.write("The chart below illustrates how prediction probability is interpreted.")

    fig, ax = plt.subplots()
    ax.bar(["Not Popular", "Popular"], [1-threshold, threshold], color=["#ff7675", "#55efc4"])
    ax.set_ylabel("Decision Threshold")
    ax.set_title("Model Decision Boundary")

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# PREDICTION PAGE
# ===============================
elif page == "Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🎯 Predict Song Popularity")

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
        explicit = st.selectbox("Explicit Content", ["No", "Yes"])

    # 🔮 Predict Button
    if st.button("🚀 Predict Popularity"):
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

        input_df = pd.DataFrame([input_dict])

        # Ensure feature match
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[features]
        input_scaled = scaler.transform(input_df)

        prob = model.predict_proba(input_scaled)[0][1]
        pred = int(prob > threshold)

        st.subheader("📊 Prediction Result")
        st.metric("Popularity Probability", f"{prob:.2f}")

        if pred == 1:
            st.success("🔥 This song is likely to be POPULAR")
        else:
            st.warning("❄️ This song is likely to be NOT popular")

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# ABOUT PAGE
# ===============================
elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("ℹ️ About This Project")

    st.write("""
    This application was developed as a **machine learning project**
    to demonstrate real‑world prediction using Spotify audio features.

    **Core Highlights**
    - Logistic Regression with class balancing  
    - Feature scaling using StandardScaler  
    - Threshold‑based probability decision  
    - Fully deployed using Streamlit  

    Designed for **academic, analytical, and professional use**.
    """)

    st.markdown("</div>", unsafe_allow_html=True)
