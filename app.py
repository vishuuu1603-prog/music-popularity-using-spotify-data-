import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# PAGE CONFIG

st.set_page_config(
    page_title="Spotify Song Success Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CUSTOM GREEN THEME CSS

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
    backdrop-filter: blur(10px);
}
.center {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# LOAD DATA & MODEL

@st.cache_data
def load_data():
    df=pd.read_csv("spotify_preprocessed_dataset.csv")
    st.write('Loading Data')
    st.write(df.head())
    return df

@st.cache_resource
def load_model():
    with open("newmodel.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

# SIDEBAR NAVIGATION

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

# HOME PAGE

if page == "Home":
    st.markdown("""
    <div class="center card">
        <h1>🎧 Spotify Song Success Intelligence Platform</h1>
        <h3>Machine Learning Based Hit Prediction</h3>
        <br>
        <p>Created by <b>Pooja Parmar</b></p>
    </div>
    """, unsafe_allow_html=True)

# SONG POPULARITY PREDICTION

elif page == "Song Popularity Prediction":
    st.title("Song Popularity Prediction")

    col1, col2 = st.columns(2)

    with col1:
        artist_popularity = st.slider("Artist Popularity", 0, 100, 50)
        artist_followers = st.number_input("Artist Followers", min_value=0, value=50000)
        track_duration = st.slider("Track Duration (minutes)", 1.0, 10.0, 3.5)

    with col2:
        album_type = st.selectbox("Album Type", ["album", "single"])
        explicit = st.selectbox("Explicit Content", ["No", "Yes"])

    album_encoded = 1 if album_type == "single" else 0
    explicit_encoded = 1 if explicit == "Yes" else 0

    # ['track_number', 'track_popularity', 'explicit', 'artist_popularity',
    #    'artist_followers', 'album_total_tracks', 'track_duration_min',
    #    'album_release_year']
    X_input = np.array([[artist_popularity,explicit_encoded ,artist_followers,track_duration,album_encoded]])

    if st.button("Predict Song Success"):
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1]

        st.subheader("Prediction Result")

        if pred == 1:
            st.success("HIT SONG")
        else:
            st.error("NOT A HIT")

        st.progress(prob)
        st.write(f"**Probability of Success:** {prob:.2f}")

        if prob > 0.7:
            st.write("🟢 **High Confidence Prediction**")
        elif prob > 0.4:
            st.write("🟡 **Medium Confidence Prediction**")
        else:
            st.write("🔴 **Low Confidence Prediction**")

        st.info(
            "Songs by popular artists with more followers and single releases "
            "have a higher chance of becoming successful."
        )

# ARTIST & GENRE ANALYSIS

elif page == "Artist & Genre Analysis":
    st.title("Artist & Genre Analysis")

    st.subheader("Top Genres by Average Popularity")
    genre_pop = df.groupby("artist_genres")["track_popularity"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(genre_pop)

    st.subheader("Genre Distribution")
    genre_count = df["artist_genres"].value_counts().head(6)
    fig1, ax1 = plt.subplots()
    ax1.pie(genre_count, labels=genre_count.index, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)

    st.subheader("Artist Popularity Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(df["artist_popularity"], bins=20)
    st.pyplot(fig2)

# ALBUM INSIGHTS

elif page == "Album Insights":
    st.title("Album Insights")

    st.subheader("Album Type vs Popularity (Spread)")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="album_type", y="track_popularity", data=df, ax=ax3)
    st.pyplot(fig3)

    st.subheader("Popularity Trend Over Years")
    year_trend = df.groupby("album_release_year")["track_popularity"].mean()
    st.area_chart(year_trend)

    st.subheader("Album Size Impact on Popularity")
    size_trend = df.groupby("album_total_tracks")["track_popularity"].mean()
    st.line_chart(size_trend)

# MODEL PERFORMANCE PAGE

elif page == "Model Performance":
    st.title("Model Performance & Insights")
    
   
    X = df[
        ["track_popularity", "explicit", "artist_popularity",  "artist_followers", "album_total_tracks",  "track_duration_min",
         "album_type"
         ]
    ].copy()

    X["album_type"] = X["album_type"].apply(lambda x: 1 if x == "single" else 0)
    X["explicit"] = X["explicit"].apply(lambda x: 1 if x == True else 0)
    st.write(X.describe())
    st.write(X.head())
    y = df["popular"]
  
    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred)

    st.subheader("Confusion Matrix")
    fig4, ax4 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax4)
    st.pyplot(fig4)

    st.subheader("Model Metrics")
    st.write(f"Accuracy: {accuracy_score(y, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y, y_pred):.2f}")
    st.write(f"Recall: {recall_score(y, y_pred):.2f}")
    st.write(f"F1 Score: {f1_score(y, y_pred):.2f}")

    st.subheader("Feature Importance (Logistic Regression)")
    importance = pd.Series(model.coef_[0], index=X.columns)
    st.bar_chart(importance.sort_values(ascending=False))
