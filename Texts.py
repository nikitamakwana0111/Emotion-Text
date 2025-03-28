import os
import streamlit as st
import zipfile
import requests
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ✅ Model URL and Paths
MODEL_ZIP_URL = "https://github.com/nikitamakwana0111/Emotion-Text/raw/main/classifier_emotions_model.zip"
MODEL_ZIP_PATH = "classifier_emotions_model.zip"
MODEL_DIR = "emotion_model"
MODEL_FILE = "emotion_model/emotion_classifier.pkl"

# ✅ Download Model
def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    try:
        st.info("📥 Downloading Model...")
        response = requests.get(MODEL_ZIP_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_ZIP_PATH, "wb") as f:
                f.write(response.content)
            st.success("✅ Model downloaded successfully!")
        else:
            st.error(f"❌ Model download failed! HTTP Error {response.status_code}")
    except Exception as e:
        st.error(f"❌ Error downloading model: {e}")

# ✅ Extract Model
def extract_model():
    if not os.path.exists(MODEL_ZIP_PATH):
        download_model()
    try:
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        st.success("✅ Model extracted successfully!")
    except zipfile.BadZipFile:
        st.error("❌ Model file is corrupt! Please re-upload.")

# ✅ Load Pre-trained Model
def load_model():
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ✅ Emotion Prediction
def predict_emotion(text, model):
    emotions = ["anger", "fear", "joy", "love", "sadness", "surprise"]
    probabilities = model.predict_proba([text])[0]  # Get confidence scores
    predicted_emotion = emotions[np.argmax(probabilities)]
    return predicted_emotion, probabilities, emotions

# ✅ Plot Confidence Chart
def plot_confidence_chart(probabilities, emotions):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(emotions, probabilities, color=['blue', 'red', 'green', 'pink', 'purple', 'lightgreen'])
    ax.set_ylabel("Probability")
    ax.set_xlabel("Emotion")
    ax.set_title("Prediction Probability")
    return fig

# ✅ Streamlit UI
st.title("Analyze the emotion behind your text")

text_input = st.text_area("Enter your text here:")

if st.button("Analyze"):
    extract_model()
    model = load_model()
    
    if text_input and model:
        predicted_emotion, probabilities, emotions = predict_emotion(text_input, model)

        # ✅ Layout with Two Columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Text")
            st.success(text_input)

            st.subheader("Prediction")
            st.markdown(f"### {predicted_emotion} 😊")
            st.write(f"**Confidence:** {round(max(probabilities), 4)}")

        with col2:
            st.subheader("Prediction Probability")
            fig = plot_confidence_chart(probabilities, emotions)
            st.pyplot(fig)
    
    else:
        st.warning("⚠️ Please enter some text!")
