import os
import streamlit as st
import zipfile
import requests
import numpy as np
import joblib

# ✅ Model URL and Paths
MODEL_ZIP_URL = "https://github.com/nikitamakwana0111/Emotion-Text/raw/main/classifier_emotions_model.zip"
MODEL_ZIP_PATH = "classifier_emotions_model.zip"
MODEL_DIR = "emotion_model"
MODEL_FILE = os.path.join(MODEL_DIR, "emotion_classifier.pkl")

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
        if not os.path.exists(MODEL_FILE):
            extract_model()
        model = joblib.load(MODEL_FILE)  # ✅ Use joblib instead of pickle
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ✅ Emotion Prediction
def predict_emotion(text, model):
    emotions = ["anger", "fear", "joy", "love", "sadness", "surprise"]
    probabilities = model.predict_proba([text])[0]
    predicted_emotion = emotions[np.argmax(probabilities)]
    return predicted_emotion, probabilities, emotions

# ✅ Streamlit UI
st.title("Analyze the Emotion Behind Your Text")
text_input = st.text_area("Enter your text here:")

if st.button("Analyze"):
    model = load_model()
    if text_input and model:
        predicted_emotion, probabilities, emotions = predict_emotion(text_input, model)

        # ✅ UI Layout
        st.subheader("Original Text")
        st.success(text_input)
        
        st.subheader("Prediction")
        st.markdown(f"### {predicted_emotion} 😊")
        st.write(f"**Confidence:** {round(max(probabilities), 4)}")

        st.subheader("Prediction Probability")
        proba_df = {emotions[i]: round(probabilities[i], 4) for i in range(len(emotions))}
        st.write(proba_df)
    else:
        st.warning("⚠️ Please enter some text!")
