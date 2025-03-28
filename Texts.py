import os
import streamlit as st
import zipfile
import requests

# ✅ Check if scikit-learn is available
try:
    import sklearn
    st.success(f"✅ Scikit-learn Installed: {sklearn.__version__}")
except ImportError:
    st.error("❌ Scikit-learn is missing! Please check requirements.txt and restart the app.")

# ✅ Model URL and Paths
MODEL_ZIP_URL = "https://github.com/nikitamakwana0111/Emotion-Text/raw/main/classifier_emotions_model.zip"
MODEL_ZIP_PATH = "classifier_emotions_model.zip"
MODEL_DIR = "emotion_model"

# ✅ Download Model Function
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

# ✅ Extract Model Function
def extract_model():
    if not os.path.exists(MODEL_ZIP_PATH):
        download_model()
    try:
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        st.success("✅ Model extracted successfully!")
    except zipfile.BadZipFile:
        st.error("❌ Model file is corrupt! Please re-upload.")

# ✅ Streamlit UI
st.title("📢 Emotion Detection App")
st.subheader("🔍 Detect Emotions in Text")

extract_model()
st.success("🚀 App is running successfully!")

text_input = st.text_area("Enter your text here:")
if st.button("Analyze Emotion"):
    if text_input:
        st.success("🎉 Emotion detected: [Example Emotion]")
    else:
        st.warning("⚠️ Please enter some text!")
