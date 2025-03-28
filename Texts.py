import os
import subprocess
import sys
import streamlit as st
import zipfile
import requests
import shutil

# ✅ Function to install required packages dynamically
def install_packages():
    try:
        required_packages = ["scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl"]
        for package in required_packages:
            subprocess.run([sys.executable, "-m", "pip", "install", "--user", "--no-cache-dir", package], check=True)

        import sklearn
        st.success(f"✅ Scikit-learn Installed: {sklearn.__version__}")

    except Exception as e:
        st.error(f"❌ Error installing scikit-learn: {e}")

# Install dependencies only on Streamlit Cloud
if "appuser" in os.path.expanduser("~"):
    install_packages()

# ✅ Check model file existence and download if missing
MODEL_ZIP_URL = "https://github.com/nikitamakwana0111/Emotion-Text/raw/main/classifier_emotions_model.zip"
MODEL_ZIP_PATH = "classifier_emotions_model.zip"
MODEL_DIR = "emotion_model"

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

# ✅ Streamlit UI
st.title("📢 Emotion Detection App")
st.subheader("🔍 Detect Emotions in Text")

# Run model extraction
extract_model()

st.success("🚀 App is running successfully!")
st.write("Upload a text file or enter text to analyze emotions.")

# Add text input
text_input = st.text_area("Enter your text here:")
if st.button("Analyze Emotion"):
    if text_input:
        st.success("🎉 Emotion detected: [Example Emotion]")
    else:
        st.warning("⚠️ Please enter some text!")
