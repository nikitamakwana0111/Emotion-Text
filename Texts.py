import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pickle
import zipfile
import requests

# ✅ Ensure installed packages are in PATH
os.environ["PATH"] += os.pathsep + os.path.expanduser("~/.local/bin")
sys.path.append(os.path.expanduser("~/.local/lib/python3.12/site-packages"))

# ✅ Debugging scikit-learn installation
st.write("🔍 Checking scikit-learn installation...")
try:
    import sklearn
    st.write(f"✅ Scikit-learn version: {sklearn.__version__}")
except ImportError:
    st.error("❌ Scikit-learn is missing! Installing...")
    os.system("pip install --no-cache-dir --force-reinstall scikit-learn")
    try:
        import sklearn
        st.success(f"✅ Scikit-learn installed successfully: {sklearn.__version__}")
    except ImportError:
        st.error("❌ Scikit-learn is still not found. Please restart the app.")
        st.stop()

# ✅ Define file paths
ZIP_PATH = "classifier_emotions_model.zip"
MODEL_PATH = "classifier_emotions_model.pkl"
MODEL_URL = "https://raw.githubusercontent.com/nikitamakwana0111/Emotion-Text/main/classifier_emotions_model.zip"

# ✅ Download and extract model if missing
def download_model():
    st.warning("Downloading model file...")
    try:
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200 and "Content-Type" in response.headers and "zip" in response.headers["Content-Type"]:
            with open(ZIP_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("✅ Download complete!")
        else:
            st.error("❌ Download failed: Invalid ZIP file. Please check the GitHub link or upload manually.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Model download failed: {str(e)}")
        st.stop()

def extract_model():
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall()
        st.success("✅ Model extracted successfully!")
    except zipfile.BadZipFile:
        st.error("❌ Error: File is not a valid ZIP archive. Please check the model file.")
        st.stop()

# ✅ Allow manual upload as a backup
uploaded_file = st.file_uploader("Upload classifier_emotions_model.zip", type=["zip"])
if uploaded_file is not None:
    with open(ZIP_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ Model uploaded successfully!")
    extract_model()

if not os.path.exists(MODEL_PATH):
    if os.path.exists(ZIP_PATH):
        extract_model()
    else:
        download_model()
        extract_model()

# ✅ Load the model with error handling
try:
    with open(MODEL_PATH, "rb") as f:
        pipe_lr = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# ✅ Emotion labels
emotion_labels = {0: "joy", 1: "sadness", 2: "anger", 3: "fear", 4: "love", 5: "surprise"}
emotions_emoji_dict = {"joy": "😊", "sadness": "😔", "anger": "😠", "fear": "😨", "love": "❤️", "surprise": "😮"}

# ✅ Prediction functions
def predict_emotions(docx):
    predicted_label = pipe_lr.predict([docx])[0]
    return emotion_labels.get(predicted_label, "Unknown")

def get_prediction_proba(docx):
    return pipe_lr.predict_proba([docx])

# ✅ Streamlit UI
def main():
    st.title("Text Emotion Detection 🎭")
    st.subheader("Analyze the emotion behind your text")

    with st.form(key='emotion_form'):
        raw_text = st.text_area("Enter your text here:")
        submit_text = st.form_submit_button(label='Analyze')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)
            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "❓")
            st.write(f"**{prediction}** {emoji_icon}")
            st.write(f"Confidence: **{np.max(probability):.4f}**")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=emotion_labels.values())
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["Emotion", "Probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='Emotion',
                y='Probability',
                color='Emotion'
            )
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
