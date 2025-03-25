import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pickle
import zipfile

# Ensure scikit-learn is installed
os.system("pip install --no-cache-dir scikit-learn pandas numpy altair")

# Define file paths
ZIP_PATH = "classifier_emotions_model.zip"
MODEL_PATH = "classifier_emotions_model.pkl"

# Unzip and extract the model if needed
if not os.path.exists(MODEL_PATH):
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall()  # Extract in the same directory
        st.success("Model extracted successfully! ✅")
    else:
        st.error(f"Model file not found: {ZIP_PATH}. Please upload it to GitHub.")
        st.stop()

# Load the model with error handling
try:
    import sklearn  # Explicitly import to verify installation
    from sklearn.feature_extraction.text import CountVectorizer  # Example sklearn dependency
    with open(MODEL_PATH, "rb") as f:
        pipe_lr = pickle.load(f)
    st.success("Model loaded successfully! ✅")
except ModuleNotFoundError as e:
    st.error("❌ scikit-learn is still missing. Try restarting the app after ensuring installation.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Emotion labels
emotion_labels = {0: "joy", 1: "sadness", 2: "anger", 3: "fear", 4: "love", 5: "surprise"}
emotions_emoji_dict = {"joy": "😊", "sadness": "😔", "anger": "😠", "fear": "😨", "love": "❤️", "surprise": "😮"}

# Prediction functions
def predict_emotions(docx):
    predicted_label = pipe_lr.predict([docx])[0]
    return emotion_labels.get(predicted_label, "Unknown")

def get_prediction_proba(docx):
    return pipe_lr.predict_proba([docx])

# Streamlit UI
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
