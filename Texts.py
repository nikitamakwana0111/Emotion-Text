import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pickle
import zipfile
import os

# Define model file paths
ZIP_FILE_PATH = "classifier_emotions_model.zip"
MODEL_FILE_PATH = "classifier_emotions_model.pkl"

# Extract model from ZIP if necessary
if not os.path.exists(MODEL_FILE_PATH):
    if os.path.exists(ZIP_FILE_PATH):
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            zip_ref.extract(MODEL_FILE_PATH)
    else:
        st.error(f"❌ Model ZIP file '{ZIP_FILE_PATH}' not found! Please check the path.")
        st.stop()

# Load the trained model
with open(MODEL_FILE_PATH, "rb") as f:
    pipe_lr = pickle.load(f)

# Emotion labels and emojis
emotion_labels = {0: "joy", 1: "sadness", 2: "anger", 3: "fear", 4: "love", 5: "surprise"}
emotions_emoji_dict = {
    "joy": "😊", "sadness": "😔", "anger": "😠",
    "fear": "😨", "love": "❤️", "surprise": "😮"
}

# Prediction functions
def predict_emotions(text):
    predicted_label = pipe_lr.predict([text])[0]
    return emotion_labels.get(predicted_label, "Unknown")

def get_prediction_proba(text):
    return pipe_lr.predict_proba([text])

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
