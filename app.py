import streamlit as st
import joblib

# Load trained model
model = joblib.load("sentiment_model.pkl")

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review and the AI will predict whether it is Positive or Negative.")

# Text input
user_input = st.text_area("Write your movie review:")

if st.button("Predict"):
    if user_input.strip() != "":
        prediction = model.predict([user_input])[0]
        probability = model.predict_proba([user_input])[0]
        confidence = max(probability)

        if prediction == 1:
            st.success(f"Positive 😊 (Confidence: {confidence:.2f})")
        else:
            st.error(f"Negative 😞 (Confidence: {confidence:.2f})")
    else:
        st.warning("Please enter a review.")