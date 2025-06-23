import streamlit as st
import pickle

# Load the saved model and vectorizer
with open('fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit interface
st.title("📰 Fake News Detection App")

st.markdown("Enter a news article below and let the app determine if it's **REAL** or **FAKE**.")

news_text = st.text_area("Enter the news article content here:")

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        transformed = vectorizer.transform([news_text])
        prediction = model.predict(transformed)
        prediction_proba = model.predict_proba(transformed)

        confidence = max(prediction_proba[0]) * 100

        if prediction[0] == "REAL":
            st.success(f"✅ This news is REAL with {confidence:.2f}% confidence.")
            st.progress(confidence / 100)
        else:
            st.error(f"❌ This news is FAKE with {confidence:.2f}% confidence.")
            st.progress(confidence / 100)

