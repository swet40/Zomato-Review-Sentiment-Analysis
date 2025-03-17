import streamlit as st
import joblib

# Load the trained model and vectorizer
classifier = joblib.load("Zomato_review_analysis.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("🍕 Zomato Review Sentiment Analyzer")
st.write("Type a review below to analyze its sentiment!")

# Input box
review = st.text_area("Enter your review here:")

# Predict sentiment
if st.button("Analyze Sentiment"):
    if review:
        # Transform input using vectorizer
        review_vectorized = vectorizer.transform([review]).toarray()
        
        # Predict sentiment
        prediction = classifier.predict(review_vectorized)[0]
        
        # Show result
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.subheader(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review to analyze.")

