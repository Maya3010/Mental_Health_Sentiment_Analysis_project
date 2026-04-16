import streamlit as st
import joblib
from src.data_preprocess import clean_text

# Load model
model = joblib.load("artifacts/svm_model.pkl")
vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")

# Page config
st.set_page_config(page_title="Mental Health Classifier", layout="centered")

# Title
st.title("🧠 AI Mental Health Assistant")
st.write("Type how you're feeling, and the model will analyze your mental state.")


# Input
user_input = st.text_area("Enter your thoughts here:")

# Predict button
if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter some text")
    else:
        # Preprocess
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])

        # Prediction
        prediction = model.predict(vectorized)[0]

        st.subheader("Prediction:")

        # 🎨 Color-based output
        if prediction == "Suicidal":
            st.error(f"🔴 {prediction}")

            st.warning("⚠ This seems serious. Please seek immediate help.")
            st.markdown("📞 *Helpline (India): 65746483873*")

            st.info("💙 I'm really sorry you're feeling this way. You are not alone. Please reach out to a trusted person or a helpline immediately ❤️")

        elif prediction in ["Depression", "Anxiety", "Stress"]:
            st.warning(f"🟡 {prediction}")
            st.info("💡 You might be going through a difficult time. Consider talking to someone you trust.")

        elif prediction == "Normal":
            st.success(f"🟢 {prediction}")
            st.write("😊 You seem to be doing okay. Keep taking care of yourself!")

        else:
            st.info(f"🔵 {prediction}")

# Footer
st.markdown("---")
st.caption("⚠ This is not a medical diagnosis tool. Please consult a qualified professional for proper guidance.")