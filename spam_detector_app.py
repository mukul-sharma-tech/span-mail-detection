# spam_detector_app.py
# To run this app, save it as a Python file (e.g., app.py) and run:
# streamlit run app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # Using joblib for model persistence is often better than pickle
import os

# --- MODEL TRAINING AND PREPARATION ---
# This part of the script will only run once, and the results will be cached.

# Define file paths
DATA_FILE = 'SMSSpamCollection'
MODEL_FILE = 'spam_model.joblib'
VECTORIZER_FILE = 'vectorizer.joblib'

@st.cache_data(show_spinner="Loading data and training model...")
def load_and_train_model():
    """
    Loads the dataset, trains the Naive Bayes model, and saves the
    model and vectorizer to disk. Returns the trained model, vectorizer,
    and evaluation metrics.
    """
    # Step 1: Load the dataset
    # The file is tab-separated and has no headers
    try:
        data = pd.read_csv(DATA_FILE, sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
    except FileNotFoundError:
        st.error(f"Error: The dataset file '{DATA_FILE}' was not found.")
        st.info("Please make sure 'SMSSpamCollection' is in the same directory as the app.")
        return None, None, None, None, None

    # Step 2: Encode the label column (ham -> 0, spam -> 1)
    data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

    # Step 3: Split data into input (X) and output (y)
    X = data['message']
    y = data['label_num']

    # Step 4: Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Step 5: Convert text into numerical vectors using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)

    # Step 6: Train a Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train_counts, y_train)

    # Step 7: Make predictions for evaluation
    y_pred = model.predict(X_test_counts)

    # Step 8: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
    matrix = confusion_matrix(y_test, y_pred)

    # Save the trained model and vectorizer for future use
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    return model, vectorizer, accuracy, report, matrix

# Load the model and other assets
# If pre-trained files exist, load them. Otherwise, train from scratch.
if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        # We still need to run the training function to get evaluation metrics
        _, _, accuracy, report, matrix = load_and_train_model()
    except Exception as e:
        st.warning(f"Could not load model files. Retraining... Error: {e}")
        model, vectorizer, accuracy, report, matrix = load_and_train_model()
else:
    model, vectorizer, accuracy, report, matrix = load_and_train_model()


# --- STREAMLIT UI ---

st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# Title and description
st.title("📧 SMS Spam Detector")
st.markdown("""
This application uses a Naive Bayes classifier to determine if a message is **spam** or **ham** (not spam).
Enter a message below and click 'Predict' to see the result.
""")

# User input
st.subheader("Enter a message to classify")
user_message = st.text_area("Message:", height=150, placeholder="e.g., Congratulations! You've won a prize...")

# Prediction button
if st.button("Predict", type="primary"):
    if user_message and model and vectorizer:
        # Transform the user's message using the loaded vectorizer
        message_counts = vectorizer.transform([user_message])

        # Make a prediction
        prediction = model.predict(message_counts)
        prediction_proba = model.predict_proba(message_counts)

        # Display the result
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"**This looks like Spam** (Confidence: {prediction_proba[0][1]:.2%})")
        else:
            st.success(f"**This looks like Ham** (Confidence: {prediction_proba[0][0]:.2%})")
    elif not user_message:
        st.warning("Please enter a message to classify.")
    else:
        st.error("Model is not loaded. Please check the data file.")


# Display model performance metrics
st.sidebar.title("Model Performance")
if accuracy and report and matrix is not None:
    st.sidebar.metric("Model Accuracy", f"{accuracy:.2%}")
    st.sidebar.subheader("Classification Report")
    st.sidebar.text(report)
    st.sidebar.subheader("Confusion Matrix")
    st.sidebar.text("         Predicted")
    st.sidebar.text("         Ham  Spam")
    st.sidebar.text(f"Actual Ham: {matrix[0]}")
    st.sidebar.text(f"Actual Spam: {matrix[1]}")
else:
    st.sidebar.warning("Could not display model performance.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Scikit-learn.")
