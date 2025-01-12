import streamlit as st
import pandas as pd
import joblib
from utils import preprocessor  # Import the preprocessor class from utils.py

def run():
    # Load the trained model
    model = joblib.load('model.joblib')

    # Create an instance of the preprocessor
    preprocessor_instance = preprocessor()

    st.title("Sentiment Analysis")
    st.text("Basic app to detect the sentiment of text.")
    st.text("")

    # User input
    userinput = st.text_input('Enter text below, then click the Predict button.', placeholder='Input text HERE')
    st.text("")
    predicted_sentiment = ""

    # Prediction
    if st.button("Predict"):
        if userinput.strip():  # Ensure input is not empty
            # Preprocess the input text
            preprocessed_text = preprocessor_instance.transform(pd.Series([userinput]))[0]
            
            # Predict sentiment
            predicted_sentiment = model.predict(pd.Series([preprocessed_text]))[0]
            
            # Determine the output sentiment
            if predicted_sentiment == 1:
                output = 'positive 👍'
            else:
                output = 'negative 👎'
            
            sentiment = f'Predicted sentiment of "{userinput}" is {output}.'
            st.success(sentiment)
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    run()

