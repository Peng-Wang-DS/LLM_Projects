import streamlit as st
from chatbot import get_response

# Title with larger font size
st.markdown("<h1 style='font-size: 40px;'>Chatbot Test - Peng 🌻</h1>", unsafe_allow_html=True)

# Input field for the user to ask a question
question = st.text_input("Question: ")

# Button to trigger the response
btn = st.button("Get Response")

# Response logic
response = None

# If the button is clicked or the question is entered
if btn or question:
    if question:
        # Get the response for the question entered
        response = get_response(question)

# Display the response if it's available
if response:
    # Check if the response is a dictionary and contains the "result" key
    if isinstance(response, dict):
        if "result" in response:
            answer = response["result"]
        else:
            answer = "No 'result' key in response."
    else:
        # If the response is a string or other type, just show it
        answer = response

    # Display the answer with a larger font size
    st.markdown(f"<h2 style='font-size: 30px;'>Answer</h2>", unsafe_allow_html=True)  # Adjust answer header size here
    st.markdown(f"<p style='font-size: 24px;'>{answer}</p>", unsafe_allow_html=True)  # Adjust answer text size here
