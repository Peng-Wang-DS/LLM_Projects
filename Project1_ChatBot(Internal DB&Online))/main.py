import streamlit as st
from chatbot import get_response
from langchain_test import get_qa_chain
# Title with larger font size
st.markdown("<h1 style='font-size: 40px;'>Chatbot Test - Peng 🌻</h1>", unsafe_allow_html=True)

# Input field for the user to ask a question
question = st.text_input("Ask your question:")

# Button to trigger the response
btn = st.button("Search Online")

# Response logic
response = None

# If the button is clicked or the question is entered
if btn:
    if question:
        # Get the response for the question entered from Online resources
        response = get_response(question)
else:
    if question:
        qa_function = get_qa_chain()
        response = qa_function(question) 

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
    st.markdown(f"<h2 style='font-size: 30px;'>Answer:</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 24px;'>{answer}</p>", unsafe_allow_html=True)  # Adjust answer text size here
