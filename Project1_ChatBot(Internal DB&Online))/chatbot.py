import os
from google import genai
from dotenv import load_dotenv

# Function to get content from Gemini API
def get_response(user_input):
    # Initialize the client with your Google API key
    load_dotenv()
    client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])

    # Generate content using the API client
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=user_input
    )
    
    # Return the generated content
    return response.text

if __name__ == "__main.py__":
    user_input = 'What does Carlsberg do?'
    print(get_response(user_input))