import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
print(f"Using API key: {GEMINI_API_KEY[:5]}...{GEMINI_API_KEY[-5:]}")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')

# Test query
test_query = "What is a car?"
print(f"\nSending test query: '{test_query}'")

try:
    # Generate response
    response = model.generate_content(test_query)
    
    # Print response
    print("\nResponse received:")
    print(response.text)
    
    print("\nGemini API test successful!")
except Exception as e:
    print(f"\nError testing Gemini API: {str(e)}") 