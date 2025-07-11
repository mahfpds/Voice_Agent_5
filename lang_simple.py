import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Simple OpenAI setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_llm_response(text: str, session_id: str = "default_session") -> str:
    """Simple OpenAI API call - no local model needed"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Jane, a helpful AI assistant for Dr. James. Keep responses under 20 words."},
                {"role": "user", "content": text}
            ],
            max_tokens=50,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, I'm having trouble right now. Error: {str(e)}"
