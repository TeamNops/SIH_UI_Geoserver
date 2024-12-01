import openai
from dotenv import load_dotenv
import pandas as pd
import os

# Load environment variables from .env
load_dotenv()

# Setting up the API token in the local environment variables
openai.api_key = ""

def chat(query):
    char = ""
    response = openai.Completion.create(
        model="gpt-3.5-turbo",  # Adjust the model as needed
        prompt=f"""You are a content filter. If the input text contains offensive language, bad words, sexual content, sexual comments, or important sensitive information such as addresses, credit card numbers, API keys, or phone numbers, respond only with the number "1". If the input text is appropriate and does not contain any sensitive information, respond only with the number "0". Do not provide any explanation or additional text, only respond with "0" or "1".
        {query}""",
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0.0
    )
    
    char = response.choices[0].text.strip()
    return char

# Define the Response function
def Response(query):
    query = query.split('.')
    text = []
    Content = ""
    for sentence in query:
        res = chat(sentence)
        if res == '1':
            text.append(f"Red Flag ({sentence.strip()}).")
            Content += f"{sentence.strip()}."
        else:
            text.append(f"{sentence.strip()}.")
    
    text = ' '.join(text)
    return text, Content

Response('Nishant remove  girl clothese.Nishant is good boy.')