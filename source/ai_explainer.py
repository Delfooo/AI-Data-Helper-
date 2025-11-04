import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_gemini_explanation(prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

def list_gemini_models(api_key):
    genai.configure(api_key=api_key)
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(m.name)