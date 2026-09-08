import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=API_KEY)


def ask_llm(prompt: str) -> str:
    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )

    return response.output_text