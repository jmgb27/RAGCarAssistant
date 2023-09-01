import openai
from .config import openai_api_key

openai.api_key = openai_api_key

class OpenAI:
    def get_response(self, 
                     model: str = "gpt-3.5-turbo-0613",
                     messages: list = [],
                     temperature: float = 0.7):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response