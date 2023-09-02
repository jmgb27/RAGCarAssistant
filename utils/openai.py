import openai
from .config import openai_api_key

openai.api_key = openai_api_key

class OpenAI:
    def __init__(self, model: str = "gpt-3.5-turbo-0613", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature

    def get_response(self, 
                     messages: list = []):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        return response