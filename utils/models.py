from langchain.embeddings import HuggingFaceEmbeddings
from .openai import OpenAI

def get_llm():
    return OpenAI()

def get_embedding_model(model_name="BAAI/bge-base-en"):
    return HuggingFaceEmbeddings(model_name=model_name)
