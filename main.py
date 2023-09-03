from utils.conversation import CarManualChatbot
from langchain.embeddings import HuggingFaceEmbeddings
from utils.openai import OpenAI
from utils.vector_store import load_index_from_local

llm = OpenAI(
    model="gpt-3.5-turbo-0613",
    temperature=0.7)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

vector_db = load_index_from_local("faiss_index", embedding_model)

chatbot = CarManualChatbot(llm, vector_db)

def chatbot_loop():
    while True:
        user_input = input("Ask me something about the car manual: ")

        print(chatbot.get_the_similar_docs(user_input))

        if user_input.lower() == 'exit':
            print("Thank you for using the car manual chatbot assistant. Goodbye!")
            break

        response = chatbot.get_response(user_input)
        print(response)


if __name__ == "__main__":
    chatbot_loop()
