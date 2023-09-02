from utils.pdf_loader import load_pdf
from utils.vector_store import create_index_from_docs
from utils.conversation import CarManualChatbot
from langchain.embeddings import HuggingFaceEmbeddings
from utils.openai import OpenAI

llm = OpenAI(
    model="gpt-4",
    temperature=0.7)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")

pages = load_pdf("data/Fronx-Owner Manual-99011M74T01-74E.pdf")
vector_db = create_index_from_docs(pages, embedding_model)

chatbot = CarManualChatbot(llm, vector_db)

def chatbot_loop():
    while True:
        user_input = input("Ask me something about the car manual: ")

        if user_input.lower() == 'exit':
            print("Thank you for using the car manual chatbot assistant. Goodbye!")
            break

        response = chatbot.get_response(user_input)
        print(response)


if __name__ == "__main__":
    chatbot_loop()
