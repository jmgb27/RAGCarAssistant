from utils.models import get_llm, get_embedding_model
from utils.pdf_loader import load_pdf
from utils.faiss_index import create_faiss_index_from_docs
from utils.conversation import handle_user_input, get_llm_response

llm = get_llm()
embedding_model = get_embedding_model()
pages = load_pdf("data/Fronx-Owner Manual-99011M74T01-74E.pdf")
faiss_index = create_faiss_index_from_docs(pages, embedding_model)
message_history = []

def chatbot_loop():
    while True:
        user_input = input("Ask me something about the car manual: ")

        if user_input.lower() == 'exit':
            print("Thank you for using the car manual chatbot assistant. Goodbye!")
            break

        context = handle_user_input(user_input, faiss_index)

        message_history.append({'role':'user', 'content': user_input})
        response = get_llm_response(llm, context, message_history)
        print(response)
        message_history.append({'role':'assistant', 'content': response})

if __name__ == "__main__":
    chatbot_loop()
