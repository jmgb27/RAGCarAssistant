from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PDFMinerLoader
from utils.openai import OpenAI

#define llm and embedding model
llm = OpenAI()
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

# load pdf
loader = PDFMinerLoader("data/Fronx-Owner Manual-99011M74T01-74E.pdf")
pages = loader.load_and_split()

# Create embeddings and store the embeddings into faiss vector database
faiss_index = FAISS.from_documents(pages, embedding_model)

# Initiate a loop for the conversation
message_history = []

while True:
    # Ask user for their query
    user_input = input("Ask me something about the car manual: ")

    # If user wants to exit
    if user_input.lower() in ['exit']:
        print("Thank you for using the car manual chatbot assistant. Goodbye!")
        break

    # Search the database for the relevant context
    docs = faiss_index.similarity_search(user_input, k=2)

    # Retrieve the embeddings from faiss vector database and create a list of outputs to be used as context
    outputs = []
    for doc in docs:
        output = doc.page_content

        outputs.append(output)

    context = '\n\n'.join(outputs)

    print(context)

    # Append the user message to the message history
    message_history.append({
        'role':'user',
        'content': user_input
    })

    # Get response from llm
    llm_output = llm.get_response(
        model='gpt-4',
        messages=[
            {"role": "system", "content": f"""
                You are a car manual chatbot assistant that helps the user to find the information they need from the car manual.
                Remember that you have a history of the conversation and you can use it to answer the question.
                Based on the context I will provide, I want you to answer my question.
                ---start of context---
                {context}
                ---end of context---
            """}
        ] + message_history, 
        temperature=0.7
    )

    response = llm_output.get('choices')[0].get('message').get('content')
    print(response)

    # Add the system's response to the message history
    message_history.append({
        'role':'assistant',
        'content': response
    })
