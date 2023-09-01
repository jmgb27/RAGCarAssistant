from .openai import OpenAI
from .faiss_index import search_similar_docs

def handle_user_input(user_input, faiss_index):
    docs = search_similar_docs(faiss_index, user_input)
    outputs = [doc.page_content for doc in docs]
    context = '\n\n'.join(outputs)
    return context

def get_llm_response(llm, context, message_history):
    llm_output = llm.get_response(
        model='gpt-4',
        messages=[
            {"role": "system", "content": f"""
                You are a car manual chatbot assistant that helps the user find the information they need from the car manual.
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
    return response
