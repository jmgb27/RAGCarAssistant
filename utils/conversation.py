from .vector_store import search_similar_docs

class CarManualChatbot:
    def __init__(self, llm, vector_db):
        self.llm = llm
        self.vector_db = vector_db
        self.message_history:list = []

    def get_message_history(self) -> list:
        return self.message_history

    def get_the_similar_docs(self, user_input: str) -> str:
        docs = search_similar_docs(self.vector_db, user_input)
        outputs = [doc.page_content for doc in docs]
        context = '\n\n'.join(outputs)
        return context

    def get_response(self, user_input:str) -> str:
        context = self.get_the_similar_docs(user_input)
        llm_output = self.llm.get_response(
            messages=[
                {"role": "system", "content": f"""
                    You are a car manual chatbot assistant that helps the user find the information they need from the car manual.
                    Remember that you have a history of the conversation and you can use it to answer the question.
                    Based on the context I will provide, I want you to answer my question.
                 
                    ---start of context---
                    {context}
                    ---end of context---
                    
                """}
            ] + self.message_history + [
                {"role": "user", "content": f"""{user_input}"""}
            ]
        )
        response = llm_output.get('choices')[0].get('message').get('content')
        self.message_history.append({'role':'user', 'content': user_input})
        self.message_history.append({'role':'assistant', 'content': response})
        return response
