from langchain.vectorstores import FAISS

def create_index_from_docs(pages:list, embedding_model:str) -> FAISS:
    return FAISS.from_documents(pages, embedding_model)

def load_index_from_local(db_name:str, embedding_model:str) -> FAISS:
    return FAISS.load_local(db_name, embedding_model)

def search_similar_docs(text:str, query:str, k:int) -> list:
    return text.similarity_search(query, k=k)
