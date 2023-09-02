from langchain.vectorstores import FAISS

def create_index_from_docs(pages, embedding_model):
    return FAISS.from_documents(pages, embedding_model)

def search_similar_docs(text, query, k=2):
    return text.similarity_search(query, k=k)
