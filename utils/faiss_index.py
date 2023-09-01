from langchain.vectorstores import FAISS

def create_faiss_index_from_docs(pages, embedding_model):
    return FAISS.from_documents(pages, embedding_model)

def search_similar_docs(faiss_index, query, k=2):
    return faiss_index.similarity_search(query, k=k)
