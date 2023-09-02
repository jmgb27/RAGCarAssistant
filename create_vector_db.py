from utils.pdf_loader import load_pdf
from utils.vector_store import create_index_from_docs
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 100,
)

pages = load_pdf("data/Fronx-Owner Manual-99011M74T01-74E.pdf", text_splitter)

vector_db = create_index_from_docs(pages, embedding_model)

vector_db.save_local("faiss_index")