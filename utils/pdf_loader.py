from langchain.document_loaders import PDFMinerLoader

def load_pdf(pdf_path):
    loader = PDFMinerLoader(pdf_path)
    return loader.load_and_split()
