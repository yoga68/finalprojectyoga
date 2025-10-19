import fitz  # PyMuPDF
import os
import shutil
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Temporary folder for PDF storage
TEMP_PDF_DIR = "temp_pdf_uploads"

def ensure_temp_dir():
    """Create temporary directory if it doesn't exist"""
    if not os.path.exists(TEMP_PDF_DIR):
        os.makedirs(TEMP_PDF_DIR)

def cleanup_temp_dir():
    """Remove all files from temporary directory"""
    if os.path.exists(TEMP_PDF_DIR):
        shutil.rmtree(TEMP_PDF_DIR)
        os.makedirs(TEMP_PDF_DIR)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory"""
    ensure_temp_dir()
    file_path = os.path.join(TEMP_PDF_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return file_path

# Ekstrak teks dari PDF

def extract_text_from_pdf(file) -> str:
    """Extract text from a single PDF file"""
    if isinstance(file, str):  # If file is a path
        doc = fitz.open(file)
    else:  # If file is a uploaded file
        doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def process_multiple_pdfs(files) -> str:
    """Process multiple PDF files and combine their text"""
    combined_text = ""
    for file in files:
        text = extract_text_from_pdf(file)
        combined_text += f"\n\n=== Document: {file.name} ===\n\n"
        combined_text += text
    return combined_text

# Split teks menjadi chunks

def split_text(text: str, chunk_size=1000, chunk_overlap=200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

# Buat vectorstore dari chunks

def create_vectorstore(chunks: List[str]):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    return Chroma.from_texts(texts=chunks, embedding=embeddings)

# Buat conversation chain RAG

def create_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        verbose=True
    )
