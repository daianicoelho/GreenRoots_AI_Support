import os
import shutil
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

def create_vector_db():
    """
    Reads the knowledge base text file, generates semantic chunks,
    and stores them in a local ChromaDB vector store.
    """
    
    # 1. Define paths
    source_file = "knowledge_base.txt"  # Formerly base_plantas.txt
    db_directory = "./vector_db"         # Formerly ./db_plantas

    if not os.path.exists(source_file):
        print(f"Error: '{source_file}' not found. Please create the file first.")
        return

    # 2. Load the knowledge base
    print(f"Loading documents from {source_file}...")
    loader = TextLoader(source_file, encoding="utf-8")
    documents = loader.load()

    # 3. Split text into semantic chunks
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 4. Initialize Local Embeddings (Runs on CPU)
    # This model is lightweight (approx. 80MB) and very efficient for RAG
    print("Generating local embeddings (processing on your machine)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Database Cleanup (Remove old version to avoid data duplication)
    if os.path.exists(db_directory):
        print("Cleaning up old database version...")
        shutil.rmtree(db_directory)

    # 6. Create and persist the Vector Store (ChromaDB)
    print("Storing vectors in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=db_directory
    )
    
    print(f"SUCCESS! Vector database created at '{db_directory}'.")

if __name__ == "__main__":
    create_vector_db()