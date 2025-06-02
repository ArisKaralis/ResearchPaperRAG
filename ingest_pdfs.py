from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os
import tqdm

# 1. Load all PDFs from the 'pdfs' folder
pdf_folder = "pdfs"
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

all_docs = []
for pdf_file in tqdm.tqdm(pdf_files):
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    all_docs.extend(docs)

# 2. Chunk the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs_chunks = text_splitter.split_documents(all_docs)

# 3. Set up embeddings and ChromaDB
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "chroma_db"

# 4. Create Chroma vector store and add documents
vectordb = Chroma.from_documents(
    docs_chunks,
    embedding_function,
    persist_directory=persist_directory
)

# 5. Persist the database to disk
vectordb.persist()

print(f"Ingested {len(docs_chunks)} chunks from {len(pdf_files)} PDFs into ChromaDB.")