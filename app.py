import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import ollama

# Detect device: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Pass device to SentenceTransformer
from sentence_transformers import SentenceTransformer

class CustomSentenceTransformerEmbeddings(SentenceTransformerEmbeddings):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.client = SentenceTransformer(model_name, device=device)

embedding_function = CustomSentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "chroma_db"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

def get_relevant_chunks(query, k=5):
    results = vectordb.similarity_search(query, k=k)
    return results

def generate_answer(context, question):
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""
    # Ollama will use GPU if available and supported by the model/hardware
    response = ollama.chat(model="llama3:8b", messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip()

st.title("RAG Chatbot with Llama 3 (8B)")

user_query = st.text_input("Ask a question about your PDFs:")

if user_query:
    with st.spinner("Retrieving relevant information and generating answer..."):
        docs = get_relevant_chunks(user_query)
        context = "\n\n".join([doc.page_content for doc in docs])
        answer = generate_answer(context, user_query)
    st.markdown("### Answer")
    st.write(answer)
    with st.expander("Show retrieved context"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Chunk {i}:**\n{doc.page_content}\n---")