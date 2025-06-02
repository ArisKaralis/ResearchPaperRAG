from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import ollama

def get_relevant_chunks(query, k=5):
    persist_directory = "chroma_db"
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    results = vectordb.similarity_search(query, k=k)
    return results

def generate_answer(context, question):
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""
    response = ollama.chat(model="llama3:8b", messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip()

if __name__ == "__main__":
    while True:
        user_query = input("\nAsk a question (or type 'exit'): ")
        if user_query.lower() == "exit":
            break
        docs = get_relevant_chunks(user_query)
        context = "\n\n".join([doc.page_content for doc in docs])
        print("\nGenerating answer...\n")
        answer = generate_answer(context, user_query)
        print(f"Answer:\n{answer}\n")