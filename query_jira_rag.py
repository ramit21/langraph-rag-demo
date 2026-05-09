import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from groq import Groq

load_dotenv()

# 1. Load FAISS index
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

# 2. Groq LLM wrapper
def groq_llm(prompt):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

# 3. Build RAG chain
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant answering questions using Jira project knowledge.

Use ONLY the provided context. If the answer is not in the context, say:
"I couldn't find this in the Jira knowledge base."

Question:
{question}

Context:
{context}

Answer:
""")

    # RAG pipeline
    rag_chain = (
        RunnableParallel(
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
        )
        | prompt
        | (lambda x: groq_llm(x))
    )

    return rag_chain

# 4. Query function
def ask(question):
    vectorstore = load_vectorstore()
    rag_chain = build_rag_chain(vectorstore)
    answer = rag_chain.invoke(question)
    return answer

# 5. CLI
if __name__ == "__main__":
    print("🔍 Jira RAG Assistant (Groq‑powered)")
    while True:
        q = input("\nAsk a question: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("\n🧠 Answer:")
        print(ask(q))
