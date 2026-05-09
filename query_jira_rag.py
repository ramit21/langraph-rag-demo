import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()


# 1. Load FAISS index
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )


# 2. Build RAG chain (NO Groq call)
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        """
Use ONLY the provided context to answer the question.
If the answer is not in the context, say:
"I couldn't find this in the Jira knowledge base."

Question:
{question}

Context:
{context}

Answer:
"""
    )

    rag_chain = (
        RunnableParallel(
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
    )

    return rag_chain


# 3. Ask function: return the raw prompt object (ChatPromptValue)
def ask(question: str):
    vectorstore = load_vectorstore()
    rag_chain = build_rag_chain(vectorstore)

    # This returns a ChatPromptValue object
    prompt_value = rag_chain.invoke(question)

    # Print the raw object exactly like before
    print(prompt_value)

    return prompt_value


# 4. CLI
if __name__ == "__main__":
    print("🔍 Jira RAG Debug Mode (showing raw prompt object)")
    while True:
        q = input("\nAsk a question (or 'quit' to exit): ")
        if q.lower() in ["exit", "quit"]:
            break
        ask(q)
