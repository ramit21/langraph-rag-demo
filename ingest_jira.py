import os
from dotenv import load_dotenv
from langchain_community.document_loaders import JiraLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

def ingest_jira():
    # 1. Setup Jira Connection
    # Use JQL to pull a small sample first (e.g., tickets from a specific project)
    loader = JiraLoader(
        host=os.getenv("JIRA_URL"),
        url=os.getenv("JIRA_URL"),
        username=os.getenv("JIRA_USER_EMAIL"),
        api_token=os.getenv("JIRA_API_TOKEN"),
        jql="project = 'POC' ORDER BY created DESC",
        limit=10
    )
    
    print("Fetching tickets from Jira...")
    documents = loader.load()
    
    # 2. Split text into chunks (LLMs can't read massive tickets all at once)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # 3. Setup Embeddings (Pointing to Groq or OpenAI)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY") # Or your Groq-compatible embedding key
    )
    
    # 4. Store in ChromaDB
    print(f"Storing {len(docs)} chunks into ChromaDB...")
    persist_dir = os.getenv("PERSIST_DIRECTORY", "./chroma_db") # Defaults to ./chroma_db if not set
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=persist_dir
    )
    print("Ingestion Complete! Your Jira data is now local.")

if __name__ == "__main__":
    ingest_jira()