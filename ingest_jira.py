import os
from dotenv import load_dotenv
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

def ingest_jira():
    # 1. Setup Jira Connection using the Wrapper
    # It automatically looks for JIRA_API_TOKEN, JIRA_USERNAME (your email), 
    # and JIRA_INSTANCE_URL (your JIRA_URL) in your .env
    jira = JiraAPIWrapper()
    
    print("Fetching tickets from Jira...")
    # Run a JQL search through the wrapper
    # This returns a list of dictionaries containing ticket data
    issues = jira.run("project = 'POC' ORDER BY created DESC")
    
    # Convert the raw string/dict results into LangChain Documents
    # (The wrapper's .run() usually returns a formatted string of results)
    documents = [Document(page_content=issues, metadata={"source": "jira"})]
    
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