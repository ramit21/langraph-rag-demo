import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS 
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables (JIRA_API_TOKEN, JIRA_USERNAME, JIRA_INSTANCE_URL, OPENAI_API_KEY)
load_dotenv()

def ingest_jira():
    # 1. Setup Jira Connection using the Wrapper
    # This is more resilient than the older JiraLoader for your environment
    try:
        jira = JiraAPIWrapper()
        print("Connected to Jira successfully.")
    except Exception as e:
        print(f"Failed to connect to Jira: {e}")
        return
    
    print("Fetching tickets from Jira...")
    # Run a JQL search (e.g., project 'POC')
    # jira.run() returns a formatted string of the issues found
    issues_text = jira.run("project = 'POC' ORDER BY created DESC")
    
    if not issues_text or "No issues found" in issues_text:
        print("No issues found or Jira returned an empty result.")
        return

    # Convert the results into a LangChain Document
    documents = [Document(page_content=issues_text, metadata={"source": "jira"})]
    
    # 2. Split text into chunks
    # This helps the LLM process large batches of ticket data efficiently
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # 3. Setup Embeddings
    # Ensure your OPENAI_API_KEY is in your .env file
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 4. Store in FAISS (Local Vector Store)
    print(f"Indexing {len(docs)} chunks into FAISS...")
    
    # Create the vectorstore from the documents and embeddings
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save the index locally to a folder
    # This replaces Chroma's persist_directory
    index_path = "faiss_index"
    vectorstore.save_local(index_path)
    
    print(f"✅ Ingestion Complete! Data is saved locally in the '{index_path}' folder.")

if __name__ == "__main__":
    ingest_jira()