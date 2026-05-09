import os
import base64
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def ingest_jira():
    # 1. Confirm Jira connection
    try:
        jira = JiraAPIWrapper()
        print("Connected to Jira successfully.")
    except Exception as e:
        print(f"Failed to connect to Jira: {e}")
        return

    #print("\n=== Underlying Jira Client Attributes ===")
    #for name in dir(jira.jira):
    #    if not name.startswith("_"):
    #        print(name)


    print("Fetching tickets from Jira...")

    # 2. Give your Jira board id
    board_id = 3

    # 3. Use JiraAPIWrapper Agile client
    response = jira.jira.get_issues_for_board(
        board_id,
        jql="project = POC ORDER BY updated DESC"
    )

    all_issues = response.get("issues", [])
    print(f"Fetched {len(all_issues)} Jira issues.\n")

    # ⭐ Print clean summary
    print("Jira Issues:")
    for issue in all_issues:
        key = issue.get("key")
        fields = issue.get("fields", {})
        summary = fields.get("summary", "")
        issue_type = fields.get("issuetype", {}).get("name", "Unknown")
        print(f"{key} [{issue_type}]: {summary}")

    # 4. Convert each issue into a Document
    documents = []
    for issue in all_issues:
        key = issue.get("key", "UNKNOWN")
        fields = issue.get("fields", {})
        summary = fields.get("summary", "")
        description = fields.get("description", "")

        text = f"{key}\n\nSummary:\n{summary}\n\nDescription:\n{description}"
        documents.append(Document(page_content=text, metadata={"issue_key": key}))

    print(f"\nLoaded {len(documents)} Jira issues into documents.")

    # 5. Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # 6. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print(f"Indexing {len(docs)} chunks into FAISS...")

    # 7. Build FAISS index
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 8. Save locally
    index_path = "faiss_index"
    vectorstore.save_local(index_path)

    print(f"✅ Ingestion Complete! Saved FAISS index to '{index_path}'.")

if __name__ == "__main__":
    ingest_jira()
