import os
import base64
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

def ingest_jira():
    # 1. Confirm Jira connection
    try:
        jira = JiraAPIWrapper()
        print("Connected to Jira successfully.")
    except Exception as e:
        print(f"Failed to connect to Jira: {e}")
        return

    print("Fetching tickets from Jira...")

    # 2. Agile API endpoint
    board_id = 3

    # 3. Auth header
    raw_auth = f"{os.getenv('JIRA_USERNAME')}:{os.getenv('JIRA_API_TOKEN')}"
    encoded_auth = base64.b64encode(raw_auth.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    # 4. JQL
    jql = "project = POC ORDER BY updated DESC"
    encoded_jql = requests.utils.quote(jql)

    start_at = 0
    max_results = 50
    all_issues = []

    # 5. Pagination loop
    while True:
        url = (
            f"{os.getenv('JIRA_INSTANCE_URL')}/rest/agile/1.0/board/{board_id}/issue"
            f"?jql={encoded_jql}&startAt={start_at}&maxResults={max_results}"
        )

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        issues = data.get("issues", [])

        if not issues:
            break

        all_issues.extend(issues)

        if len(issues) < max_results:
            break

        start_at += max_results

    print(f"Fetched {len(all_issues)} Jira issues.\n")

    # ⭐ PRINT ONLY KEY + TYPE + SUMMARY
    print("Jira Issues:")
    for issue in all_issues:
        key = issue.get("key")
        fields = issue.get("fields", {})
        summary = fields.get("summary", "")
        issue_type = fields.get("issuetype", {}).get("name", "Unknown")
        print(f"{key} [{issue_type}]: {summary}")

    # 6. Convert each issue into a Document
    documents = []
    for issue in all_issues:
        key = issue.get("key", "UNKNOWN")
        fields = issue.get("fields", {})
        summary = fields.get("summary", "")
        description = fields.get("description", "")

        text = f"{key}\n\nSummary:\n{summary}\n\nDescription:\n{description}"
        documents.append(Document(page_content=text, metadata={"issue_key": key}))

    print(f"\nLoaded {len(documents)} Jira issues into documents.")

    # 7. Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # 8. Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print(f"Indexing {len(docs)} chunks into FAISS...")

    # 9. Build FAISS index
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 10. Save locally
    index_path = "faiss_index"
    vectorstore.save_local(index_path)

    print(f"✅ Ingestion Complete! Saved FAISS index to '{index_path}'.")

if __name__ == "__main__":
    ingest_jira()
