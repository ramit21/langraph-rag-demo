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
    # 1. Keep JiraAPIWrapper for connection confirmation
    try:
        jira = JiraAPIWrapper()
        print("Connected to Jira successfully.")
    except Exception as e:
        print(f"Failed to connect to Jira: {e}")
        return

    print("Fetching tickets from Jira...")

    # 2. Correct endpoint for your Jira Cloud instance
    url = f"{os.getenv('JIRA_INSTANCE_URL')}/rest/api/3/search/jql"

    # 3. Jira Cloud requires Base64("email:token")
    raw_auth = f"{os.getenv('JIRA_USERNAME')}:{os.getenv('JIRA_API_TOKEN')}"
    encoded_auth = base64.b64encode(raw_auth.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    # 4. Correct payload format for /search/jql
    board_id = 3

    jql = 'assignee = "21.ramit@gmail.com" AND project = POC ORDER BY updated DESC'
    encoded_jql = requests.utils.quote(jql)

    url = f"{os.getenv('JIRA_INSTANCE_URL')}/rest/agile/1.0/board/{board_id}/issue?jql={encoded_jql}&maxResults=50"

    response = requests.get(url, headers=headers)
    print("Status:", response.status_code)
    print("Body:", response.text)

    response.raise_for_status()

    data = response.json()
    issues = data.get("issues", [])

    if not issues:
        print("No issues returned from Jira.")
        return

    # 5. Convert each issue into a Document
    documents = []
    for issue in issues:
        key = issue.get("key", "UNKNOWN")
        fields = issue.get("fields", {})
        summary = fields.get("summary", "")
        description = fields.get("description", "")

        text = f"{key}\n\nSummary:\n{summary}\n\nDescription:\n{description}"
        documents.append(Document(page_content=text, metadata={"issue_key": key}))

    print(f"Loaded {len(documents)} Jira issues.")

    # 6. Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # 7. Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print(f"Indexing {len(docs)} chunks into FAISS...")

    # 8. Build FAISS index
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 9. Save locally
    index_path = "faiss_index"
    vectorstore.save_local(index_path)

    print(f"✅ Ingestion Complete! Saved FAISS index to '{index_path}'.")

if __name__ == "__main__":
    ingest_jira()
