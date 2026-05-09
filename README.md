# langraph-rag-demo

Setup and steps:

Step 1: setup python env in your project folder
```
rm -rf venv
/usr/local/bin/python3 -m venv venv 
source venv/bin/activate # run this everytime you re open the terminal
```

Step 2: Install dependencies
```
pip install --upgrade "pip<24.1"
pip install --prefer-binary -r requirements.txt
```

Step 3: Create an account in Groq AI, and create an API key.

Step 4: Add a .env file (gitignored) that contains your keys:
(TODO: udate with latest)
```
# Your .env file contents
JIRA_API_TOKEN=your_token_here 
CONFLUENCE_API_TOKEN=your_token_here
GITLAB_PERSONAL_ACCESS_TOKEN=your_token_here 
GROQ_API_KEY=your_groq_token_here
```

Step 5. Test GROQ connectivity by running:
```
python hello_langchain.py
```

Step 6. Create account on jira and setup a project.
```
Sign Up: Go to Atlassian Free Signup and use your Google account or email to create a site (e.g., yourname-test.atlassian.net).  

Create a Project:  Choose the "Kanban" or "Scrum" template.  

Select "Team-managed" for the project type—it’s much faster to configure than "Company-managed" for testing.  

Crucial: Note your Project Key (e.g., POC, TEST or DEV). You’ll need this for your JQL query in the Python script.

Create some jiras on the Kanban board.

Generate Your API Token:  

Visit https://id.atlassian.com/manage-profile/security/api-tokens 

Create a token labeled "RAG-Test" and copy it into your .env file immediately.

Note the jira url that I created for this POC following above steps:
https://21ramit.atlassian.net/jira/software/projects/POC/boards/3
```

Step 7: Ingest Jira into vector DB
```
python ingest_jira.py
```
It connects to your jira instance, pulls jiras from your board.
chunks them, and then embeds using local and free model SentenceTransformers, and stores the vectors into FAISS DB. 

Step 8: Query Jira faiss store.

LangChain RAG query script that works with: 
FAISS (your local vector DB),
SentenceTransformers embeddings (free, local),
Groq LLM for answering queries.

```
python query_jira_rag.py
```

