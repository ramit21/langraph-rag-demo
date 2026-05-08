# langraph-rag-demo

Setup and steps:

Step 1: setup python env in your project folder
```
/usr/local/bin/python3 -m venv venv 
source venv/bin/activate. # run this everytime you re open the terminal
```

Step 2: Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

Step 3: Add a .env file (gitignored) that contains your keys:
# Your .env file contents
JIRA_API_TOKEN=your_token_here 
CONFLUENCE_API_TOKEN=your_token_here
GITLAB_PERSONAL_ACCESS_TOKEN=your_token_here 
GROQ_API_KEY=your_groq_token_here


