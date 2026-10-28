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

## QnA

**Q1. MLOPS: Detecting drift in a dockerised LangChain/LangGraph application hosted on AWS Bedrock AgentCore**

**Ans:**

In an agentic LLM setup, "drift" manifests as changes in user prompt distributions, shifts in tool usage patterns, rising token consumption, semantic drift in embedding spaces, or degrading evaluation scores.

1. Semantic Drift Detection on Inputs/Outputs:

Monitor user prompts or model responses over time, and compare with initial set doing cosine similarity. If difference is increasing, then tool usage is drifting than initially intended - raise alarms on low cosine simlarity scores.

2. Leverage OpenTelemetry (OTel) & Bedrock AgentCore Native Observability:

AgentCore Runtime natively handles OpenTelemetry (OTel) instrumentation for frameworks like LangGraph (via packages like openinference-instrumentation-langchain). Every invocation emits structured spans to CloudWatch:

- Invoke Agent Spans: Capture top-level user prompts and final agent responses.
- Inference Spans: Track message history and model outputs.
- Execute Tool Spans: Track which tools your LangGraph nodes are invoking, input arguments, and execution success/failure.

**Drift Detection Mechanism**: Set up CloudWatch metric filters and alerts on these traces to catch structural shifts:

- Tool Selection Drift: Track the ratio of tool calls. If your LangGraph agent suddenly starts calling a fallback search tool 80% of the time instead of querying the primary database tool, that’s a structural drift in agent behavior.
- Step/Iteration Count Drift: Track the average number of graph nodes/steps executed per session. If a complex task that used to take 3 steps now averages 8 steps, your agent is looping more due to prompt/model drift.

3. Use Automated LLM-as-a-Judge Evaluations (Online Evaluation):

AgentCore Evaluations supports online and batch evaluation workflows that tap directly into your production spans.

Continuous Evaluators: Run asynchronous or scheduled evaluation jobs using a judge model (like Claude hosted on Bedrock) against live session logs.

Metrics to Track:
- Faithfulness / Groundedness: Is the agent starting to hallucinate more than it used to?
- Answer Relevance: Are the responses still addressing the shifting user intent?
Threshold Gating: If the rolling 7-day average score for faithfulness drops below your threshold (e.g., < 0.85), an MLOps alert fires indicating **concept/performance drift**.

4. Monitor cost and latency:

Token Consumption Inflation: Track input/output token counts per session via OTel cost attribution. If prompts are growing longer or the model requires more reasoning tokens to reach an answer, it points to a shifting user behavior or prompt degradation.

Latency & Timeout Drift: Track graph execution duration. Rising execution times often signal that the LLM is struggling with ambiguous inputs or hitting deeper graph loops.

-------

**Q2. Scaling**:



-----

**Q3. Security**:



-----

**Q4. Cost**:







