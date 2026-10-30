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

Traditional scaling of pods based on CPU/RAM is not so much relevant in world of Gen AI. Because when agent sends prompt to LLM, or does a tool call or a db lookup, the cpu becomes idle and usage drops. It is just sitting there holding an open socket, waiting for an asynchronous network response. So Agent core takes care of underlying infrastructure scaling as a serverless offering. 

The bottleneck here is how much requests per time unit the LLM supports.

The important CW metric to track:
- **ActiveSessionCount**: which AgentCore sends every minute to CW. This tracks how many users are actively connected and running agent interactions simultaneously. The platform natively scales up to handle high default thresholds (such as up to 5,000 active concurrent sessions and 200 interactions per second depending on the AWS region). Set a CloudWatch Alarm at ~70% capacity (3,500 concurrent sessions) to proactively request AWS quota increases before hitting HTTP 429 throttling errors.
- **InvocationLatency & ThrottledRequests**: Tracks downstream LLM API bottlenecks.

**LLM Rate Limits & Provisioned Throughput**:
- On-Demand Quotas: Your primary bottleneck will often be AWS Bedrock's **Tokens Per Minute (TPM) and Requests Per Minute (RPM)** limits on underlying foundation models (e.g., Claude 3.5 Sonnet).
- Provisioned Throughput (PT): For high-volume banking workloads, allocate Model Units (MUs) on Bedrock to guarantee dedicated, unthrottled inference throughput regardless of global AWS traffic.

-----

**Q3. Security**:

- The Agent core solution sits behind API gw, which does the jwt validation etc for the incoming request before passing it down to agent core.

AWS Bedrock Guardrails (Inference Level):
- Content Filtering & Denied Topics: Block system prompt injections, jailbreak attempts, and non-banking topics (e.g., financial advice disclaimers).
- Agent uses middlewares to redact PII data before feeding to the model. 
- Contextual Grounding Checks: Validate RAG outputs against retrieved documents to prevent hallucinations before the response is returned to the user.


Access Control & Fine-Grained Authorization (Tool/Data Level):
- Identity Propagation: Pass JWT tokens / OAuth2 assertions from the customer session directly into the LangGraph state. The agent must execute tool calls (e.g., get_account_balance) using RBAC / ABAC tied to the authenticated user ID, preventing Cross-Tenant Data Leakage.
- Least-Privilege IAM Roles: IAM Roles for Service Accounts so the container only has access to specific DynamoDB tables, S3 buckets, Bedrock model ARNs, or access to call external tools.


Human-in-the-Loop (HITL) Gating:
- LangGraph Intercepts: Configure LangGraph nodes to require human approval for state updates involving high-risk financial transactions (e.g., fund transfers, address changes). The agent enters a paused state until an authorized operator/customer approves the payload.

-----

**Q4. Cost**

Main cost is the multiple LLM calls, 3–5 times per request (planning, tool selection, response aggregation).

Sample monthly estimation:

Assumptions: 100,000 monthly active conversations | Average 3 tool calls per request | ~2,000 input tokens & ~300 output tokens per interaction.

- Primary Reasoning LLM (Claude 3.5 Sonnet) would cost around $2,100 – $2,700.
- Lightweight Model Routing (Amazon Nova Micro / Lite for simple steps): ~$50 – $100
- Vector DB / Knowledge Base (S3 Vectors / Aurora PGVector): ~$100 – $300
- AgentCore Runtime (container hosting): 200 – $400

Total expense: ~$2,600 – $3,750 / mo

Ways to save cost:
- Semantic caching
- Lazy load MCP tools and agent skills
- Context trimming
- Distillation
- Context trimming
- Model routing
- Quantization

**Q5. AgentCore Memory types** 

Ans. Agent core provides thse types of memories:
- Short term (session based)
- Long term (survives across sessions)

Agent write code to handle events, based on which short term memory is to be updated, or cleared. No code needed for long term, as that is extracted from short term based on stratgies defined. Each strategy has a namespace, against which data is stored. 3 strategy types:
- User Preference: what a user likes
- Semantic: facts, like user's name, place of living etc
- Summary: conversation summary. Unlike other 2, this one has a session id as well in the namespace path, as this is per session.

Reference video: https://www.youtube.com/watch?v=mIvoceSPVj0&list=PLrDJzKfz9AUuq0Rt-I0HyWgCNOYME1f6Y&index=9

You can override the defaults of strategies, select type=Semantic, and give a markdown prompt, on how the memory should be extracted.


