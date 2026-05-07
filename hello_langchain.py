import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Load environment variables from .env
load_dotenv()

# 2. Initialize the model using the OpenAI bridge for Groq
# Note: Groq is compatible with the OpenAI API format.
llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

# 3. Create a simple prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{input}")
])

# 4. Chain them together and invoke
chain = prompt | llm

try:
    print("Connecting to Groq via LangChain...")
    response = chain.invoke({"input": "Hello! If you can read this, our RAG project setup is successful. Say 'Hello World'!"})
    print("\n--- Response ---")
    print(response.content)
    print("----------------\n")
    print("SUCCESS: Your environment is ready for Jira, Confluence, and GitLab integration.")
except Exception as e:
    print(f"\nERROR: Something went wrong. Check your API key and SSL settings.\nDetails: {e}")
