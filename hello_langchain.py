import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Force load the environment
load_dotenv()

# 2. Extract and Validate the Key
raw_key = os.getenv("GROQ_API_KEY")

# If for some reason the key is wrapped in a way that looks like a function, 
# we force it to a string here.
if not isinstance(raw_key, str):
    raw_key = str(raw_key)

# 3. Initialize with explicit sync configuration
llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    openai_api_key=raw_key,
    openai_api_base="https://api.groq.com/openai/v1",
    # Strictly disable any background async behavior for this test
    default_headers={"Connection": "close"} 
)

# 4. Simple Chain
prompt = ChatPromptTemplate.from_template("Say 'Project Setup Successful' if you can read this: {input}")
chain = prompt | llm

try:
    print(f"Attempting to reach Groq with key: {raw_key[:5]}***")
    response = chain.invoke({"input": "Verification task."})
    print("\n--- Response ---")
    print(response.content)
    print("----------------\n")
except Exception as e:
    print(f"\nERROR: {e}")
