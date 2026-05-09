import langchain
import langchain_community
import os
import importlib.util

print(f"LangChain Version: {langchain.__version__}")
print(f"Community Version: {langchain_community.__version__}")

# Check where the community loaders are located
loader_path = os.path.dirname(langchain_community.__file__)
target_dir = os.path.join(loader_path, "document_loaders")

print(f"\nLooking for Jira in: {target_dir}")

# List all files that contain 'jira'
if os.path.exists(target_dir):
    files = [f for f in os.listdir(target_dir) if 'jira' in f.lower()]
    print(f"Found Jira-related files: {files}")
else:
    print("Document loaders directory not found!")