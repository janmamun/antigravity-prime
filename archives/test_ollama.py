import requests
import json

url = "http://localhost:11434/api/generate"
model = "qwen2.5-coder:7b"
prompt = "Determine the status of Phase 16."
payload = {"model": model, "prompt": prompt, "stream": False}

print(f"Testing connection to {url} with model {model}...")
try:
    response = requests.post(url, json=payload, timeout=30)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json().get('response')}")
    else:
        print(f"Error Body: {response.text}")
except Exception as e:
    print(f"FAILED: {e}")
