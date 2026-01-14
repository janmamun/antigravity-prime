from meta_brain import LocalBrain
import requests

brain = LocalBrain()
print(f"Testing Chat with model: {brain.chat_model}")
print(f"URL: {brain.url}")

try:
    resp = brain.chat("Are you there?")
    print(f"Response: {resp}")
except Exception as e:
    print(f"Error: {e}")
