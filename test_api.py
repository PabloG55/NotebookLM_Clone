import requests
import json
import time
import subprocess

API_URL = "http://localhost:8000/api"

print("Starting FastAPI Backend...")
proc = subprocess.Popen(["venv/bin/uvicorn", "api:app", "--port", "8000"])

try:
    time.sleep(3) # Wait for server boot
    print("Testing Notebooks Endpoint...")
    
    # 1. Test Without Header
    res = requests.get(f"{API_URL}/notebooks")
    print(f"Missing Header (Expected 401): {res.status_code}")
    
    # 2. Test With Header
    res = requests.get(f"{API_URL}/notebooks", headers={"X-HF-User": "auto-tester"})
    print(f"Fetch Notebooks (Expected 200, []): {res.status_code} - {res.json()}")
    
finally:
    proc.terminate()
    print("Test finished.")
