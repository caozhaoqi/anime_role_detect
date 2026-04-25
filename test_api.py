#!/usr/bin/env python3
import requests

try:
    response = requests.get('http://localhost:8000/api/health')
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
