import requests

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "my-secret-key-123"
headers = {"x-api-key": API_KEY}

print("=" * 50)
print("Test 1: Home")
print("=" * 50)
response = requests.get(f"{BASE_URL}/")
print(response.json())

print("\n" + "=" * 50)
print("Test 2: Check Credits")
print("=" * 50)
response = requests.get(f"{BASE_URL}/credits", headers=headers)
print(response.json())

print("\n" + "=" * 50)
print("Test 3: Generate Response")
print("=" * 50)
response = requests.post(
    f"{BASE_URL}/generate",
    params={"prompt": "What is 2 + 2? Answer in one word."},
    headers=headers
)
print(response.json())

print("\n" + "=" * 50)
print("Test 4: Check Credits Again")
print("=" * 50)
response = requests.get(f"{BASE_URL}/credits", headers=headers)
print(response.json())