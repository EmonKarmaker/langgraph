import requests

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "my-secret-key-123"
headers = {"x-api-key": API_KEY}

print("Chat with Ollama (type 'quit' to exit)")
print("=" * 50)

while True:
    prompt = input("\nYou: ")
    
    if prompt.lower() in {"quit", "exit"}:
        print("Goodbye!")
        break
    
    response = requests.post(
        f"{BASE_URL}/generate",
        params={"prompt": prompt},
        headers=headers
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Bot: {data['response']}")
        print(f"(Credits remaining: {data['credits_remaining']})")
    else:
        print(f"Error: {response.json()}")