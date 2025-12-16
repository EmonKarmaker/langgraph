from fastapi import FastAPI, Depends, HTTPException, Header
import ollama
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Key credits system (in-memory)
API_KEY_CREDITS = {os.getenv("API_KEY"): 10}  # 10 credits to start
print(f"Loaded API Keys: {API_KEY_CREDITS}")

app = FastAPI(title="Ollama Chat API")


def verify_api_key(x_api_key: str = Header(None)):
    """Check if API key is valid and has credits"""
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing API Key. Add 'x-api-key' header.")
    
    credits = API_KEY_CREDITS.get(x_api_key, 0)
    if credits <= 0:
        raise HTTPException(status_code=401, detail="Invalid API Key or no credits remaining.")
    
    return x_api_key


@app.get("/")
def home():
    """Home endpoint"""
    return {"message": "Ollama Chat API is running!", "endpoints": ["/generate", "/credits"]}


@app.get("/credits")
def check_credits(x_api_key: str = Depends(verify_api_key)):
    """Check remaining credits"""
    return {"credits": API_KEY_CREDITS[x_api_key]}


@app.post("/generate")
def generate(prompt: str, x_api_key: str = Depends(verify_api_key)):
    """Generate a response from Ollama"""
    
    # Deduct credit
    API_KEY_CREDITS[x_api_key] -= 1
    remaining = API_KEY_CREDITS[x_api_key]
    
    # Call Ollama (change model name if needed)
    response = ollama.chat(
        model="llama3.2",  # <-- CHANGE THIS to your model
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "response": response["message"]["content"],
        "credits_remaining": remaining
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)