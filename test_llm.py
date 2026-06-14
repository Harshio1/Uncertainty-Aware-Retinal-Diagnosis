
import os
import json
from huggingface_hub import InferenceClient

api_key = os.environ.get("HF_TOKEN", "").strip()
print(f"Token present: {bool(api_key)}")

client = InferenceClient(api_key=api_key)

messages = [
    {
        "role": "user",
        "content": "Hello, can you hear me? Respond with 'Yes' if you can."
    }
]

try:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
        max_tokens=10,
    )
    print("Response:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
