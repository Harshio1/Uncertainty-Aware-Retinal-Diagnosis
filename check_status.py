import urllib.request
import json

url = 'https://huggingface.co/api/spaces/Harshio/adaptive-oct-classifier'
req = urllib.request.Request(url, headers={'Authorization': 'Bearer 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'})
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(f"Status: {data.get('runtime', {}).get('stage')}")
except Exception as e:
    print(f"Error: {e}")
