import os
from google import genai

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
for m in client.models.list(config={"page_size": 100}):
    print(m.name)
