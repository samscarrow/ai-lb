#!/usr/bin/env python3
"""Debug script to list available Gemini models."""

import os
import sys

try:
    from google import genai
except ImportError:
    print("ERROR: google-genai package not installed")
    print("Install with: pip install google-genai")
    sys.exit(1)

if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY environment variable not set")
    sys.exit(1)

client = genai.Client()

print("Listing available Gemini models...\n")

try:
    models = client.models.list()

    print(f"Found {len(list(models))} models:\n")

    # List again since we consumed the iterator
    for model in client.models.list():
        print(f"  Name: {model.name}")
        if hasattr(model, 'display_name'):
            print(f"    Display Name: {model.display_name}")
        if hasattr(model, 'supported_generation_methods'):
            print(f"    Supported Methods: {model.supported_generation_methods}")
        print()

except Exception as e:
    print(f"ERROR listing models: {e}")
    sys.exit(1)

print("\nTo use a model with the orchestrator:")
print('  export ORCH_MODEL="<model_name_from_above>"')
