import time
import uuid
import json
import shlex
import subprocess
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

app = FastAPI(title="CLI Bridge Node")

# Configuration for available CLI models
CLI_MODELS = {
    "claude-cli": {
        "command": "claude -p '{prompt}' --output-format json",
        "format": "json"
    },
    "gemini-cli": {
        "command": "gemini '{prompt}' -o json",
        "format": "json"
    },
    "codex-cli": {
        "command": "codex exec '{prompt}' --json --output-last-message /tmp/codex_out_{request_id}.txt",
        "format": "json_file"
    }
}

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "cli-bridge"
            } for model_id in CLI_MODELS.keys()
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    model_id = body.get("model")
    messages = body.get("messages", [])
    
    if model_id not in CLI_MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    config = CLI_MODELS[model_id]
    req_id = str(uuid.uuid4())
    
    # Extract the last user message as the prompt
    prompt = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            prompt = m.get("content", "")
            break
            
    if not prompt:
        prompt = "Hello"

    # Construct command with placeholders
    cmd_str = config["command"].replace("{request_id}", req_id)
    
    # Safe substitution using shlex
    # Replace '{prompt}' or "{prompt}" with a marker we can swap later
    # This prevents shell injection via the prompt itself
    placeholder = f"PROMPT_PLACEHOLDER_{req_id}"
    cmd_template = cmd_str.replace("'{prompt}'", placeholder).replace('"{prompt}"', placeholder).replace("{prompt}", placeholder)
    
    cmd_parts = shlex.split(cmd_template)
    final_args = [part.replace(placeholder, prompt) for part in cmd_parts]

    print(f"Executing: {final_args}")

    try:
        result = subprocess.run(final_args, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"CLI Error: {result.stderr}")
            # If it's codex and it just didn't find anything to do, it might return non-zero but still have output
            if not result.stdout.strip():
                raise HTTPException(status_code=502, detail=f"CLI execution failed: {result.stderr}")

        output_text = result.stdout.strip()
        content = output_text

        # Handle different output formats
        if config["format"] == "json":
            try:
                parsed = json.loads(output_text)
                if isinstance(parsed, dict):
                    if "content" in parsed and isinstance(parsed["content"], list):
                         parts = [p.get("text", "") for p in parsed["content"] if p.get("type") == "text"]
                         content = "".join(parts)
                    elif "text" in parsed:
                        content = parsed["text"]
            except Exception:
                pass
        elif config["format"] == "json_file":
            # For codex, we told it to write the last message to a file
            file_path = f"/tmp/codex_out_{req_id}.txt"
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    content = f.read().strip()
                os.remove(file_path) # Cleanup
            else:
                # Fallback to parsing the JSONL from stdout
                try:
                    for line in reversed(output_text.splitlines()):
                        if not line.strip(): continue
                        data = json.loads(line)
                        if data.get("event") == "message" and data.get("message", {}).get("role") == "assistant":
                            content = data["message"].get("content", "")
                            break
                except Exception:
                    pass

        return {
            "id": f"chatcmpl-{req_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="CLI command timed out")
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9996)