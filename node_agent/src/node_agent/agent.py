from fastapi import FastAPI, Request
import psutil
import json

app = FastAPI(title="Node Agent")

@app.get("/metrics")
def get_metrics():
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory": psutil.virtual_memory()._asdict(),
        # GPU metrics will be added here later
    }

@app.get("/v1/models")
def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-30b-a3b-2507",
                "object": "model",
                "created": 1677610602,
                "owned_by": "fake-company"
            },
        ],
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_name = body.get("model")
    messages = body.get("messages")

    # For now, just return a dummy response
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677610602,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"This is a dummy response from the node agent for model {model_name}. You said: {messages[-1]['content']}",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
