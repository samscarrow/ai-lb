import uuid
import time
import json
import asyncio
import logging
from typing import AsyncGenerator
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

from .orchestrator import Orchestrator
from .config import OrchestratorConfig
from .types import TaskResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_server")

app = FastAPI(title="Hybrid Agent Server")

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "hybrid-agent-orchestrator",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "agent-server",
                "permission": []
            }
        ]
    }

async def stream_orchestrator_output(request_text: str, request_id: str) -> AsyncGenerator[str, None]:
    """
    Runs the orchestrator and yields SSE chunks for Open WebUI.
    Ideally, the Orchestrator would support streaming callbacks.
    For now, we run it synchronously (blocking) and dump the result, 
    but we wrap it to look like a stream.
    
    TODO: Refactor Orchestrator to be truly async/streaming.
    """
    
    # Send generic "thinking" start
    created = int(time.time())
    
    def chunk(content: str):
        return json.dumps({
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": "hybrid-agent-orchestrator",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
        }) + "\n\n"

    try:
        # yield chunk("🧠 **Planning** with configured provider...\n\n")
        
        # We need to run the synchronous Orchestrator.run in a thread to not block the event loop
        orchestrator = Orchestrator()
        
        # We can't easily stream the *internals* of Orchestrator without refactoring it.
        # So we'll await the full result and then present it nicely.
        # A future improvement would be adding a callback hook to Orchestrator.
        
        result = await asyncio.to_thread(orchestrator.run, request_text)
        
        # Format the output for the user
        yield "data: " + chunk("### 📋 Execution Plan\n")
        for task in result.plan.tasks:
            yield "data: " + chunk(f"- **{task.id}**: {task.description}\n")
        yield "data: " + chunk("\n---\n### 🛠️ Execution Results\n")
        
        for task_id, res in result.results.items():
            icon = "✅" if res.success else "❌"
            yield "data: " + chunk(f"**{icon} Task {task_id}**\n")
            if res.output:
                yield "data: " + chunk(f"```\n{res.output[:2000]}\n```\n") # Truncate long outputs
            if res.error:
                yield "data: " + chunk(f"> *Error: {res.error}*\n")
            yield "data: " + chunk("\n")

        # Final Done
        yield "data: " + json.dumps({
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": "hybrid-agent-orchestrator",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }) + "\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Orchestrator failed: {e}", exc_info=True)
        yield "data: " + chunk(f"\n❌ **Agent Error**: {str(e)}\n")
        yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    messages = body.get("messages", [])
    # Extract the last user message
    request_text = "Hello"
    for m in reversed(messages):
        if m.get("role") == "user":
            request_text = m.get("content", "")
            break
            
    request_id = str(uuid.uuid4())
    
    return StreamingResponse(
        stream_orchestrator_output(request_text, request_id),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9995)
