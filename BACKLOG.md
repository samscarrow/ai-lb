# Engineering Backlog

## High Priority
- [x] **Infrastructure**: Add rate-limit handling for Cloud Providers (currently prints error, should backoff/retry). ✅ Implemented with exponential backoff, jitter, and Retry-After header support.

## Medium Priority (Reliability)
- [ ] **Worker**: Implement Type Coercion for MCP Tools.
    - **Context**: Local models (like `llama3.2`) often output numbers as strings (e.g., `{"tail": "3"}`) instead of integers.
    - **Impact**: Causes `MCP error -32602` validation failures.
    - **Proposed Fix**: Middleware in `mcp_client.py` or `worker.py` to inspect the tool schema and cast string digits to int/float automatically before sending to the MCP server.

## Low Priority
- [ ] **UX**: Add a loading spinner state for the Agent Server in Open WebUI.
- [ ] **Observability**: Export Prometheus metrics from the Agent Server (currently only in Load Balancer).
