"""FastAPI backend for Multi-Source Agentic RAG System.

Endpoints:
- POST /api/chat : Processes user messages via LangGraph multi-source agent
- GET /          : Serves static/index.html UI
- /static        : Static assets mount
"""
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from graph import graph

app = FastAPI(title="Multi-Source Agentic RAG API")

# Enable CORS for local dev / testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


def extract_text(content: Any) -> str:
    """Extract string content whether content is a string or list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content) if content is not None else ""


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest) -> Dict[str, Any]:
    # Validate non-empty message
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    thread_id = request.thread_id.strip() if request.thread_id and request.thread_id.strip() else str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Count messages in the thread prior to invocation
        prior_state = graph.get_state(config)
        prior_msgs = prior_state.values.get("messages", []) if (prior_state and prior_state.values) else []
        prior_count = len(prior_msgs)

        # Invoke the LangGraph agent
        result = graph.invoke({"messages": [HumanMessage(request.message)]}, config)
        all_msgs = result.get("messages", [])

        # Extract only messages generated during this turn (skip the user's HumanMessage at turn_msgs[0])
        turn_msgs = all_msgs[prior_count:]
        steps: List[Dict[str, Any]] = []

        # Iterate through turn messages skipping user's initial prompt
        for msg in turn_msgs[1:]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    steps.append({
                        "type": "call",
                        "tool": tc.get("name", "tool"),
                        "input": tc.get("args", {})
                    })
            elif isinstance(msg, ToolMessage):
                content_str = extract_text(msg.content)
                steps.append({
                    "type": "result",
                    "tool": msg.name or "tool",
                    "output": content_str[:1500]
                })
            elif isinstance(msg, HumanMessage):
                # Grader node injects a HumanMessage advisory when doc search is irrelevant
                content_str = extract_text(msg.content)
                steps.append({
                    "type": "retry",
                    "tool": "grader",
                    "output": content_str
                })

        # Final answer is the last message's text
        last_msg = all_msgs[-1] if all_msgs else None
        answer = extract_text(last_msg.content) if last_msg else ""

        return {
            "thread_id": thread_id,
            "answer": answer,
            "steps": steps
        }

    except HTTPException:
        raise
    except Exception as e:
        # 502 with readable error message for agent failures (rate limit, API, network)
        raise HTTPException(
            status_code=502,
            detail=f"Agent error: {str(e)}"
        )


# Mount static assets and serve root index.html
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")
