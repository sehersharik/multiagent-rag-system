"""Run 5 distinct queries to test that each of the 5 tools is triggered by the agent."""
import json
from graph import ask, graph, get_last_tools_called
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

test_queries = [
    {
        "type": "document_search",
        "question": "According to the uploaded document, what is the course code and title for the Cognitive Computing module?",
        "thread_id": "test-doc-1"
    },
    {
        "type": "web_search",
        "question": "Search the web for the latest major technological news about James Webb Space Telescope in 2026.",
        "thread_id": "test-web-1"
    },
    {
        "type": "wikipedia_search",
        "question": "Look up Alan Turing on Wikipedia and give me a brief summary of his life.",
        "thread_id": "test-wiki-1"
    },
    {
        "type": "stock_price",
        "question": "What is the latest closing stock price for Apple (ticker: AAPL)?",
        "thread_id": "test-stock-1"
    },
    {
        "type": "calculator",
        "question": "Calculate (450 * 12) + (3500 / 7) - 2**6. Do not compute in your head, use the calculator tool.",
        "thread_id": "test-calc-1"
    }
]

print("=== TESTING 5 QUERIES ACROSS 5 TOOLS ===")
results = []

for idx, item in enumerate(test_queries, 1):
    q = item["question"]
    tid = item["thread_id"]
    expected = item["type"]
    print(f"\n[{idx}/5] Target: {expected}")
    print(f"Query: {q}")
    
    answer = ask(q, thread_id=tid)
    tools_called = get_last_tools_called(tid)
    
    print(f"Tools Chosen: {tools_called}")
    print(f"Answer snippet: {answer[:250]}...")
    
    results.append({
        "target_tool": expected,
        "query": q,
        "tools_chosen": tools_called,
        "success": expected in tools_called,
        "answer_preview": answer[:200]
    })

print("\n=== SUMMARY REPORT ===")
for r in results:
    status = "PASS" if r["success"] else "FAIL"
    print(f"[{status}] Expected '{r['target_tool']}' -> Agent chose: {r['tools_chosen']}")
