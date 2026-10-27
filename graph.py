"""LangGraph agent: agent -> tools -> (grade doc relevance) -> agent loop.

Architecture Overview:
This multi-source agentic RAG system uses a cyclic LangGraph architecture.
The graph consists of three primary nodes:
1. 'agent': Calls Groq LLM (openai/gpt-oss-120b) bound to 5 tools.
2. 'tools': Executes the chosen tools via ToolNode.
3. 'grade': Corrective RAG (CRAG) node that checks retrieved document relevance.
"""
import time
from dotenv import load_dotenv
from groq import RateLimitError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import ALL_TOOLS

load_dotenv()  # also picks up LANGSMITH_* vars for tracing

SYSTEM_PROMPT = (
    "You are a helpful research assistant with tools for document search, web search, "
    "Wikipedia, stock prices, and a calculator. Pick the right tool(s) for each question. "
    "For questions about the uploaded documents, use document_search first. "
    "Never do arithmetic yourself; use the calculator. Cite which source you used. "
    "When calling tools, always supply the required argument: 'query' for search tools, "
    "'ticker' for stock_price, and 'expression' for calculator. "
    "Answer concisely and directly to minimize token usage."
)

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_retries=5)
llm_with_tools = llm.bind_tools(ALL_TOOLS)
# Grade node uses openai/gpt-oss-20b to conserve the agent's 120b daily token limit
grade_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_retries=5)


# ---------------------------------------------------------------------------
# LangGraph Nodes
# ---------------------------------------------------------------------------

def agent(state: MessagesState):
    """AGENT NODE:
    The central reasoning engine. Prepends the system prompt to the message
    history and invokes the tool-bound LLM. It decides either to:
    a) Call one or more tools (returns AIMessage with tool_calls)
    b) Generate the final natural language answer (returns plain AIMessage)
    Includes automatic rate-limit backoff and tool validation error handling.
    """
    msgs = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
    for attempt in range(5):
        try:
            response = llm_with_tools.invoke(msgs)
            return {"messages": [response]}
        except RateLimitError:
            time.sleep(3 * (attempt + 1))
        except Exception as e:
            if "tool_use_failed" in str(e) or "BadRequestError" in type(e).__name__:
                retry_msgs = msgs + [
                    SystemMessage("Reminder: All search tools take a single string parameter named 'query'.")
                ]
                try:
                    response = llm_with_tools.invoke(retry_msgs)
                    return {"messages": [response]}
                except Exception:
                    response = llm.invoke(msgs)
                    return {"messages": [response]}
            if attempt == 4:
                raise e
            time.sleep(2)
    return {"messages": [llm.invoke(msgs)]}


def grade(state: MessagesState):
    """GRADE NODE (Corrective RAG / Self-Correction):
    Evaluates retrieval quality after tool execution. If the last executed tool
    was 'document_search', this node uses the LLM to grade whether the retrieved
    passages actually contain relevant information to answer the user's question.
    - If YES: Passes through without changes.
    - If NO: Injects an advisory HumanMessage nudging the agent to fall back
      to 'web_search' or 'wikipedia_search' with the explicit user question.
    """
    last = state["messages"][-1]
    if getattr(last, "name", None) != "document_search":
        return {}
    question = next(m.content for m in state["messages"] if isinstance(m, HumanMessage))
    
    # Evaluate relevance with retry for TPM limits using lightweight grade_llm (openai/gpt-oss-20b)
    verdict = "yes"
    for attempt in range(3):
        try:
            res = grade_llm.invoke(
                f"Question: {question}\n\nRetrieved text:\n{last.content[:1500]}\n\n"
                "Is the retrieved text relevant enough to answer the question? Answer only yes or no."
            )
            verdict = res.content.strip().lower()
            break
        except RateLimitError:
            time.sleep(2 * (attempt + 1))
        except Exception:
            break

    if verdict.startswith("no"):
        return {
            "messages": [
                HumanMessage(
                    f"The document results were not relevant to the question: '{question}'. "
                    f"Please use web_search(query='{question}') or wikipedia_search(query='{question}') instead."
                )
            ]
        }
    return {}


# ---------------------------------------------------------------------------
# LangGraph Graph Construction: Nodes and Edges
# ---------------------------------------------------------------------------

builder = StateGraph(MessagesState)

# 1. Register Nodes
builder.add_node("agent", agent)                # Node: LLM Reasoning
builder.add_node("tools", ToolNode(ALL_TOOLS))  # Node: Tool Execution
builder.add_node("grade", grade)                # Node: Retrieval Evaluation / CRAG

# 2. Add Edges
# EDGE: START -> agent
# Every user query enters the graph at START and is routed directly to the agent node.
builder.add_edge(START, "agent")

# CONDITIONAL EDGE: agent -> tools OR END
# After the agent node produces an AIMessage, 'tools_condition' checks whether
# the LLM requested any tool calls:
# - If tool_calls exist: route to the 'tools' node for execution.
# - If no tool_calls: the agent has finished reasoning, route to END.
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})

# EDGE: tools -> grade
# Once tools execute and append their ToolMessages, route immediately to 'grade'
# to verify the quality of document retrieval before synthesizing an answer.
builder.add_edge("tools", "grade")

# EDGE: grade -> agent
# After grading (and potential fallback nudging), route back to 'agent' so it
# can examine the tool outputs (and feedback) to either formulate a final answer
# or trigger alternative search tools.
builder.add_edge("grade", "agent")

# Compile graph with MemorySaver checkpointer for multi-turn thread memory
graph = builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Query Interface
# ---------------------------------------------------------------------------

def ask(question: str, thread_id: str = "default") -> str:
    """Invokes the multi-source agentic RAG graph for a given question and thread_id."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(question)]}, config)
    return result["messages"][-1].content


def get_last_tools_called(thread_id: str = "default") -> list[str]:
    """Inspects the thread's checkpointed state to find tool names called in the recent turn."""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state or not state.values or "messages" not in state.values:
        return []
    tools_called = []
    # Collect tool calls from recent messages
    for msg in reversed(state.values["messages"]):
        if isinstance(msg, HumanMessage):
            # Hit previous user question, stop looking further back
            break
        if isinstance(msg, ToolMessage) and msg.name:
            if msg.name not in tools_called:
                tools_called.append(msg.name)
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name")
                if name and name not in tools_called:
                    tools_called.append(name)
    return list(reversed(tools_called))


if __name__ == "__main__":
    print(ask("What is 15% of 2480, and what is Apple's latest stock price?"))
