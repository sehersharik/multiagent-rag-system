"""Agent tools. The docstrings and Pydantic schemas matter: the LLM reads them to choose and call tools."""
import pickle

import numexpr
import wikipedia
import yfinance as yf
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from ingest import CHROMA_DIR, CHUNKS_PATH, get_embeddings

load_dotenv()

# ---------- Hybrid retriever (Chroma + BM25, fused with RRF) ----------
_vector = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())
_vector_retriever = _vector.as_retriever(search_kwargs={"k": 5})

with open(CHUNKS_PATH, "rb") as f:
    _bm25 = BM25Retriever.from_documents(pickle.load(f))
_bm25.k = 5


def hybrid_search(query: str, k: int = 4, c: int = 60):
    """Reciprocal Rank Fusion over vector and BM25 results."""
    scores, docs = {}, {}
    for retriever in (_vector_retriever, _bm25):
        for rank, doc in enumerate(retriever.invoke(query)):
            key = doc.page_content
            docs[key] = doc
            scores[key] = scores.get(key, 0) + 1 / (c + rank + 1)
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return [docs[key] for key in top]


class DocumentSearchInput(BaseModel):
    query: str = Field(description="The natural language query or topic to search within the uploaded documents.")


@tool("document_search", args_schema=DocumentSearchInput)
def document_search(query: str) -> str:
    """Search the user's uploaded documents (PDFs). Use this FIRST for any question
    about the content of the provided documents."""
    results = hybrid_search(query)
    if not results:
        return "No relevant passages found."
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in results
    )


# ---------- Web search ----------
_tavily = TavilySearch(max_results=3)


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query string to look up on the live web for current events and news.")


@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """Search the web for current events, recent news, or anything not covered by the
    documents or Wikipedia."""
    try:
        return str(_tavily.invoke({"query": query}))
    except Exception as e:
        return f"Web search failed: {e}"


# ---------- Wikipedia ----------
wikipedia.set_user_agent("MultiSourceRAG/1.0 (academic-research-bot; mailto:contact@example.com)")


class WikipediaSearchInput(BaseModel):
    query: str = Field(description="The search term, concept, person, or historical event to look up on Wikipedia.")


@tool("wikipedia_search", args_schema=WikipediaSearchInput)
def wikipedia_search(query: str) -> str:
    """Look up general background knowledge (people, places, concepts, history)
    on Wikipedia."""
    try:
        return wikipedia.summary(query, sentences=5, auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        return f"Ambiguous query. Options: {e.options[:5]}"
    except wikipedia.PageError:
        return "No Wikipedia page found."
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"


# ---------- Stock prices ----------
class StockPriceInput(BaseModel):
    ticker: str = Field(description="The stock ticker symbol, e.g. 'AAPL', 'MSFT', 'NVDA', or 'RELIANCE.NS'.")


@tool("stock_price", args_schema=StockPriceInput)
def stock_price(ticker: str) -> str:
    """Get the latest closing price for a stock ticker symbol, e.g. 'AAPL' or
    'RELIANCE.NS' for NSE-listed stocks."""
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            return f"No price data for {ticker}."
        return f"{ticker} last close: {hist['Close'].iloc[-1]:.2f} on {hist.index[-1].date()}"
    except Exception as e:
        return f"Stock lookup failed: {e}"


# ---------- Calculator ----------
class CalculatorInput(BaseModel):
    expression: str = Field(description="The mathematical arithmetic expression to evaluate, e.g. '(15 * 3.2) / 4 + 2**5'.")


@tool("calculator", args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression, e.g. '(15 * 3.2) / 4 + 2**5'. Use for any
    arithmetic instead of computing in your head."""
    try:
        return str(numexpr.evaluate(expression).item())
    except Exception as e:
        return f"Could not evaluate: {e}"


ALL_TOOLS = [document_search, web_search, wikipedia_search, stock_price, calculator]
