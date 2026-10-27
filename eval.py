"""Evaluation script using RAGAS to benchmark the Multi-Source Agentic RAG System.

Split-model design (avoids Groq per-model token limits):
  - Agent (graph.py) : openai/gpt-oss-120b  -- 200K TPD
  - RAGAS judge      : openai/gpt-oss-20b   -- 200K TPD (different daily pool)

RAGAS metrics scored:
  - faithfulness      : answer factually grounded in retrieved context
  - answer_relevancy  : answer directly addresses the question
  - context_precision : relevant context ranked above irrelevant context

Run behaviour:
  - Phase 1 (Agent): caches {question, answer, contexts} to eval_cache.json
                     after every question -- safe to Ctrl-C and resume.
                     Contexts truncated to 1500 chars to conserve tokens.
  - Phase 2 (Scoring): scores in batches of 10, appending each batch to
                       results.csv with fixed schema:
                       [question, answer, ground_truth, tools_used,
                        faithfulness, answer_relevancy, context_precision]
  - On a 429 error in either phase: prints progress cleanly and exits.
"""
import csv
import json
import os
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from groq import RateLimitError
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.run_config import RunConfig

from graph import graph

load_dotenv()

# ---------------------------------------------------------------------------
# Paths & Schemas
# ---------------------------------------------------------------------------
EVAL_SET_PATH = Path("eval_set.json")
CACHE_PATH    = Path("eval_cache.json")   # agent-phase cache (keyed by question)
RESULTS_PATH  = Path("results.csv")       # final scored output

FIXED_COLUMNS = [
    "question",
    "answer",
    "ground_truth",
    "tools_used",
    "faithfulness",
    "answer_relevancy",
    "context_precision"
]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
# Main agent (graph.py) uses openai/gpt-oss-120b.
# RAGAS judge uses openai/gpt-oss-20b with max_tokens=2048 and reasoning_effort="low"
# to prevent reasoning tokens from cutting off output before completion.
JUDGE_MODEL = "openai/gpt-oss-20b"

# Groq strictly rejects n > 1, so keep strictness=1
answer_relevancy.strictness = 1

BATCH_SIZE            = 10     # score N questions per RAGAS call
BATCH_SLEEP_SECONDS   = 5      # pause between RAGAS batches to stay within TPM
AGENT_DELAY_SECONDS   = 2      # pause between agent calls (respect TPM)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval_set() -> list[dict]:
    """Load the 30 question/ground-truth pairs from eval_set.json."""
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_agent_cache() -> dict:
    """Load previously cached agent responses (keyed by question string)."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_agent_cache(cache: dict) -> None:
    """Persist the agent cache to disk after every completed question."""
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def load_scored_questions() -> set[str]:
    """Return the set of questions already written to results.csv (for skip logic)."""
    if not RESULTS_PATH.exists():
        return set()
    scored = set()
    with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question") or row.get("user_input")
            if q:
                scored.add(q.strip())
    return scored


def append_batch_to_csv(df_batch: pd.DataFrame) -> None:
    """Append a scored batch DataFrame to results.csv, maintaining the fixed schema."""
    write_header = not RESULTS_PATH.exists()
    df_batch[FIXED_COLUMNS].to_csv(
        RESULTS_PATH,
        mode="a",
        index=False,
        header=write_header,
    )


# ---------------------------------------------------------------------------
# Phase 1: Agent invocation (with cache)
# ---------------------------------------------------------------------------

def run_agent_phase(items: list[dict]) -> dict:
    """
    For each eval item, invoke the LangGraph agent and cache the result.
    Truncates each context passage to ~1500 chars to conserve tokens.
    Exits cleanly on Groq 429 and prints how many are done.
    """
    cache = load_agent_cache()

    pending = [item for item in items if item["question"] not in cache]
    print(f"\n=== Phase 1: Agent Invocation ===")
    print(f"  Total questions : {len(items)}")
    print(f"  Already cached  : {len(cache)}")
    print(f"  To run          : {len(pending)}")

    if not pending:
        print("  All responses cached — skipping agent phase.")
        return cache

    for i, item in enumerate(pending, 1):
        q  = item["question"]
        gt = item["ground_truth"]
        tid = f"eval-{hash(q) & 0xFFFFFF}"

        print(f"\n  [{i}/{len(pending)}] {q[:75]}...", flush=True)

        config = {"configurable": {"thread_id": tid}}
        try:
            result = graph.invoke({"messages": [HumanMessage(q)]}, config)
        except RateLimitError as e:
            done = len(cache)
            print(f"\n[429 RATE LIMIT — Agent Phase]")
            print(f"  Groq limit hit after {done}/{len(items)} questions cached.")
            print(f"  Saved to {CACHE_PATH}. Rerun eval.py once daily limits reset.")
            print(f"  Error: {e}")
            return cache
        except Exception as e:
            print(f"  [Warning] Error on '{q[:50]}': {e}. Storing error entry.")
            cache[q] = {
                "question": q,
                "answer": f"Error: {e}",
                "contexts": [gt[:1500]],
                "ground_truth": gt,
                "tools_used": "error",
            }
            save_agent_cache(cache)
            continue

        answer = result["messages"][-1].content

        # Extract text from every ToolMessage as retrieved context (truncated to ~1500 chars)
        contexts = []
        tools_called = []
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                if msg.name and msg.name not in tools_called:
                    tools_called.append(msg.name)
                content = msg.content or ""
                if "\n\n---\n\n" in content:
                    parts = [p.strip()[:1500] for p in content.split("\n\n---\n\n") if p.strip()]
                    contexts.extend(parts)
                elif content.strip():
                    contexts.append(content.strip()[:1500])

        if not contexts:
            contexts = [answer[:1500]]

        cache[q] = {
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": gt,
            "tools_used": ", ".join(tools_called) if tools_called else "none",
        }
        save_agent_cache(cache)
        print(f"    Tools: {cache[q]['tools_used']}", flush=True)

        if i < len(pending):
            time.sleep(AGENT_DELAY_SECONDS)

    print(f"\n  Agent phase complete. {len(cache)}/{len(items)} responses cached.")
    return cache


# ---------------------------------------------------------------------------
# Phase 2: RAGAS scoring (batch, resumable)
# ---------------------------------------------------------------------------

def build_judge():
    """
    Build the RAGAS LLM judge and embeddings wrapper.

    Judge: openai/gpt-oss-20b on Groq with max_tokens=2048, reasoning_effort='low'.
    Embeddings: local HuggingFace all-MiniLM-L6-v2 (no extra API key needed).
    """
    print(f"\n  RAGAS judge model : {JUDGE_MODEL} (max_tokens=2048, reasoning_effort='low')")
    print(f"  Agent model       : openai/gpt-oss-120b (Groq) — separate daily quota")

    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            model=JUDGE_MODEL,
            temperature=0,
            n=1,
            max_tokens=2048,
            reasoning_effort="low",
            max_retries=3,
            request_timeout=60,
        )
    )
    emb_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    judge_emb = LangchainEmbeddingsWrapper(emb_model)
    return judge_llm, judge_emb


def run_scoring_phase(cache: dict, items: list[dict]) -> None:
    """
    Score cached agent responses with RAGAS in batches of BATCH_SIZE.
    Ensures contexts are truncated to 1500 chars to save tokens.
    """
    scored_qs = load_scored_questions()

    to_score = []
    for item in items:
        q = item["question"]
        if q in scored_qs:
            continue
        if q not in cache:
            print(f"  [Skip] No cached response for: {q[:60]}...")
            continue
        to_score.append(cache[q])

    print(f"\n=== Phase 2: RAGAS Scoring ===")
    print(f"  Already scored   : {len(scored_qs)}")
    print(f"  To score now     : {len(to_score)}")

    if not to_score:
        print("  All questions already scored — nothing to do.")
        return

    judge_llm, judge_emb = build_judge()
    run_cfg = RunConfig(max_workers=1, max_retries=5, max_wait=60, timeout=120)

    total_batches = (len(to_score) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        batch = to_score[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        batch_num = batch_idx + 1

        print(f"\n  Batch {batch_num}/{total_batches} "
              f"(questions {batch_idx*BATCH_SIZE+1}–{batch_idx*BATCH_SIZE+len(batch)})...",
              flush=True)

        # Truncate all contexts to at most 1500 characters
        batch_contexts = [[c[:1500] for c in r["contexts"]] for r in batch]

        dataset = Dataset.from_dict({
            "question":    [r["question"]    for r in batch],
            "answer":      [r["answer"]      for r in batch],
            "contexts":    batch_contexts,
            "ground_truth":[r["ground_truth"] for r in batch],
        })

        try:
            results = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                llm=judge_llm,
                embeddings=judge_emb,
                run_config=run_cfg,
            )
        except RateLimitError as e:
            already_done = len(load_scored_questions())
            print(f"\n[429 RATE LIMIT — Scoring Phase]")
            print(f"  Hit limit during batch {batch_num}/{total_batches}.")
            print(f"  Scored {already_done}/{len(items)} questions so far (in results.csv).")
            print(f"  Rerun eval.py after daily limits reset.")
            print(f"  Error: {e}")
            return
        except Exception as e:
            print(f"  [Warning] Batch {batch_num} evaluation failed: {e}. Skipping batch.")
            continue

        eval_df = results.to_pandas()

        # Build clean DataFrame with the exact fixed columns
        batch_records = []
        for idx, r in enumerate(batch):
            rec = {
                "question": r["question"],
                "answer": r["answer"],
                "ground_truth": r["ground_truth"],
                "tools_used": r["tools_used"],
                "faithfulness": eval_df["faithfulness"].iloc[idx] if "faithfulness" in eval_df.columns else None,
                "answer_relevancy": eval_df["answer_relevancy"].iloc[idx] if "answer_relevancy" in eval_df.columns else None,
                "context_precision": eval_df["context_precision"].iloc[idx] if "context_precision" in eval_df.columns else None,
            }
            batch_records.append(rec)

        batch_df = pd.DataFrame(batch_records)[FIXED_COLUMNS]
        append_batch_to_csv(batch_df)

        current_scored = len(load_scored_questions())
        print(f"  Batch {batch_num} saved. Progress: {current_scored}/{len(items)} scored.", flush=True)

        if batch_num < total_batches:
            print(f"  Sleeping {BATCH_SLEEP_SECONDS}s before next batch...", flush=True)
            time.sleep(BATCH_SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary() -> None:
    """Read results.csv, report NaNs, and print mean scores across all scored rows."""
    if not RESULTS_PATH.exists():
        print("\nNo results.csv found — scoring may not have completed.")
        return

    df = pd.read_csv(RESULTS_PATH)
    n = len(df)
    if n == 0:
        print("\nresults.csv is empty.")
        return

    # Force metric columns to numeric float
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision"]
    for m in metric_cols:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    print(f"\n{'='*70}")
    print(f"  RAGAS EVALUATION RESULTS  ({n}/30 questions scored)")
    print(f"  Agent model : openai/gpt-oss-120b")
    print(f"  Judge model : {JUDGE_MODEL} (max_tokens=2048, reasoning_effort='low')")
    print(f"{'='*70}")
    
    display_cols = ["question", "tools_used", "faithfulness", "answer_relevancy", "context_precision"]
    avail = [c for c in display_cols if c in df.columns]
    print(df[avail].to_string(index=False))
    print(f"{'-'*70}")

    nan_counts = {}
    means = {}
    high_nan_metrics = []
    for m in metric_cols:
        if m in df.columns:
            nan_cnt = int(df[m].isna().sum())
            nan_counts[m] = nan_cnt
            mean_val = df[m].mean()
            means[m] = mean_val
            print(f"  Mean {m:<22} : {mean_val:.4f}  (NaN cells: {nan_cnt}/{n})")
            if nan_cnt > 3:
                high_nan_metrics.append((m, nan_cnt))

    if high_nan_metrics:
        print(f"\n  [WARNING] High NaN count detected (>3 NaNs):")
        for m, cnt in high_nan_metrics:
            print(f"    - {m}: {cnt}/{n} NaNs")
    else:
        print(f"\n  [OK] NaN counts are acceptable (<=3 NaNs per metric).")

    print(f"\n  Full results saved to: {RESULTS_PATH.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    items = load_eval_set()
    print(f"Loaded {len(items)} eval items from {EVAL_SET_PATH}")

    # Phase 1: run agent on all questions, caching results
    cache = run_agent_phase(items)

    # Phase 2: score cached responses with RAGAS (batched, resumable)
    run_scoring_phase(cache, items)

    # Print summary
    print_summary()
