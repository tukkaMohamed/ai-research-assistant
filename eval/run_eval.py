"""
Evaluate the RAG pipeline on eval/questions.json.

Usage (run from the project root):
    python eval/run_eval.py --retrieval-only   # no LLM calls, tests retrieval only
    python eval/run_eval.py                    # retrieval + answer check
    python eval/run_eval.py --judge            # also ask the LLM to grade each answer
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

# Allow `from core...` when running `python eval/run_eval.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.retriever import Retriever
from core.llm import get_llm
from core.rag import ask

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_PATH = Path(__file__).parent / "results.json"
REFUSAL = "could not find enough information"


def clean(text):
    return " ".join(text.lower().split())


def contains_any(text, keywords):
    text = clean(text)
    return any(clean(k) in text for k in keywords)


def source_recall(expected, retrieved_sources):
    """Fraction of the expected papers that appear in the retrieved chunks."""
    if not expected:
        return None
    got = {s.lower() for s in retrieved_sources}
    return sum(e.lower() in got for e in expected) / len(expected)


def llm_judge(llm, question, reference, answer):
    prompt = f"""You are grading an answer from a research assistant.

Question: {question}
Reference answer: {reference}
Assistant answer: {answer}

Does the assistant answer agree with the reference answer on the key facts?
Reply with exactly one word: CORRECT or INCORRECT."""
    return llm.generate(prompt).strip().upper().startswith("CORRECT")


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(v):
    return "  - " if v is None else f"{v:5.0%}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--sleep", type=float, default=4.0,
                        help="seconds between LLM calls (free-tier rate limits)")
    parser.add_argument("--diverse", action="store_true",
                        help="max 3 chunks per paper (helps cross-paper questions)")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    retriever = Retriever()
    retriever.load()
    llm = None if args.retrieval_only else get_llm()

    rows = []
    for q in questions:
        row = {"id": q["id"], "type": q["type"], "question": q["question"]}

        try:
            mps = 3 if args.diverse else None
            t0 = time.time()
            if args.retrieval_only:
                sources = retriever.search(q["question"], top_k=args.top_k, max_per_source=mps)
                answer = None
            else:
                out = ask(q["question"], retriever, llm, top_k=args.top_k, max_per_source=mps)
                sources, answer = out["sources"], out["answer"]
                row["latency_s"] = round(time.time() - t0, 2)
        except Exception as e:
            row["error"] = str(e)
            rows.append(row)
            print(f"{q['id']}  ERROR: {e}")
            continue

        # ---- Retrieval metrics ----
        row["retrieved"] = [f"{s['source']} p.{s['page']}" for s in sources]
        row["source_recall"] = source_recall(
            q["expected_sources"], [s["source"] for s in sources]
        )
        if q["keywords_any"]:
            row["keyword_in_chunks"] = float(
                any(contains_any(s["text"], q["keywords_any"]) for s in sources)
            )
        else:
            row["keyword_in_chunks"] = None

        # ---- Answer metrics ----
        if answer is not None:
            row["answer"] = answer
            if q["type"] == "out_of_scope":
                row["answer_ok"] = float(REFUSAL in clean(answer))
            elif q["keywords_any"]:
                row["answer_ok"] = float(contains_any(answer, q["keywords_any"]))
            else:
                row["answer_ok"] = None

            row["has_citation"] = float(bool(re.search(r"\[\d+\]", answer)))

            if args.judge and q["type"] != "out_of_scope":
                try:
                    row["judge_ok"] = float(
                        llm_judge(llm, q["question"], q["reference_answer"], answer)
                    )
                except Exception as e:
                    row["judge_ok"] = None
                    row["judge_error"] = str(e)

            time.sleep(args.sleep)

        rows.append(row)
        print(
            f"{q['id']}  {q['type']:<12} "
            f"src={fmt(row['source_recall'])}  chunk_kw={fmt(row['keyword_in_chunks'])}  "
            f"answer={fmt(row.get('answer_ok'))}  judge={fmt(row.get('judge_ok'))}"
        )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"SUMMARY  (top_k={args.top_k}, questions={len(rows)})")
    print("=" * 60)
    for label, subset in [
        ("single-paper", [r for r in rows if r["type"] == "single"]),
        ("cross-paper", [r for r in rows if r["type"] == "cross"]),
        ("out-of-scope", [r for r in rows if r["type"] == "out_of_scope"]),
    ]:
        if not subset:
            continue
        print(
            f"{label:<13} "
            f"source_recall={fmt(mean([r.get('source_recall') for r in subset]))}  "
            f"chunk_kw={fmt(mean([r.get('keyword_in_chunks') for r in subset]))}  "
            f"answer_ok={fmt(mean([r.get('answer_ok') for r in subset]))}  "
            f"cited={fmt(mean([r.get('has_citation') for r in subset]))}  "
            f"judge={fmt(mean([r.get('judge_ok') for r in subset]))}"
        )
    lat = mean([r.get("latency_s") for r in rows])
    if lat is not None:
        print(f"avg latency: {lat:.2f}s per question")

    tag = "retrieval" if args.retrieval_only else (llm.name if llm else "run")
    out_path = RESULTS_PATH.with_name(f"results_{tag}.json")
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()