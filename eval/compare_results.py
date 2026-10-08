"""
Cloud vs Local comparison table from the files written by run_eval.py.
    python eval/run_eval.py --backend gemini --judge
    python eval/run_eval.py --backend remote --judge
    python eval/compare_results.py
"""


import json
from pathlib import Path

HERE = Path(__file__).parent


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def pct(v):
    return "-" if v is None else f"{v:.0%}"


def main():
    files = sorted(p for p in HERE.glob("results_*.json") if p.stem != "results_retrieval")
    if not files:
        print("No results_*.json found. Run eval/run_eval.py first (without --retrieval-only).")
        return
    print("| Model | Single answer ok | Cross answer ok | Refuses out-of-scope | Judge (non-OOS) | Cited | Avg latency |")
    print("|---|---|---|---|---|---|---|")
    for f in files:
        rows = json.loads(f.read_text(encoding="utf-8"))
        by = lambda t: [r for r in rows if r["type"] == t]
        judge = [r.get("judge_ok") for r in rows if r["type"] != "out_of_scope"]
        lat = mean([r.get("latency_s") for r in rows])
        print(
            f"| {f.stem.replace('results_', '')} "
            f"| {pct(mean([r.get('answer_ok') for r in by('single')]))} "
            f"| {pct(mean([r.get('answer_ok') for r in by('cross')]))} "
            f"| {pct(mean([r.get('answer_ok') for r in by('out_of_scope')]))} "
            f"| {pct(mean(judge))} "
            f"| {pct(mean([r.get('has_citation') for r in rows]))} "
            f"| {'-' if lat is None else f'{lat:.1f}s'} |"
        )

if __name__ == "__main__":
    main()