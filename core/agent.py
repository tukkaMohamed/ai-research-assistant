"""Router agent: the LLM decides WHICH chain/tool to run for the user's request."""
import math
import re
from pathlib import Path

from core import chains
from core.parsers import RouterDecision, parse_with_retry
from core.prompts import ROUTER_PROMPT
from core.rag import ask

INTENTS = ["qa", "summarize", "extract", "compare", "gaps", "review"]


def _tokens(source):
    stem = re.sub(r"^\d+[_\-\s]*", "", Path(source).stem.lower())
    return [t for t in re.split(r"[_\-\s]+", stem) if t]


def _pattern(source):
    toks = _tokens(source)
    return r"(?<![a-z0-9])" + r"[\s_\-]?".join(map(re.escape, toks)) + r"(?![a-z0-9])"


def find_papers_in_text(text, available):
    """Rule-based fallback: match paper names inside the question (longest names first)."""
    work, found = text.lower(), []
    ordered = sorted(available, key=lambda s: len("".join(_tokens(s))), reverse=True)
    for src in ordered:
        if not _tokens(src):
            continue
        m = re.search(_pattern(src), work)
        if m:
            found.append((m.start(), src))
            work = work[: m.start()] + " " * (m.end() - m.start()) + work[m.end() :]
    return [s for _, s in sorted(found)]


def _resolve(names, available):
    lower = {a.lower(): a for a in available}
    out = []
    for n in names:
        key = n.strip().lower()
        hit = lower.get(key) or lower.get(key + ".pdf")
        if not hit:
            matches = find_papers_in_text(n, available)
            hit = matches[0] if matches else None
        if hit and hit not in out:
            out.append(hit)
    return out


def heuristic_intent(query, n_papers):
    q = query.lower()
    if re.search(r"\b(compare|comparison|versus|vs\.?|differences?|differ)\b", q) and n_papers >= 2:
        return "compare"
    if "summar" in q and n_papers >= 1:
        return "summarize"
    if re.search(r"literature review|overview of", q):
        return "review"
    if re.search(r"limitation|research gap|future work|weakness", q):
        return "gaps"
    if re.search(r"datasets?|metrics?|methodology|extract", q) and n_papers == 1:
        return "extract"
    return "qa"


def route(query, llm, available):
    """Returns (intent, papers, aspect, routed_by)."""
    try:
        prompt = ROUTER_PROMPT.format(
            papers="\n".join(f"- {a}" for a in available), question=query
        )
        d = parse_with_retry(llm, prompt, RouterDecision, retries=1)
        papers = _resolve(d.papers, available) or find_papers_in_text(query, available)
        return d.intent, papers, d.aspect, "llm"
    except Exception:
        papers = find_papers_in_text(query, available)
        return heuristic_intent(query, len(papers)), papers, None, "rules"


def run_agent(query, retriever, llm, top_k=5, force_intent=None):
    available = retriever.sources()
    if force_intent:
        intent, papers, aspect, via = force_intent, find_papers_in_text(query, available), None, "manual"
    else:
        intent, papers, aspect, via = route(query, llm, available)

    # Safety nets: a chain that needs papers falls back to plain Q&A
    if intent == "compare" and len(papers) < 2:
        intent = "qa"
    if intent in ("summarize", "extract") and not papers:
        intent = "qa"

    out = {"intent": intent, "papers": papers, "routed_by": via, "table": None, "sources": []}

    if intent == "summarize":
        parts = []
        for p in papers[:3]:
            r = chains.summarize_paper(p, retriever, llm)
            parts.append(f"### {p}\n{r['summary']}")
            out["sources"] += r["sources"]
        out["answer"] = "\n\n".join(parts)

    elif intent == "extract":
        infos = {p: chains.extract_paper(p, retriever, llm) for p in papers[:3]}
        out["answer"] = "\n\n---\n\n".join(
            f"### {p}\n{chains.format_info(i)}" for p, i in infos.items()
        )
        out["table"] = chains.info_rows(infos)

    elif intent == "compare":
        r = chains.compare_papers(papers, retriever, llm, aspect)
        out["answer"], out["table"] = r["analysis"], r["table"]

    elif intent in ("gaps", "review") and papers:
        fn = chains.research_gaps if intent == "gaps" else chains.literature_review
        r = fn(papers, retriever, llm)
        out["answer"], out["table"] = r["analysis"], r["table"]

    elif intent in ("gaps", "review"):
        # No papers named: retrieve one best excerpt per paper from the whole library
        hint = "limitations, weaknesses, future work and open problems" if intent == "gaps" else "main contribution and approach"
        r = ask(query, retriever, llm, top_k=8, max_per_source=1, search_query=f"{hint} {query}")
        out["answer"], out["sources"] = r["answer"], r["sources"]

    else:  # qa
        if papers:
            k = min(12, max(top_k, 4 * len(papers)))
            r = ask(query, retriever, llm, top_k=k, sources=papers,
                    max_per_source=math.ceil(k / len(papers)))
        else:
            r = ask(query, retriever, llm, top_k=top_k, max_per_source=3)
        out["answer"], out["sources"] = r["answer"], r["sources"]

    return out