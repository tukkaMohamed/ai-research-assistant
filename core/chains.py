"""Chains: multi-step workflows built on the RAG engine + output parsers."""
import json
from pathlib import Path

from core import prompts as P
from core.parsers import PaperInfo, parse_with_retry
from core.rag import build_context

CACHE_DIR = Path("data/extractions")

SECTION_QUERIES = [
    "paper title authors publication year",
    "abstract main contribution proposed method",
    "datasets and benchmarks used in the experiments",
    "evaluation metrics and main results",
    "limitations and future work",
]


def _key(c):
    return (c["source"], c["page"], c["text"][:60])


def paper_context(retriever, paper, queries=SECTION_QUERIES, intro_chunks=3, per_query=2, max_chars=9000):
    """Step 1 of every paper-level chain: abstract/intro + best chunk per section, from ONE paper."""
    own = [c for c in retriever.chunks if c["source"] == paper]
    if not own:
        raise ValueError(f"Paper not found in the index: {paper}")
    selected = list(own[:intro_chunks])
    seen = {_key(c) for c in selected}
    for q in queries:
        for r in retriever.search(q, top_k=per_query, sources=[paper]):
            if _key(r) not in seen:
                seen.add(_key(r))
                selected.append(r)
    kept, total = [], 0
    for c in selected:
        if total + len(c["text"]) > max_chars:
            break
        kept.append(c)
        total += len(c["text"])
    return kept


def summarize_paper(paper, retriever, llm):
    chunks = paper_context(retriever, paper)
    prompt = P.SUMMARY_PROMPT.format(rules=P.RULES, paper=paper, context=build_context(chunks))
    return {"summary": llm.generate(prompt), "sources": chunks}


def _cache_path(paper, llm):
    name = f"{Path(paper).stem}__{getattr(llm, 'name', 'llm')}.json".replace(" ", "_")
    return CACHE_DIR / name


def extract_paper(paper, retriever, llm, refresh=False) -> PaperInfo:
    """retrieve -> prompt -> LLM -> JSON -> validated PaperInfo (cached on disk)."""
    path = _cache_path(paper, llm)
    if path.exists() and not refresh:
        return PaperInfo.model_validate_json(path.read_text(encoding="utf-8"))
    chunks = paper_context(retriever, paper)
    prompt = P.EXTRACT_PROMPT.format(
        paper=paper, hint=P.PAPER_JSON_HINT, context=build_context(chunks)
    )
    info = parse_with_retry(llm, prompt, PaperInfo)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(info.model_dump_json(indent=2), encoding="utf-8")
    return info


def format_info(info: PaperInfo) -> str:
    def bullets(items):
        return "\n".join(f"  - {x}" for x in items) if items else "  - Not stated"

    return (
        f"**Title:** {info.title}\n\n"
        f"**Year:** {info.year or 'Not stated'}\n\n"
        f"**Task:** {info.task}\n\n"
        f"**Methodology:** {info.methodology}\n\n"
        f"**Models:**\n{bullets(info.models)}\n\n"
        f"**Datasets:**\n{bullets(info.datasets)}\n\n"
        f"**Metrics:**\n{bullets(info.metrics)}\n\n"
        f"**Key results:** {info.key_results}\n\n"
        f"**Limitations:**\n{bullets(info.limitations)}"
    )


def info_rows(infos: dict):
    """paper -> PaperInfo  ==>  list of dict rows (comparison table)."""
    join = lambda xs: ", ".join(xs) if xs else "Not stated"
    return [
        {
            "Paper": paper,
            "Year": i.year or "-",
            "Task": i.task,
            "Method": i.methodology,
            "Models": join(i.models),
            "Datasets": join(i.datasets),
            "Metrics": join(i.metrics),
            "Key results": i.key_results,
            "Limitations": join(i.limitations),
        }
        for paper, i in infos.items()
    ]


def _infos(papers, retriever, llm):
    return {p: extract_paper(p, retriever, llm) for p in papers}


def _payload(infos):
    return json.dumps({p: i.model_dump() for p, i in infos.items()}, indent=2)


def compare_papers(papers, retriever, llm, aspect=None):
    """extract each paper -> compare the STRUCTURED data (not raw text)."""
    infos = _infos(papers, retriever, llm)
    clause = f" with a focus on {aspect}" if aspect else ""
    analysis = llm.generate(
        P.COMPARE_PROMPT.format(aspect_clause=clause, papers_json=_payload(infos))
    )
    return {"analysis": analysis, "table": info_rows(infos), "infos": infos}


def research_gaps(papers, retriever, llm):
    infos = _infos(papers, retriever, llm)
    analysis = llm.generate(P.GAPS_PROMPT.format(papers_json=_payload(infos)))
    return {"analysis": analysis, "table": info_rows(infos), "infos": infos}


def literature_review(papers, retriever, llm):
    infos = _infos(papers, retriever, llm)
    analysis = llm.generate(P.REVIEW_PROMPT.format(papers_json=_payload(infos)))
    return {"analysis": analysis, "table": info_rows(infos), "infos": infos}