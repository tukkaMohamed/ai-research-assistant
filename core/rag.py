from core.prompts import QA_PROMPT, RULES

NOT_FOUND = "I could not find enough information in the provided papers."


def build_context(results):
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] ({r['source']}, page {r['page']})\n{r['text']}")
    return "\n\n---\n\n".join(blocks)


def ask(query, retriever, llm, top_k=5, sources=None, max_per_source=None, search_query=None):
    results = retriever.search(
        search_query or query, top_k=top_k, sources=sources, max_per_source=max_per_source
    )
    if not results:
        return {"answer": NOT_FOUND, "sources": []}
    prompt = QA_PROMPT.format(rules=RULES, question=query, context=build_context(results))
    return {"answer": llm.generate(prompt), "sources": results}