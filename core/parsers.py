"""Output parsers: make the LLM return validated, structured results."""
import json
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator


def _to_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() and v.strip().lower() != "not stated" else []
    return [str(x) for x in v]


def _to_str(v):
    if v is None:
        return "Not stated"
    if isinstance(v, (list, tuple)):
        return "; ".join(map(str, v))
    return str(v)


class PaperInfo(BaseModel):
    title: str = "Not stated"
    year: Optional[int] = None
    task: str = "Not stated"
    methodology: str = "Not stated"
    models: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    key_results: str = "Not stated"
    limitations: List[str] = Field(default_factory=list)

    @field_validator("models", "datasets", "metrics", "limitations", mode="before")
    @classmethod
    def _lists(cls, v):
        return _to_list(v)

    @field_validator("title", "task", "methodology", "key_results", mode="before")
    @classmethod
    def _strings(cls, v):
        return _to_str(v)

    @field_validator("year", mode="before")
    @classmethod
    def _year(cls, v):
        if v is None:
            return None
        m = re.search(r"\b(?:19|20)\d{2}\b", str(v))
        return int(m.group(0)) if m else None


class RouterDecision(BaseModel):
    intent: Literal["qa", "summarize", "extract", "compare", "gaps", "review"]
    papers: List[str] = Field(default_factory=list)
    aspect: Optional[str] = None

    @field_validator("intent", mode="before")
    @classmethod
    def _intent(cls, v):
        return str(v).strip().lower()

    @field_validator("papers", mode="before")
    @classmethod
    def _papers(cls, v):
        return _to_list(v)

    @field_validator("aspect", mode="before")
    @classmethod
    def _aspect(cls, v):
        if v is None or str(v).strip().lower() in ("", "null", "none", "not stated"):
            return None
        return str(v)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (ignores ```json fences and chatter)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in the reply")
    return json.loads(text[start : end + 1])


def parse_with_retry(llm, prompt: str, model_cls, retries: int = 2):
    """Ask the LLM, validate with Pydantic, and re-ask with the error if invalid."""
    last_error = None
    current = prompt
    for _ in range(retries + 1):
        raw = llm.generate(current)
        try:
            return model_cls.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as e:
            last_error = e
            current = (
                prompt
                + f"\n\nYour previous reply was invalid ({str(e)[:200]}). "
                "Reply again with ONLY the valid JSON object."
            )
    raise ValueError(f"Could not get valid JSON from the model: {last_error}")