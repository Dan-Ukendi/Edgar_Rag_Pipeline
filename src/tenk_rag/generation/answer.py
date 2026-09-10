import re
from dataclasses import dataclass

from ..llm.base import LLMClient
from .prompt import MULTI_COMPANY_SYSTEM_PROMPT, SYSTEM_PROMPT, build_prompt


@dataclass
class AnswerResult:
    question: str
    answer: str
    excerpts: list[dict]
    cited_indices: list[int]
    backend: str


CITATION_PATTERN = re.compile(r"[\[【](\d+)[\]】]")


def _extract_citations(answer_text: str) -> list[int]:
    return sorted({int(n) for n in CITATION_PATTERN.findall(answer_text)})


def answer_question(
    question: str,
    excerpts: list[dict],
    llm_client: LLMClient,
    backend: str,
    multi_company: bool = False,
) -> AnswerResult:
    prompt = build_prompt(question, excerpts)
    system = MULTI_COMPANY_SYSTEM_PROMPT if multi_company else SYSTEM_PROMPT
    answer_text = llm_client.generate(prompt, system=system)
    return AnswerResult(
        question=question,
        answer=answer_text,
        excerpts=excerpts,
        cited_indices=_extract_citations(answer_text),
        backend=backend,
    )
