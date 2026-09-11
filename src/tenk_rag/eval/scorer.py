from dataclasses import dataclass

from ..service import RagService
from .ground_truth import parse_expected_source
from .judge import judge_answer


@dataclass
class QuestionResult:
    id: str
    question: str
    type: str
    difficulty: str
    mode: str
    answer: str
    expected_pairs: list
    retrieved_pairs: list
    retrieval_applicable: bool
    retrieval_hit: bool | None
    retrieval_recall: float | None
    faithful: bool
    relevant: bool
    correct: bool
    judge_reasoning: str


def score_question(service: RagService, pair: dict, backend: str) -> QuestionResult:
    result = service.ask(pair["question"], ticker=None, backend=backend)

    expected_pairs = parse_expected_source(pair["doc"], pair["expected_source"])
    retrieved_pairs = [(e["ticker"], e["item_section"]) for e in result.excerpts]

    retrieval_applicable = bool(expected_pairs)
    retrieval_hit = None
    retrieval_recall = None
    if retrieval_applicable:
        expected_set = set(expected_pairs)
        overlap = expected_set & set(retrieved_pairs)
        retrieval_hit = len(overlap) > 0
        retrieval_recall = len(overlap) / len(expected_set)

    judged = judge_answer(pair["question"], result.excerpts, result.answer, pair["gold_answer"])

    return QuestionResult(
        id=pair["id"],
        question=pair["question"],
        type=pair["type"],
        difficulty=pair["difficulty"],
        mode=result.mode,
        answer=result.answer,
        expected_pairs=expected_pairs,
        retrieved_pairs=retrieved_pairs,
        retrieval_applicable=retrieval_applicable,
        retrieval_hit=retrieval_hit,
        retrieval_recall=retrieval_recall,
        faithful=bool(judged["faithful"]),
        relevant=bool(judged["relevant"]),
        correct=bool(judged["correct"]),
        judge_reasoning=judged.get("reasoning", ""),
    )
