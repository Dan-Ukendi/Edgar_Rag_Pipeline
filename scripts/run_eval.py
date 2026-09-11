"""Run the eval set end-to-end: retrieve+generate for each question, score
retrieval/groundedness/relevance/correctness, write results, print a summary.

Usage:
    uv run scripts/run_eval.py [--backend=groq|local] [--limit=N]
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from tenk_rag.config import PipelineConfig
from tenk_rag.eval.scorer import score_question
from tenk_rag.service import RagService


def main() -> None:
    backend_override = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--backend=")), None)
    limit_override = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--limit=")), None)

    config = PipelineConfig.from_yaml("configs/pipeline.yaml")
    backend = backend_override or config.llm.backend
    service = RagService.build(config)

    eval_set = json.loads(Path("eval/eval_set.json").read_text(encoding="utf-8"))
    pairs = eval_set["pairs"]
    if limit_override:
        pairs = pairs[:limit_override]

    results = []
    errors = []
    for i, pair in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {pair['id']}: {pair['question'][:70]}")
        try:
            result = score_question(service, pair, backend)
        except Exception as e:
            print(f"    ERROR: {e}")
            errors.append({"id": pair["id"], "error": str(e)})
            continue
        results.append(result)
        print(
            f"    mode={result.mode} retrieval_hit={result.retrieval_hit} "
            f"faithful={result.faithful} relevant={result.relevant} correct={result.correct}"
        )

    if errors:
        print(f"\n{len(errors)} question(s) failed and were skipped: {[e['id'] for e in errors]}")

    out_path = Path("eval/results.json")
    out_path.write_text(
        json.dumps({"backend": backend, "results": [asdict(r) for r in results], "errors": errors}, indent=2),
        encoding="utf-8",
    )

    n = len(results)
    retrieval_applicable = [r for r in results if r.retrieval_applicable]
    n_retrieval_hit = sum(1 for r in retrieval_applicable if r.retrieval_hit)
    n_faithful = sum(1 for r in results if r.faithful)
    n_relevant = sum(1 for r in results if r.relevant)
    n_correct = sum(1 for r in results if r.correct)

    print(f"\n=== Summary ({backend} backend, {n} questions) ===")
    if retrieval_applicable:
        pct = 100 * n_retrieval_hit / len(retrieval_applicable)
        skipped = n - len(retrieval_applicable)
        print(f"Retrieval hit rate: {n_retrieval_hit}/{len(retrieval_applicable)} ({pct:.0f}%)  [{skipped} questions had no item-level ground truth]")
    print(f"Faithful (grounded): {n_faithful}/{n} ({100 * n_faithful / n:.0f}%)")
    print(f"Relevant:            {n_relevant}/{n} ({100 * n_relevant / n:.0f}%)")
    print(f"Correct:             {n_correct}/{n} ({100 * n_correct / n:.0f}%)")
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
