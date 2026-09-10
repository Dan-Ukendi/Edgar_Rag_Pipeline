"""Ask a question end-to-end: retrieve chunks, generate a grounded, cited answer.

Usage:
    uv run scripts/ask.py "What are Apple's main risk factors?" [TICKER] [--backend groq|local]
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from tenk_rag.config import PipelineConfig
from tenk_rag.service import RagService


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit('Usage: uv run scripts/ask.py "<question>" [TICKER] [--backend groq|local]')
    question = args[0]
    ticker = args[1] if len(args) > 1 else None

    backend_override = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--backend=")), None)

    config = PipelineConfig.from_yaml("configs/pipeline.yaml")
    service = RagService.build(config)
    result = service.ask(question, ticker=ticker, backend=backend_override)

    print(f"\nQuestion: {result.question}")
    print(f"Backend: {result.backend} | Mode: {result.mode}\n")
    print(result.answer)
    print(f"\nCited excerpts: {result.cited_indices or 'none'}")
    print("\nRetrieved excerpts:")
    for i, ex in enumerate(result.excerpts, 1):
        cited = "cited" if i in result.cited_indices else "not cited"
        print(f"  [{i}] {ex['ticker']} | {ex['filing_date']} | {ex['item_section']} | {cited}")


if __name__ == "__main__":
    main()
