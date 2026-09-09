from fastapi import APIRouter, HTTPException, Request

from .schemas import ExcerptOut, HealthResponse, QueryRequest, QueryResponse, VectorStoreHealth

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    service = request.app.state.service
    try:
        result = service.ask(
            question=body.question,
            ticker=body.ticker,
            top_k=body.top_k,
            backend=body.backend,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    excerpts = [
        ExcerptOut(
            index=i,
            ticker=ex["ticker"],
            filing_date=ex["filing_date"],
            item_section=ex["item_section"],
            text=ex["text"],
            cited=i in result.cited_indices,
        )
        for i, ex in enumerate(result.excerpts, 1)
    ]

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        backend=result.backend,
        cited_indices=result.cited_indices,
        excerpts=excerpts,
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = request.app.state.service

    try:
        info = service.store.collection_info()
        vector_store = VectorStoreHealth(reachable=True, points_count=info["points_count"])
    except Exception:
        vector_store = VectorStoreHealth(reachable=False)

    llm_backends = {"groq": "groq" in service.llm_clients}
    local_client = service.llm_clients.get("local")
    llm_backends["local"] = local_client.is_reachable() if local_client else False

    status = "ok" if vector_store.reachable and any(llm_backends.values()) else "degraded"

    return HealthResponse(
        status=status,
        embedding_model="loaded",
        vector_store=vector_store,
        llm_backends=llm_backends,
    )
