"""
Team 19 - DocPrompting REST API
================================
FastAPI server expose 3 endpoints cho DocPrompting pipeline.

Chạy server:
    # Trên Colab (sau khi đã symlink data/ và models/)
    cd /content/team19-docprompting
    PYTHONPATH=/content/team19-docprompting uvicorn team19_generator_pipeline_api:app --host 0.0.0.0 --port 8000

    # Hoặc dùng script khởi động kèm theo:
    bash start_api.sh

Endpoints:
    POST /retrieve-queries          → retrieve top-k docs cho list queries
    POST /generate-codes            → sinh code từ list doc prompts (FiD format)
    POST /generate-pipeline-codes   → full pipeline: queries → code
    GET  /health                    → health check
    GET  /docs                      → Swagger UI (tự động bởi FastAPI)
"""

import os
import sys
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ────────────────────── Pydantic Models ──────────────────────────

# --- Request models ---

class RetrieveRequest(BaseModel):
    """Request body cho /retrieve-queries"""
    queries: List[str] = Field(..., description="Danh sách câu hỏi NL", min_length=1)
    top_k: int = Field(default=10, description="Số docs trả về cho mỗi query", ge=1, le=200)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "queries": [
                        "sort a list of dictionaries by a value of the dictionary in python",
                        "how to download a file from http url in python"
                    ],
                    "top_k": 10
                }
            ]
        }
    }


class DocContext(BaseModel):
    """Một context document (dùng cho FiD input)"""
    title: str = Field(default="", description="ID hoặc tiêu đề tài liệu")
    text: str = Field(default="", description="Nội dung tài liệu")
    score: float = Field(default=1.0, description="Relevance score")


class FiDInput(BaseModel):
    """Một item input cho FiD generator"""
    question: str = Field(..., description="Câu hỏi NL")
    ctxs: List[DocContext] = Field(..., description="Danh sách context docs")


class GenerateRequest(BaseModel):
    """Request body cho /generate-codes"""
    doc_prompts: List[FiDInput] = Field(..., description="Danh sách FiD inputs", min_length=1)
    n_context: int = Field(default=10, description="Số context sử dụng", ge=1)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "doc_prompts": [
                        {
                            "question": "sort a list of dictionaries by a value of the dictionary",
                            "ctxs": [
                                {
                                    "title": "sorted",
                                    "text": "sorted(iterable, *, key=None, reverse=False) Return a new sorted list.",
                                    "score": 0.85
                                }
                            ]
                        }
                    ],
                    "n_context": 10
                }
            ]
        }
    }


class PipelineRequest(BaseModel):
    """Request body cho /generate-pipeline-codes"""
    queries: List[str] = Field(..., description="Danh sách câu hỏi NL", min_length=1)
    top_k: int = Field(default=10, description="Số docs retrieve", ge=1, le=200)
    n_context: int = Field(default=10, description="Số context cho FiD", ge=1)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "queries": [
                        "sort a list of dictionaries by a value of the dictionary in python",
                        "how to download a file from http url in python",
                        "convert a string to datetime in python"
                    ],
                    "top_k": 10,
                    "n_context": 10
                }
            ]
        }
    }


# --- Response models ---

class RetrievedDoc(BaseModel):
    rank: int
    doc_id: str
    text: str
    score: float


class RetrieveResultItem(BaseModel):
    query: str
    retrieved_docs: List[RetrievedDoc]


class RetrieveResponse(BaseModel):
    results: List[RetrieveResultItem]
    total_time_seconds: float


class GenerateResultItem(BaseModel):
    question: str
    generated_code: str


class GenerateResponse(BaseModel):
    results: List[GenerateResultItem]
    total_time_seconds: float


class PipelineResultItem(BaseModel):
    query: str
    retrieved_docs: List[dict]
    generated_code: str
    retrieval_time: float
    generation_time: float


class PipelineResponse(BaseModel):
    results: List[PipelineResultItem]
    total_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    retriever_loaded: bool
    generator_loaded: bool
    device: str


# ────────────────────── FastAPI App ──────────────────────────────

app = FastAPI(
    title="Team 19 - DocPrompting API",
    description=(
        "REST API cho hệ thống sinh code tự động DocPrompting.\n\n"
        "**Pipeline:** NL Query → Dense Retriever → Doc Prompts → FiD Generator → Code\n\n"
        "Dựa trên paper: *DocPrompting: Generating Code by Retrieving the Docs* (Zhou et al., 2022)"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state: models được load 1 lần khi startup ──
retriever = None
generator = None
pipeline_obj = None


@app.on_event("startup")
async def load_models():
    """Load tất cả models khi server khởi động."""
    global retriever, generator, pipeline_obj

    print("\n🚀 Đang khởi động server và load models...")

    # Đọc cấu hình từ env (hoặc dùng mặc định)
    use_cpu = os.environ.get("USE_CPU", "").lower() in ("1", "true", "yes")
    top_k = int(os.environ.get("DEFAULT_TOP_K", "10"))
    n_context = int(os.environ.get("DEFAULT_N_CONTEXT", "10"))
    batch_size = int(os.environ.get("BATCH_SIZE", "2"))

    from team19_generator_pipeline import DocPromptingPipeline

    pipeline_obj = DocPromptingPipeline(
        top_k=top_k,
        n_context=n_context,
        batch_size=batch_size,
        use_cpu=use_cpu,
    )
    retriever = pipeline_obj.retriever
    generator = pipeline_obj.generator

    print("\n✅ Server sẵn sàng! Truy cập /docs để xem Swagger UI.\n")


# ────────────────────── Endpoints ────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Kiểm tra trạng thái server và models."""
    return HealthResponse(
        status="ok" if retriever and generator else "loading",
        retriever_loaded=retriever is not None,
        generator_loaded=generator is not None,
        device=str(generator.device) if generator else "unknown",
    )


@app.post("/retrieve-queries", response_model=RetrieveResponse, tags=["Retriever"])
async def retrieve_queries(req: RetrieveRequest):
    """
    **Dense Retrieval**: Tìm top-k tài liệu API Python liên quan nhất cho mỗi câu hỏi.

    - Input: danh sách câu hỏi NL
    - Output: top-k docs với doc_id, text, score
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever chưa sẵn sàng")

    t0 = time.time()
    raw = retriever.retrieve(req.queries, top_k=req.top_k)
    elapsed = time.time() - t0

    results = []
    for item in raw:
        docs = [
            RetrievedDoc(
                rank=d["rank"],
                doc_id=d["doc_id"],
                text=d["text"],
                score=d["score"],
            )
            for d in item["retrieved_docs"]
        ]
        results.append(RetrieveResultItem(query=item["query"], retrieved_docs=docs))

    return RetrieveResponse(results=results, total_time_seconds=round(elapsed, 3))


@app.post("/generate-codes", response_model=GenerateResponse, tags=["Generator"])
async def generate_codes(req: GenerateRequest):
    """
    **FiD Code Generation**: Sinh code từ doc prompts đã chuẩn bị sẵn (FiD format).

    - Input: danh sách {question, ctxs: [{title, text, score}]}
    - Output: generated code cho mỗi câu hỏi
    """
    if generator is None:
        raise HTTPException(status_code=503, detail="Generator chưa sẵn sàng")

    t0 = time.time()
    # Chuyển Pydantic → dict cho generator
    fid_inputs = [dp.model_dump() for dp in req.doc_prompts]
    raw = generator.generate(fid_inputs, n_context=req.n_context)
    elapsed = time.time() - t0

    results = [
        GenerateResultItem(question=r["question"], generated_code=r["generated_code"])
        for r in raw
    ]
    return GenerateResponse(results=results, total_time_seconds=round(elapsed, 3))


@app.post("/generate-pipeline-codes", response_model=PipelineResponse, tags=["Pipeline"])
async def generate_pipeline_codes(req: PipelineRequest):
    """
    **Full Pipeline**: NL Query → Dense Retriever → FiD Generator → Code.

    - Input: chỉ cần danh sách câu hỏi NL
    - Output: code sinh ra + top docs đã retrieve + timing
    """
    if pipeline_obj is None:
        raise HTTPException(status_code=503, detail="Pipeline chưa sẵn sàng")

    t0 = time.time()
    raw = pipeline_obj.run(req.queries, top_k=req.top_k, n_context=req.n_context)
    elapsed = time.time() - t0

    results = [
        PipelineResultItem(
            query=r["query"],
            retrieved_docs=r["retrieved_docs"],
            generated_code=r["generated_code"],
            retrieval_time=r["retrieval_time"],
            generation_time=r["generation_time"],
        )
        for r in raw
    ]
    return PipelineResponse(results=results, total_time_seconds=round(elapsed, 3))


# ────────────────────── Main ─────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Team 19 - DocPrompting API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    if args.cpu:
        os.environ["USE_CPU"] = "1"

    uvicorn.run(app, host=args.host, port=args.port)
