"""
Team 19 - Full Pipeline: Query → Retrieve Docs → Generate Code
================================================================
Pipeline hoàn chỉnh kết nối Dense Retriever và FiD Code Generator.
Truyền vào câu hỏi NL → tự động retrieve tài liệu liên quan → sinh code.

Sử dụng:
    python team19_generator_pipeline.py                         # demo mặc định
    python team19_generator_pipeline.py --top_k 5 --n_context 5 # tùy chỉnh
    python team19_generator_pipeline.py --cpu                   # chạy trên CPU

Hoặc import như module:
    from team19_generator_pipeline import DocPromptingPipeline
    pipeline = DocPromptingPipeline()
    results = pipeline.run(["sort a list in python"])
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Optional

# ─────────────────────────── Cấu hình mặc định ────────────────────────────
# Retriever defaults
DEFAULT_RETRIEVER_MODEL = "neulab/docprompting-codet5-python-doc-retriever"
DEFAULT_TOKENIZER_FALLBACK = "models/generator/codet5-base"
DEFAULT_TARGET_FILE = "data/conala/python_manual_firstpara.tok.txt"
DEFAULT_TARGET_ID_FILE = "data/conala/python_manual_firstpara.tok.id"
DEFAULT_TARGET_EMBED_CACHE = "data/conala/.tmp/tgt_embedding.npy"
DEFAULT_TOP_K = 10
DEFAULT_NUM_LAYERS = 12

# Generator defaults
DEFAULT_MODEL_PATH = "models/generator/conala.fid.codet5.top10/checkpoint/best_dev/best_dev"
DEFAULT_GEN_TOKENIZER = "models/generator/codet5-base"
DEFAULT_N_CONTEXT = 10
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_LENGTH = 150
DEFAULT_NUM_BEAMS = 10
DEFAULT_LENPEN = 1.0


class DocPromptingPipeline:
    """
    Pipeline hoàn chỉnh của DocPrompting:
      NL Query → Dense Retriever → Doc Prompts → FiD Generator → Code

    Load cả 2 model (retriever + generator) một lần duy nhất,
    sau đó gọi pipeline.run(queries) nhiều lần.
    """

    def __init__(
        self,
        # Retriever params
        retriever_model: str = DEFAULT_RETRIEVER_MODEL,
        tokenizer_fallback: str = DEFAULT_TOKENIZER_FALLBACK,
        target_file: str = DEFAULT_TARGET_FILE,
        target_id_file: str = DEFAULT_TARGET_ID_FILE,
        target_embed_cache: str = DEFAULT_TARGET_EMBED_CACHE,
        top_k: int = DEFAULT_TOP_K,
        num_layers: int = DEFAULT_NUM_LAYERS,
        # Generator params
        generator_model_path: str = DEFAULT_MODEL_PATH,
        generator_tokenizer: str = DEFAULT_GEN_TOKENIZER,
        n_context: int = DEFAULT_N_CONTEXT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        num_beams: int = DEFAULT_NUM_BEAMS,
        lenpen: float = DEFAULT_LENPEN,
        # General
        use_cpu: bool = False,
    ):
        self.top_k = top_k
        self.n_context = n_context

        print("╔" + "═" * 58 + "╗")
        print("║  Team 19 - DocPrompting Full Pipeline                    ║")
        print("║  NL Query → Dense Retriever → FiD Generator → Code      ║")
        print("╚" + "═" * 58 + "╝")

        # ─── Step 1: Load Dense Retriever ───
        print("\n" + "━" * 60)
        print("  BƯỚC 1: Load Dense Retriever")
        print("━" * 60)
        from team19_retriever import DenseRetriever
        self.retriever = DenseRetriever(
            model_name=retriever_model,
            tokenizer_fallback=tokenizer_fallback,
            target_file=target_file,
            target_id_file=target_id_file,
            target_embed_cache=target_embed_cache,
            top_k=top_k,
            num_layers=num_layers,
            use_cpu=use_cpu,
        )

        # ─── Step 2: Load FiD Generator ───
        print("\n" + "━" * 60)
        print("  BƯỚC 2: Load FiD Code Generator")
        print("━" * 60)
        from team19_generator import FiDCodeGenerator
        self.generator = FiDCodeGenerator(
            model_path=generator_model_path,
            tokenizer_name=generator_tokenizer,
            n_context=n_context,
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            lenpen=lenpen,
            use_cpu=use_cpu,
        )

        print("\n✅ Pipeline sẵn sàng! Cả 2 model đã được load.")

    def run(self, queries: list, top_k: int = None, n_context: int = None) -> list:
        """
        Chạy full pipeline: query → retrieve → generate.

        Args:
            queries: List[str] - danh sách câu hỏi NL
            top_k: int - số docs retrieve (mặc định dùng self.top_k)
            n_context: int - số context cho FiD (mặc định dùng self.n_context)

        Returns:
            List[dict] - mỗi dict chứa:
                - query: câu hỏi gốc
                - retrieved_docs: top-k docs tìm được
                - generated_code: code sinh ra
                - retrieval_time: thời gian retrieve (giây)
                - generation_time: thời gian generate (giây)
        """
        if top_k is None:
            top_k = self.top_k
        if n_context is None:
            n_context = self.n_context

        print("\n" + "═" * 60)
        print(f"  🚀 CHẠY PIPELINE: {len(queries)} câu hỏi")
        print(f"     top_k={top_k}, n_context={n_context}")
        print("═" * 60)

        # ─── Stage 1: Retrieve ───
        print("\n📌 Stage 1: Dense Retrieval")
        t0 = time.time()
        fid_inputs = self.retriever.retrieve_as_doc_prompt(queries, top_k=top_k)
        retrieval_time = time.time() - t0
        print(f"   ⏱️  Retrieve xong trong {retrieval_time:.2f}s")

        # ─── Stage 2: Generate ───
        print("\n📌 Stage 2: FiD Code Generation")
        t0 = time.time()
        gen_results = self.generator.generate(fid_inputs, n_context=n_context)
        generation_time = time.time() - t0
        print(f"   ⏱️  Generate xong trong {generation_time:.2f}s")

        # ─── Combine Results ───
        final_results = []
        for query, fid_input, gen_result in zip(queries, fid_inputs, gen_results):
            final_results.append({
                "query": query,
                "retrieved_docs": [
                    {"doc_id": ctx["title"], "text": ctx["text"][:200], "score": ctx["score"]}
                    for ctx in fid_input.get("ctxs", [])[:5]  # chỉ giữ top 5 cho display
                ],
                "generated_code": gen_result["generated_code"],
                "retrieval_time": retrieval_time / len(queries),
                "generation_time": generation_time / len(queries),
            })

        return final_results


# ────────────────────────── CLI Demo ─────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Team 19 - DocPrompting Full Pipeline")
    # Retriever args
    parser.add_argument("--retriever_model", type=str, default=DEFAULT_RETRIEVER_MODEL)
    parser.add_argument("--tokenizer_fallback", type=str, default=DEFAULT_TOKENIZER_FALLBACK)
    parser.add_argument("--target_file", type=str, default=DEFAULT_TARGET_FILE)
    parser.add_argument("--target_id_file", type=str, default=DEFAULT_TARGET_ID_FILE)
    parser.add_argument("--target_embed_cache", type=str, default=DEFAULT_TARGET_EMBED_CACHE)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    # Generator args
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer_name", type=str, default=DEFAULT_GEN_TOKENIZER)
    parser.add_argument("--n_context", type=int, default=DEFAULT_N_CONTEXT)
    parser.add_argument("--per_gpu_batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--num_beams", type=int, default=DEFAULT_NUM_BEAMS)
    # General
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    # ─── Demo queries ───
    demo_queries = [
        "sort a list of dictionaries by a value of the dictionary in python",
        "how to download a file from http url in python",
        "convert a string to datetime in python",
        "read a csv file into a pandas dataframe",
        "how to flatten a nested list in python",
    ]

    # ─── Load Pipeline (1 lần duy nhất) ───
    pipeline = DocPromptingPipeline(
        retriever_model=args.retriever_model,
        tokenizer_fallback=args.tokenizer_fallback,
        target_file=args.target_file,
        target_id_file=args.target_id_file,
        target_embed_cache=args.target_embed_cache,
        top_k=args.top_k,
        num_layers=args.num_layers,
        generator_model_path=args.model_path,
        generator_tokenizer=args.tokenizer_name,
        n_context=args.n_context,
        batch_size=args.per_gpu_batch_size,
        max_length=args.max_length,
        num_beams=args.num_beams,
        use_cpu=args.cpu,
    )

    # ─── Chạy Pipeline Lần 1 ───
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║          DEMO LẦN 1 - 5 câu hỏi                         ║")
    print("╚" + "═" * 58 + "╝")
    results = pipeline.run(demo_queries)
    _print_results(results)

    # ─── Chạy Pipeline Lần 2 (model đã load sẵn) ───
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║          DEMO LẦN 2 - model đã load sẵn                  ║")
    print("╚" + "═" * 58 + "╝")
    extra_queries = [
        "remove duplicates from a list in python",
        "merge two dictionaries in python 3",
    ]
    results2 = pipeline.run(extra_queries, top_k=5, n_context=5)
    _print_results(results2)

    # ─── Lưu kết quả ra file ───
    all_results = results + results2
    output_path = "team19_pipeline_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Đã lưu kết quả vào: {output_path}")


def _print_results(results: list):
    """In kết quả pipeline đẹp."""
    for r in results:
        print(f"\n{'─' * 60}")
        print(f"🔹 NL Query:     {r['query']}")
        print(f"💻 Generated Code: {r['generated_code']}")
        print(f"⏱️  Time: retrieve={r['retrieval_time']:.2f}s, generate={r['generation_time']:.2f}s")
        print(f"📚 Top Retrieved Docs:")
        for i, doc in enumerate(r['retrieved_docs'][:3]):
            print(f"   [{i+1}] {doc['doc_id']} (score: {doc['score']:.4f})")
    print(f"\n{'─' * 60}")


if __name__ == "__main__":
    main()
