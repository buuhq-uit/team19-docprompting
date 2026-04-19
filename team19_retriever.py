"""
Team 19 - Dense Retriever Module
=================================
Load mô hình Dense Retriever (CodeT5-based SimCSE) một lần,
sau đó có thể gọi retrieve(queries) nhiều lần để tìm Top-K tài liệu
liên quan nhất cho mỗi câu hỏi.

Sử dụng:
    python team19_retriever.py                     # chạy demo mặc định
    python team19_retriever.py --top_k 5           # lấy top 5 thay vì top 10

Hoặc import như module:
    from team19_retriever import DenseRetriever
    retriever = DenseRetriever()
    results = retriever.retrieve(["sort a list in python"])
"""

import os
import sys
import json
import argparse

import faiss
import numpy as np
import torch
import transformers
from tqdm import tqdm

# ─────────────────────────── Cấu hình mặc định ────────────────────────────
DEFAULT_RETRIEVER_MODEL = "neulab/docprompting-codet5-python-doc-retriever"
DEFAULT_TOKENIZER_FALLBACK = "models/generator/codet5-base"
DEFAULT_TARGET_FILE = "data/conala/python_manual_firstpara.tok.txt"
DEFAULT_TARGET_ID_FILE = "data/conala/python_manual_firstpara.tok.id"
DEFAULT_TARGET_EMBED_CACHE = "data/conala/.tmp/tgt_embedding.npy"
DEFAULT_TOP_K = 10
DEFAULT_NUM_LAYERS = 12
DEFAULT_SIM_FUNC = "cls_distance.cosine"
DEFAULT_BATCH_SIZE = 128


class Dummy:
    """Dummy object để truyền model_args cho RetrievalModel."""
    pass


class DenseRetriever:
    """
    Dense Retriever dựa trên CodeT5 SimCSE.

    Workflow:
      1. Load model retriever + tokenizer (chỉ 1 lần)
      2. Load & cache target embeddings (tài liệu API Python)
      3. Với mỗi list query → encode → FAISS search → trả về top-k docs
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RETRIEVER_MODEL,
        tokenizer_fallback: str = DEFAULT_TOKENIZER_FALLBACK,
        target_file: str = DEFAULT_TARGET_FILE,
        target_id_file: str = DEFAULT_TARGET_ID_FILE,
        target_embed_cache: str = DEFAULT_TARGET_EMBED_CACHE,
        top_k: int = DEFAULT_TOP_K,
        num_layers: int = DEFAULT_NUM_LAYERS,
        sim_func: str = DEFAULT_SIM_FUNC,
        batch_size: int = DEFAULT_BATCH_SIZE,
        use_cpu: bool = False,
    ):
        self.model_name = model_name
        self.tokenizer_fallback = tokenizer_fallback
        self.target_file = target_file
        self.target_id_file = target_id_file
        self.target_embed_cache = target_embed_cache
        self.top_k = top_k
        self.num_layers = num_layers
        self.sim_func = sim_func
        self.batch_size = batch_size
        self.use_cpu = use_cpu

        self.model = None
        self.tokenizer = None
        self.device = None
        self.target_embed = None
        self.target_id_map = None
        self.target_texts = None
        self.faiss_index = None

        print("=" * 60)
        print("  Team 19 - Dense Retriever (CodeT5 SimCSE)")
        print("=" * 60)
        self._load_model()
        self._load_target_data()

    # ─────────────────────── Load Model ─────────────────────────
    def _load_model(self):
        """Load mô hình retriever và tokenizer."""
        print(f"\n📦 Đang load model: {self.model_name}")
        transformers.logging.set_verbosity_warning()

        # --- Cần import RetrievalModel từ retriever/simcse ---
        # Thêm path để import module gốc
        simcse_dir = os.path.join(os.path.dirname(__file__), "retriever", "simcse")
        if simcse_dir not in sys.path:
            sys.path.insert(0, simcse_dir)

        from model import RetrievalModel
        from transformers import AutoConfig

        # Load tokenizer
        try:
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained(self.model_name)
        except (TypeError, OSError):
            print(f"⚠️  Tokenizer HF Hub lỗi, fallback sang: {self.tokenizer_fallback}")
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained(self.tokenizer_fallback)

        # Load model
        model_arg = Dummy()
        setattr(model_arg, 'sim_func', self.sim_func)
        config = AutoConfig.from_pretrained(self.model_name)
        self.model = RetrievalModel(
            config=config,
            model_type=self.model_name,
            num_layers=self.num_layers,
            tokenizer=self.tokenizer,
            training_args=None,
            model_args=model_arg,
        )

        self.device = torch.device('cpu') if self.use_cpu else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        print(f"✅ Model loaded trên device: {self.device}")

    # ─────────────────── Load Target Data ───────────────────────
    def _load_target_data(self):
        """Load target texts, IDs, và pre-computed embeddings (hoặc tính mới)."""
        print(f"\n📄 Đang load target docs từ: {self.target_file}")

        # Load target texts
        with open(self.target_file, "r", encoding="utf-8") as f:
            self.target_texts = [line.strip() for line in f]
        print(f"   Số lượng tài liệu: {len(self.target_texts)}")

        # Load target IDs
        with open(self.target_id_file, "r", encoding="utf-8") as f:
            self.target_id_map = {}
            for idx, line in enumerate(f):
                self.target_id_map[idx] = line.strip()

        # Load hoặc tính target embeddings
        if os.path.exists(self.target_embed_cache):
            print(f"   ♻️  Sử dụng cache embeddings: {self.target_embed_cache}")
            self.target_embed = np.load(self.target_embed_cache)
        else:
            print(f"   ⏳ Chưa có cache, đang encode {len(self.target_texts)} tài liệu...")
            self.target_embed = self._encode_texts(self.target_texts)
            # Lưu cache
            cache_dir = os.path.dirname(self.target_embed_cache)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            np.save(self.target_embed_cache, self.target_embed)
            print(f"   💾 Đã lưu cache: {self.target_embed_cache}")

        print(f"   Target embeddings shape: {self.target_embed.shape}")

        # Build FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.target_embed.shape[1])
        self.faiss_index.add(self.target_embed)
        print(f"   🔍 FAISS index sẵn sàng ({self.faiss_index.ntotal} vectors)")

    # ──────────────────── Encode Texts ──────────────────────────
    def _encode_texts(self, texts: list) -> np.ndarray:
        """Encode list of texts thành embeddings dùng retriever model."""
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding"):
                batch = texts[i: i + self.batch_size]
                padded_batch = self._pad_batch(batch)
                for k in padded_batch:
                    if isinstance(padded_batch[k], torch.Tensor):
                        padded_batch[k] = padded_batch[k].to(self.device)
                output = self.model.get_pooling_embedding(**padded_batch, normalize=False)
                all_embeddings.append(output.detach().cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def _pad_batch(self, sentences: list) -> dict:
        """Tokenize và padding cho một batch câu."""
        sent_features = self.tokenizer(
            sentences,
            add_special_tokens=True,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        arr = sent_features['input_ids']
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = lens.max().item()
        padded = torch.ones(len(arr), max_len, dtype=torch.long) * self.tokenizer.pad_token_id
        mask = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, a in enumerate(arr):
            padded[i, :lens[i]] = torch.tensor(a, dtype=torch.long)
            mask[i, :lens[i]] = 1
        return {'input_ids': padded, 'attention_mask': mask, 'lengths': lens}

    # ──────────────────── Retrieve ──────────────────────────────
    def retrieve(self, queries: list, top_k: int = None) -> list:
        """
        Tìm top-k tài liệu liên quan nhất cho mỗi query.

        Args:
            queries: List[str] - danh sách câu hỏi NL
            top_k: int - số lượng tài liệu trả về (mặc định dùng self.top_k)

        Returns:
            List[dict] - mỗi dict chứa:
                - query: câu hỏi gốc
                - retrieved_docs: list các dict {doc_id, text, score}
        """
        if top_k is None:
            top_k = self.top_k

        print(f"\n🔎 Đang retrieve cho {len(queries)} câu hỏi (top_k={top_k})...")

        # Encode queries
        query_embeddings = self._encode_texts(queries)

        # FAISS search
        scores, indices = self.faiss_index.search(query_embeddings, top_k)

        # Format kết quả
        results = []
        for q_idx, (query, dist, retrieved_idx) in enumerate(zip(queries, scores, indices)):
            docs = []
            for rank, (d_idx, score) in enumerate(zip(retrieved_idx, dist)):
                doc_id = self.target_id_map.get(d_idx, f"unknown_{d_idx}")
                doc_text = self.target_texts[d_idx] if d_idx < len(self.target_texts) else ""
                docs.append({
                    "rank": rank + 1,
                    "doc_id": doc_id,
                    "text": doc_text,
                    "score": float(score),
                })
            results.append({
                "query": query,
                "retrieved_docs": docs,
            })

        return results

    def retrieve_as_doc_prompt(self, queries: list, top_k: int = None) -> list:
        """
        Retrieve và trả về dưới dạng doc prompts (giống format FiD input).

        Args:
            queries: List[str]
            top_k: int

        Returns:
            List[dict] với format FiD-compatible:
                - id: index
                - question: query gốc
                - ctxs: list {title, text, score}
        """
        if top_k is None:
            top_k = self.top_k

        raw_results = self.retrieve(queries, top_k)
        fid_inputs = []
        for idx, result in enumerate(raw_results):
            ctxs = []
            for doc in result["retrieved_docs"]:
                ctxs.append({
                    "title": doc["doc_id"],
                    "text": doc["text"],
                    "score": doc["score"],
                })
            fid_inputs.append({
                "id": idx,
                "question": result["query"],
                "ctxs": ctxs,
            })
        return fid_inputs


# ────────────────────────── CLI Demo ─────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Team 19 - Dense Retriever Demo")
    parser.add_argument("--model_name", type=str, default=DEFAULT_RETRIEVER_MODEL)
    parser.add_argument("--tokenizer_fallback", type=str, default=DEFAULT_TOKENIZER_FALLBACK)
    parser.add_argument("--target_file", type=str, default=DEFAULT_TARGET_FILE)
    parser.add_argument("--target_id_file", type=str, default=DEFAULT_TARGET_ID_FILE)
    parser.add_argument("--target_embed_cache", type=str, default=DEFAULT_TARGET_EMBED_CACHE)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    # ─── Demo queries (gọn hơn so với conala_nl.txt) ───
    demo_queries = [
        "sort a list of dictionaries by a value of the dictionary in python",
        "how to download a file from http url in python",
        "convert a string to datetime in python",
        "read a csv file into a pandas dataframe",
        "how to flatten a nested list in python",
    ]

    # Load retriever (chỉ 1 lần)
    retriever = DenseRetriever(
        model_name=args.model_name,
        tokenizer_fallback=args.tokenizer_fallback,
        target_file=args.target_file,
        target_id_file=args.target_id_file,
        target_embed_cache=args.target_embed_cache,
        top_k=args.top_k,
        num_layers=args.num_layers,
        use_cpu=args.cpu,
    )

    # Gọi retrieve nhiều lần
    print("\n" + "=" * 60)
    print("  📋 KẾT QUẢ RETRIEVE (Lần 1)")
    print("=" * 60)
    results = retriever.retrieve(demo_queries)
    _print_results(results)

    # Gọi lại lần 2 để demo "load 1 lần, chạy nhiều lần"
    print("\n" + "=" * 60)
    print("  📋 KẾT QUẢ RETRIEVE (Lần 2 - query khác)")
    print("=" * 60)
    extra_queries = [
        "remove duplicates from a list",
        "merge two dictionaries in python 3",
    ]
    results2 = retriever.retrieve(extra_queries, top_k=5)
    _print_results(results2)

    # Trả về dạng FiD-compatible
    print("\n" + "=" * 60)
    print("  📋 FID-COMPATIBLE DOC PROMPTS")
    print("=" * 60)
    fid_inputs = retriever.retrieve_as_doc_prompt(demo_queries[:2], top_k=3)
    print(json.dumps(fid_inputs, indent=2, ensure_ascii=False)[:2000])
    print("...")


def _print_results(results: list):
    """In kết quả retrieve đẹp."""
    for r in results:
        print(f"\n🔹 Query: {r['query']}")
        for doc in r["retrieved_docs"][:5]:  # chỉ hiển thị top 5
            print(f"   [{doc['rank']}] {doc['doc_id']} (score: {doc['score']:.4f})")
            print(f"       {doc['text'][:100]}...")
        print("-" * 50)


if __name__ == "__main__":
    main()
