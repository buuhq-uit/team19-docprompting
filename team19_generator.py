"""
Team 19 - FiD Code Generator Module
=====================================
Load mô hình FiD (Fusion-in-Decoder) Code Generator một lần,
sau đó có thể gọi generate(doc_prompts) nhiều lần để sinh code.

Sử dụng:
    python team19_generator.py                          # chạy demo mặc định
    python team19_generator.py --per_gpu_batch_size 1   # batch size nhỏ hơn

Hoặc import như module:
    from team19_generator import FiDCodeGenerator
    generator = FiDCodeGenerator()
    codes = generator.generate(fid_inputs)
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional

import torch
import transformers
from tqdm import tqdm

# ─────────────────────────── Cấu hình mặc định ────────────────────────────
DEFAULT_MODEL_PATH = "models/generator/conala.fid.codet5.top10/checkpoint/best_dev/best_dev"
DEFAULT_TOKENIZER_NAME = "models/generator/codet5-base"
DEFAULT_N_CONTEXT = 10
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_LENGTH = 150
DEFAULT_NUM_BEAMS = 10
DEFAULT_LENPEN = 1.0


class FiDCodeGenerator:
    """
    FiD Code Generator dựa trên CodeT5.

    Workflow:
      1. Load FiDT5 model + tokenizer (chỉ 1 lần)
      2. Với mỗi list doc_prompts (format FiD) → tokenize → generate → decode
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
        n_context: int = DEFAULT_N_CONTEXT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        num_beams: int = DEFAULT_NUM_BEAMS,
        lenpen: float = DEFAULT_LENPEN,
        use_cpu: bool = False,
    ):
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.n_context = n_context
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_beams = num_beams
        self.lenpen = lenpen
        self.use_cpu = use_cpu

        self.model = None
        self.tokenizer = None
        self.device = None

        print("=" * 60)
        print("  Team 19 - FiD Code Generator (CodeT5)")
        print("=" * 60)
        self._load_model()

    # ─────────────────────── Load Model ─────────────────────────
    def _load_model(self):
        """Load FiDT5 model và tokenizer."""
        print(f"\n📦 Đang load model: {self.model_path}")

        # Thêm path để import FiD modules
        fid_dir = os.path.join(os.path.dirname(__file__), "generator", "fid")
        if fid_dir not in sys.path:
            sys.path.insert(0, fid_dir)

        import src.model as fid_model

        # Device
        self.device = torch.device('cpu') if self.use_cpu else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )

        # Load tokenizer
        print(f"   📝 Tokenizer: {self.tokenizer_name}")
        if 'codet5' in self.tokenizer_name:
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained(self.tokenizer_name)
        else:
            self.tokenizer = transformers.T5Tokenizer.from_pretrained(self.tokenizer_name)

        # Load FiD model
        self.model = fid_model.FiDT5.from_pretrained(self.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded trên device: {self.device}")

    # ──────────── Encode Passages (FiD format) ──────────────────
    def _encode_passages(self, batch_text_passages: list, text_maxlength: int = 200):
        """Encode passages cho FiD input."""
        passage_ids, passage_masks = [], []
        for text_passages in batch_text_passages:
            p = self.tokenizer(
                text_passages,
                max_length=text_maxlength,
                padding='max_length',
                return_tensors='pt',
                truncation=True,
            )
            passage_ids.append(p['input_ids'][None])
            passage_masks.append(p['attention_mask'][None])

        passage_ids = torch.cat(passage_ids, dim=0)
        passage_masks = torch.cat(passage_masks, dim=0)
        return passage_ids, passage_masks.bool()

    # ──────────────────── Generate Code ─────────────────────────
    def generate(self, fid_inputs: list, n_context: int = None) -> list:
        """
        Sinh code từ danh sách FiD-format inputs.

        Args:
            fid_inputs: List[dict] - mỗi dict có:
                - question: str
                - ctxs: list of {title: str, text: str, score: float}
            n_context: int - số context passages sử dụng

        Returns:
            List[dict] - mỗi dict chứa:
                - question: câu hỏi gốc
                - generated_code: code sinh ra
        """
        if n_context is None:
            n_context = self.n_context

        print(f"\n⚙️  Đang sinh code cho {len(fid_inputs)} câu hỏi (n_context={n_context}, beams={self.num_beams})...")

        results = []
        with torch.no_grad():
            for i in tqdm(range(0, len(fid_inputs), self.batch_size), desc="Generating"):
                batch = fid_inputs[i: i + self.batch_size]

                # Chuẩn bị input: question + passages
                batch_text_passages = []
                for item in batch:
                    question = "question: " + item["question"]
                    ctxs = item.get("ctxs", [])[:n_context]
                    passages = []
                    for ctx in ctxs:
                        title = ctx.get("title", "")
                        text = ctx.get("text", "")
                        passages.append(f"{question} title: {title} context: {text}")

                    # Nếu không đủ context, pad bằng question
                    while len(passages) < n_context:
                        passages.append(question)

                    batch_text_passages.append(passages)

                # Encode
                context_ids, context_mask = self._encode_passages(batch_text_passages)
                context_ids = context_ids.to(self.device)
                context_mask = context_mask.to(self.device)

                # Generate
                outputs = self.model.generate(
                    input_ids=context_ids,
                    attention_mask=context_mask,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    lenpen=self.lenpen,
                )

                # Decode
                for k, output in enumerate(outputs):
                    ans = self.tokenizer.decode(output, skip_special_tokens=False)
                    # Clean up giống test_reader_simple.py
                    ans = (ans.replace("{{", " {{")
                              .replace("\n", " ")
                              .replace("\r", "")
                              .replace("<pad>", "")
                              .replace("<s>", "")
                              .replace("</s>", "")
                              .strip())
                    ans = " ".join(ans.split())

                    results.append({
                        "question": batch[k]["question"],
                        "generated_code": ans,
                    })

        return results

    def generate_from_json_file(self, json_path: str, n_context: int = None) -> list:
        """
        Sinh code từ file JSON (format giống fid.cmd_test.codet5.t10.json).

        Args:
            json_path: đường dẫn tới file JSON

        Returns:
            List[dict] - kết quả generate
        """
        print(f"\n📂 Đang load dữ liệu từ: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"   Số mẫu: {len(data)}")
        return self.generate(data, n_context)


# ────────────────────────── CLI Demo ─────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Team 19 - FiD Code Generator Demo")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer_name", type=str, default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--n_context", type=int, default=DEFAULT_N_CONTEXT)
    parser.add_argument("--per_gpu_batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--num_beams", type=int, default=DEFAULT_NUM_BEAMS)
    parser.add_argument("--eval_data", type=str, default=None,
                        help="Path tới file JSON chứa FiD input (VD: data/conala/fid.cmd_test.codet5.t10.json)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    # Load generator (chỉ 1 lần)
    generator = FiDCodeGenerator(
        model_path=args.model_path,
        tokenizer_name=args.tokenizer_name,
        n_context=args.n_context,
        batch_size=args.per_gpu_batch_size,
        max_length=args.max_length,
        num_beams=args.num_beams,
        use_cpu=args.cpu,
    )

    if args.eval_data:
        # Chạy với file JSON đầu vào
        results = generator.generate_from_json_file(args.eval_data)
    else:
        # Demo với dữ liệu tự tạo (giả lập doc prompts)
        print("\n📝 Chạy demo với dữ liệu giả lập (không có file --eval_data)")
        demo_inputs = [
            {
                "question": "sort a list of dictionaries by a value of the dictionary",
                "ctxs": [
                    {
                        "title": "sorted",
                        "text": "sorted(iterable, *, key=None, reverse=False) Return a new sorted list from the items in iterable.",
                        "score": 0.85,
                    },
                    {
                        "title": "list.sort",
                        "text": "list.sort(*, key=None, reverse=False) This method sorts the list in place, using only < comparisons between items.",
                        "score": 0.80,
                    },
                ],
            },
            {
                "question": "download a file from http url and save to disk",
                "ctxs": [
                    {
                        "title": "urllib.request.urlretrieve",
                        "text": "urllib.request.urlretrieve(url, filename=None, reporthook=None, data=None) Copy a network object denoted by a URL to a local file.",
                        "score": 0.90,
                    },
                    {
                        "title": "urllib.request.urlopen",
                        "text": "urllib.request.urlopen(url, data=None, timeout, *, cafile=None, capath=None, cadefault=False, context=None) Open the URL url, which can be either a string or a Request object.",
                        "score": 0.75,
                    },
                ],
            },
        ]
        results = generator.generate(demo_inputs, n_context=2)

    # In kết quả
    print("\n" + "=" * 60)
    print("  📋 KẾT QUẢ SINH CODE")
    print("=" * 60)
    for r in results[:10]:  # Chỉ hiển thị 10 kết quả đầu
        print(f"\n🔹 Question: {r['question']}")
        print(f"   💻 Code:  {r['generated_code']}")
        print("-" * 50)

    # Gọi lại lần 2 để demo "load 1 lần, chạy nhiều lần"
    if not args.eval_data:
        print("\n" + "=" * 60)
        print("  📋 KẾT QUẢ SINH CODE (Lần 2 - model đã load sẵn)")
        print("=" * 60)
        extra_inputs = [
            {
                "question": "remove duplicates from list while preserving order",
                "ctxs": [
                    {
                        "title": "dict.fromkeys",
                        "text": "classmethod fromkeys(iterable[, value]) Create a new dictionary with keys from iterable and values set to value.",
                        "score": 0.70,
                    },
                ],
            },
        ]
        results2 = generator.generate(extra_inputs, n_context=1)
        for r in results2:
            print(f"\n🔹 Question: {r['question']}")
            print(f"   💻 Code:  {r['generated_code']}")


if __name__ == "__main__":
    main()
