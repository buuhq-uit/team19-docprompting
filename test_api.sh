#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Team 19 - Test API Endpoints với curl
# ═══════════════════════════════════════════════════════════════
#
#  Chạy sau khi server đã start:
#    bash test_api.sh
#    bash test_api.sh http://localhost:9000   # port khác
#
# ═══════════════════════════════════════════════════════════════

BASE_URL="${1:-http://localhost:8000}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Team 19 - Test DocPrompting API                        ║"
echo "║  Server: $BASE_URL"
echo "╚══════════════════════════════════════════════════════════╝"

# ─── 0. Health Check ─────────────────────────────────────────
echo ""
echo "━━━ 0. Health Check ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""

# ─── 1. Retrieve Queries ─────────────────────────────────────
echo "━━━ 1. POST /retrieve-queries ━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s -X POST "$BASE_URL/retrieve-queries" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "sort a list of dictionaries by a value of the dictionary in python",
      "how to download a file from http url in python"
    ],
    "top_k": 5
  }' | python3 -m json.tool
echo ""

# ─── 2. Generate Codes (with doc prompts) ────────────────────
echo "━━━ 2. POST /generate-codes ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s -X POST "$BASE_URL/generate-codes" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_prompts": [
      {
        "question": "sort a list of dictionaries by a value of the dictionary",
        "ctxs": [
          {
            "title": "sorted",
            "text": "sorted(iterable, *, key=None, reverse=False) Return a new sorted list from the items in iterable.",
            "score": 0.85
          },
          {
            "title": "list.sort",
            "text": "list.sort(*, key=None, reverse=False) This method sorts the list in place, using only < comparisons between items.",
            "score": 0.80
          }
        ]
      }
    ],
    "n_context": 2
  }' | python3 -m json.tool
echo ""

# ─── 3. Full Pipeline ────────────────────────────────────────
echo "━━━ 3. POST /generate-pipeline-codes ━━━━━━━━━━━━━━━━━━━━"
curl -s -X POST "$BASE_URL/generate-pipeline-codes" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "sort a list of dictionaries by a value of the dictionary in python",
      "how to download a file from http url in python",
      "convert a string to datetime in python"
    ],
    "top_k": 10,
    "n_context": 10
  }' | python3 -m json.tool
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Test hoàn tất!"
echo "═══════════════════════════════════════════════════════════"
