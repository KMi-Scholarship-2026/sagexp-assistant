# KMi Stress-Sense Retrieval Evaluation

An offline evaluation and benchmarking script for the **KMi Stress-Sense** assistant's retrieval pipeline. This script replicates the deployed TF-IDF retriever (`rag_pipeline.py`) and benchmarks its retrieval accuracy, latency, and storage footprint against two alternative lexical retrieval configurations.

## Overview

The evaluation measures how effectively different sparse retrieval methods surface relevant context chunks (top K = 4) across 24 test queries split into two categories:
* **Lexical (12 queries):** Direct questions sharing exact terminology with the source documents.
* **Paraphrase (12 queries):** Natural-language questions using synonyms or indirect phrasing to test vocabulary mismatch limitations.

### Evaluated Systems
1. **`tfidf` (Deployed Baseline):** Exact replication of `rag_pipeline.py`, computing term frequency normalized by document length (TF / L) multiplied by \(\log(N / (\text{df} + 1))\).
2. **`tfidf_stop`:** Identical to the deployed TF-IDF retriever, but strips common English stopwords from the query prior to scoring.
3. **`bm25`:** Standard Okapi BM25 scoring (k₁ = 1.5, b = 0.75) incorporating term frequency saturation and document length normalization.

---

## Prerequisites & Setup

### Directory Structure
Ensure your workspace is structured as follows before running the script:
```text
.
├── evaluate.py   # This evaluation script
└── documents/              # Directory containing knowledge base .txt files
    └── *.txt
```

### Dependencies
* Python 3.8+
* `langchain-text-splitters`

Install the required package via pip:
```bash
pip install langchain-text-splitters
```

---

## Usage

Run the script directly from the root directory containing the `documents/` folder:

```bash
python evaluate_retrieval.py
```

### Side Effects
* Generates a serialized index file named `simple_db.json` in the working directory to measure index build time and disk footprint.

---

## Evaluation Methodology & Outputs

1. **Chunking:** Reads all `.txt` files in `documents/` in sorted order and splits them using `RecursiveCharacterTextSplitter` (`chunk_size=500`, `chunk_overlap=50`).
2. **Ground Truth Matching:** Identifies "gold" chunks by checking for exact substring matches (whitespace-normalized) against a pre-specified answer phrase for each query.
3. **Metrics Reported:**
   * **Hit@1:** Proportion of queries where a gold chunk is ranked #1.
   * **Recall@4:** Proportion of queries where at least one gold chunk appears in the top $K=4$ results (alongside a random baseline expectation $K/N$).
   * **MRR (Mean Reciprocal Rank):** Average of $1 / \text{rank}$ of the first retrieved gold chunk across queries.
   * **Per-Question Breakdown:** Prints the exact rank of the first gold chunk for every query across all three systems.
   * **Rank Distribution:** Shows where gold chunks land within the top-4 window (`pos 1–4`) for the deployed TF-IDF system.
   * **System Footprint:** Reports index build/serialization time (ms), `simple_db.json` file size (KB), and query latency percentiles (median, p95, max over $200 \times 24 = 4,800$ runs).

