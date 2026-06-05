"""
Inspection script for the processed CUAD dataset.

Purpose:
- Load one example from the processed JSONL file.
- Verify that the processed dataset contains the fields needed for the RAG attribution pipeline.
- Confirm that gold_chunk_ids point to actual chunks in the chunks list.
- Print a preview of the gold-supporting chunk for manual inspection.

Input:
- data/processed/answerable_examples_with_gold_chunks.jsonl

Expected record fields:
- contract_id
- contract_title
- question_id
- clause_type
- question
- is_impossible
- answer_texts
- gold_spans
- gold_chunk_ids
- chunks

Current successful inspection:
- The first processed example is:
      Contract: LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT
      Clause type: Document Name
      Gold answer: "DISTRIBUTOR AGREEMENT"
      Gold chunk ID: 0_chunk_0
      Number of chunks: 13
- The gold chunk span is:
      0-5000
- The gold chunk preview contains:
      "DISTRIBUTOR AGREEMENT"

Interpretation:
- The processed dataset was built correctly.
- The chunking step preserved character offsets.
- The CUAD gold answer span was successfully mapped to a chunk ID.
- This processed JSONL file is ready to be used by the retrieval pipeline.

How this connects to later stages:
- retrieval.py will load these records.
- It will embed chunks and retrieve top-k chunks for each question.
- evaluation.py will check whether retrieved or attributed chunks match gold_chunk_ids.
"""

import json
from pathlib import Path


def main():
    path = Path("data/processed/answerable_examples_with_gold_chunks.jsonl")

    with path.open("r", encoding="utf-8") as f:
        first = json.loads(next(f))

    print("Keys:")
    print(first.keys())

    print("\nContract title:")
    print(first["contract_title"])

    print("\nClause type:")
    print(first["clause_type"])

    print("\nQuestion:")
    print(first["question"])

    print("\nGold chunk IDs:")
    print(first["gold_chunk_ids"])

    print("\nNumber of chunks:")
    print(len(first["chunks"]))

    print("\nFirst answer text:")
    print(first["answer_texts"][0])

    gold_chunk_id = first["gold_chunk_ids"][0]
    gold_chunk = [
        chunk for chunk in first["chunks"] if chunk["chunk_id"] == gold_chunk_id
    ][0]

    print("\nGold chunk span:")
    print(gold_chunk["start_char"], "-", gold_chunk["end_char"])

    print("\nGold chunk preview:")
    print(gold_chunk["text"][:1000])


if __name__ == "__main__":
    main()
