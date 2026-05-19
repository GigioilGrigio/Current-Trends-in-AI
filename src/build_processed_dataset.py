"""
Build a small processed CUAD dataset for the RAG attribution pipeline.

Purpose:
- Load answerable CUAD examples from the raw CUADv1.json file.
- Chunk each contract using the character-based chunking utilities in chunking.py.
- Map each CUAD gold answer span to the chunk(s) that overlap it.
- Save the result as a JSONL file that can be used by later retrieval and attribution experiments.

Input:
- data/raw/data/CUADv1.json

Output:
- data/processed/answerable_examples_with_gold_chunks.jsonl

Why we build this processed file:
- The raw CUAD file is deeply nested and inconvenient to use directly.
- The RAG pipeline works with chunks, not full contracts.
- CUAD gold labels are span-based, so we need to convert gold answer spans into gold chunk IDs.
- Once this file exists, later steps can load clean records without repeating the raw JSON parsing and chunk mapping logic.

Current processing choices:
- We use only answerable examples:
      load_cuad(path, only_answerable=True)
- This means we keep examples where:
      is_impossible == False
- These examples contain gold answer spans, which are required for evaluating whether attribution methods identify the true supporting chunks.

Chunking setup:
- Character-based chunking is used for now.
- Current defaults:
      chunk_size = 5000 characters
      overlap = 800 characters
- This was chosen because CUAD answer spans are also character-based, making gold-span-to-chunk mapping simple and exact.

Gold chunk construction:
- For each answer:
      gold_start = answer.answer_start
      gold_end = answer.answer_start + len(answer.text)
- A chunk is considered a gold-supporting chunk if it overlaps the gold answer span.
- The resulting gold chunk IDs are stored in:
      gold_chunk_ids

Output record structure:
Each JSONL line corresponds to one answerable contract-question example and contains:
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

Important validation:
- We first tested the span-to-chunk mapping manually on two cases:
  1. Document Name:
     - Gold answer: "DISTRIBUTOR AGREEMENT"
     - Gold span: 44-65
     - Supporting chunk: 0_chunk_0
     - Recovered text matched exactly.
  2. Governing Law:
     - Gold answer: "This Agreement is to be construed according to the laws          of the State of Illinois."
     - Gold span: 52061-52151
     - Supporting chunk: 0_chunk_12
     - Recovered text matched exactly.

Current successful run:
- Command:
      python src/build_processed_dataset.py
- Output:
      Processed dataset written to: data/processed/answerable_examples_with_gold_chunks.jsonl
      Examples written: 100
      Examples without gold chunk: 0

Interpretation of the successful run:
- 100 answerable CUAD examples were processed.
- Every processed example had at least one gold-supporting chunk.
- This confirms that the current chunking and gold-span mapping are working on the initial sample.

How this connects to later pipeline stages:
- retrieval.py will embed the chunks and retrieve top-k chunks for each question.
- attribution.py will compare retrieval scores, LLM citations, leave-one-out, and optional Shapley-style scores.
- evaluation.py will compare predicted support chunks against gold_chunk_ids.
"""

import json
from pathlib import Path

from dataset import load_cuad
from chunking import chunk_text, find_supporting_chunks


def build_processed_dataset(
    input_path: str = "data/raw/data/CUADv1.json",
    output_path: str = "data/processed/answerable_examples_with_gold_chunks.jsonl",
    chunk_size: int = 5000,
    overlap: int = 800,
    max_examples: int | None = None,
):
    examples = load_cuad(input_path, only_answerable=True)

    if max_examples is not None:
        examples = examples[:max_examples]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_written = 0
    num_without_gold_chunk = 0

    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            chunks = chunk_text(
                text=example.context,
                contract_id=example.contract_id,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            gold_spans = []
            gold_chunk_ids = set()

            for answer in example.answers:
                gold_start = answer.answer_start
                gold_end = answer.answer_end
                gold_spans.append([gold_start, gold_end])

                supporting_chunks = find_supporting_chunks(
                    chunks=chunks,
                    answer_start=gold_start,
                    answer_end=gold_end,
                    min_overlap_chars=1,
                )

                for chunk in supporting_chunks:
                    gold_chunk_ids.add(chunk.chunk_id)

            if not gold_chunk_ids:
                num_without_gold_chunk += 1

            record = {
                "contract_id": example.contract_id,
                "contract_title": example.contract_title,
                "question_id": example.question_id,
                "clause_type": example.clause_type,
                "question": example.question,
                "is_impossible": example.is_impossible,
                "answer_texts": [answer.text for answer in example.answers],
                "gold_spans": gold_spans,
                "gold_chunk_ids": sorted(gold_chunk_ids),
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "text": chunk.text,
                    }
                    for chunk in chunks
                ],
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

    print("Processed dataset written to:", output_path)
    print("Examples written:", num_written)
    print("Examples without gold chunk:", num_without_gold_chunk)


if __name__ == "__main__":
    build_processed_dataset(max_examples=100)
