"""
Manual test script for CUAD chunking.

Purpose:
- Load answerable CUAD examples using dataset.py.
- Chunk a contract into overlapping character-based chunks.
- Select a gold CUAD answer span.
- Identify which chunk(s) overlap that gold span.
- Verify that the gold answer text can be recovered from the supporting chunk.

Why this test matters:
- Our project evaluates source attribution for RAG-based contract review.
- To evaluate attribution, we need ground-truth supporting chunks.
- CUAD gives ground-truth answer spans, but our RAG system works with chunks.
- This test confirms that we can map CUAD answer spans to chunk IDs.

Current confirmed cases:
1. Document Name:
   - Contract: LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT
   - Gold answer: "DISTRIBUTOR AGREEMENT"
   - Gold span: 44-65
   - Supporting chunk: 0_chunk_0
   - Result: recovered answer matched exactly.

2. Governing Law:
   - Contract: LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT
   - Gold answer: "This Agreement is to be construed according to the laws          of the State of Illinois."
   - Gold span: 52061-52151
   - Supporting chunk: 0_chunk_12
   - Result: recovered answer matched exactly.

Interpretation:
- The mapping from CUAD character-level annotations to our chunk-level representation works.
- This lets us later evaluate retrieval scores, LLM citations, leave-one-out attribution, and optional Shapley-style attribution against gold supporting chunks.
"""

from dataset import load_cuad
from chunking import chunk_text, find_supporting_chunks


def main():
    examples = load_cuad(
        "data/raw/data/CUADv1.json",
        only_answerable=True,
    )

    target_clause = "Governing Law"

    matching_examples = [
        ex for ex in examples if ex.clause_type == target_clause and len(ex.answers) > 0
    ]

    example = matching_examples[0]
    answer = example.answers[0]

    chunks = chunk_text(
        text=example.context,
        contract_id=example.contract_id,
        chunk_size=5000,
        overlap=800,
    )

    supporting_chunks = find_supporting_chunks(
        chunks=chunks,
        answer_start=answer.answer_start,
        answer_end=answer.answer_end,
    )

    print("Contract title:", example.contract_title)
    print("Clause type:", example.clause_type)
    print("Question:", example.question)

    print("\nGold answer:")
    print(answer.text)
    print("Gold start:", answer.answer_start)
    print("Gold end:", answer.answer_end)

    print("\nNumber of chunks:", len(chunks))
    print("Number of supporting chunks:", len(supporting_chunks))

    for chunk in supporting_chunks:
        print("\nSupporting chunk:")
        print("Chunk ID:", chunk.chunk_id)
        print("Chunk span:", chunk.start_char, "-", chunk.end_char)

        local_start = max(0, answer.answer_start - chunk.start_char)
        local_end = local_start + len(answer.text)

        print("\nRecovered answer inside chunk:")
        print(chunk.text[local_start:local_end])


if __name__ == "__main__":
    main()
