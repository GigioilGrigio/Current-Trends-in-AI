"""
Chunking utilities for the CUAD RAG attribution project.

Purpose:
- Split each full contract text into overlapping chunks.
- Preserve character offsets for each chunk: start_char and end_char.
- Use those offsets to map CUAD gold answer spans to the chunk(s) that contain them.
- These gold-supporting chunks will later be used as ground truth for evaluating source-attribution methods.

Why character-based chunking for now:
- CUAD annotations are character-based: each answer has answer_start and answer text.
- Therefore, the gold span can be computed as:
      answer_start -> answer_start + len(answer_text)
- Character-based chunks make it easy and exact to check whether a chunk overlaps a gold CUAD span.
- This is simpler than token-based chunking for the first implementation.
- Legal text often requires larger chunks because clauses can be long and context-dependent, so we currently use large overlapping chunks.

Current default:
- chunk_size = 5000 characters
- overlap = 800 characters

Confirmed test cases:
1. Document Name example:
   - Gold answer: "DISTRIBUTOR AGREEMENT"
   - Gold span: 44-65
   - Found in chunk: 0_chunk_0
   - Recovered text matched the gold answer exactly.

2. Governing Law example:
   - Gold answer: "This Agreement is to be construed according to the laws          of the State of Illinois."
   - Gold span: 52061-52151
   - Found in chunk: 0_chunk_12
   - Recovered text matched the gold answer exactly.

These tests confirm that our chunk span logic correctly links CUAD gold answers to supporting chunks.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    contract_id: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    contract_id: str,
    chunk_size: int = 5000,
    overlap: int = 800,
) -> list[Chunk]:
    """
    Simple character-based chunking.

    We use characters instead of tokens for now because CUAD gold annotations
    are also character-based via answer_start.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk = Chunk(
            contract_id=contract_id,
            chunk_id=f"{contract_id}_chunk_{chunk_index}",
            text=text[start:end],
            start_char=start,
            end_char=end,
        )
        chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap
        chunk_index += 1

    return chunks


def span_overlap(
    span_start: int,
    span_end: int,
    chunk_start: int,
    chunk_end: int,
) -> int:
    """
    Returns number of overlapping characters between a gold answer span
    and a chunk span.
    """
    overlap_start = max(span_start, chunk_start)
    overlap_end = min(span_end, chunk_end)

    return max(0, overlap_end - overlap_start)


def find_supporting_chunks(
    chunks: list[Chunk],
    answer_start: int,
    answer_end: int,
    min_overlap_chars: int = 1,
) -> list[Chunk]:
    """
    Returns chunks that overlap with a gold CUAD answer span.
    """
    supporting_chunks = []

    for chunk in chunks:
        overlap = span_overlap(
            span_start=answer_start,
            span_end=answer_end,
            chunk_start=chunk.start_char,
            chunk_end=chunk.end_char,
        )

        if overlap >= min_overlap_chars:
            supporting_chunks.append(chunk)

    return supporting_chunks


from dataset import load_cuad
from chunking import chunk_text, find_supporting_chunks


def main():
    examples = load_cuad(
        "data/raw/data/CUADv1.json",
        only_answerable=True,
    )

    example = examples[0]
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
    print("Question:", example.question)
    print("Clause type:", example.clause_type)

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
