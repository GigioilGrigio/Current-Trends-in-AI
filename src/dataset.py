"""
Dataset loading utilities for the CUAD RAG attribution project.

Purpose:
- Load the raw CUADv1.json file.
- Convert the nested JSON structure into cleaner Python dataclasses.
- Make each contract-question pair easier to use in later pipeline stages.
- Preserve the full contract text, CUAD question, clause type, answer spans, and is_impossible flag.

CUAD JSON structure:
- The raw file is SQuAD-like:
      CUADv1.json
      ├── version
      └── data
          └── list of contracts
              ├── title
              └── paragraphs
                  └── paragraph
                      ├── context
                      └── qas

Important fields:
- contract["title"]:
      The contract/document title.
- paragraph["context"]:
      The full contract text.
- paragraph["qas"]:
      The list of legal clause questions for that contract.
- qa["id"]:
      Contains the contract identifier and clause type.
      Example:
      LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT__Governing Law
- qa["question"]:
      The legal review question used as the RAG query.
- qa["is_impossible"]:
      Whether the requested clause is absent from the contract.
- qa["answers"]:
      Expert-labeled answer spans when the clause exists.

Meaning of is_impossible:
- is_impossible = False:
      The clause exists in the contract.
      CUAD provides one or more gold answer spans.
      These examples are useful for evaluating source attribution.
- is_impossible = True:
      The clause is absent from the contract.
      There may be no answer span.
      These examples may be useful later for testing unsupported citations or hallucinations,
      but they are not used in the first attribution experiment.

Gold answer spans:
- Each answer contains:
      answer["text"]
      answer["answer_start"]
- The gold span is computed as:
      answer_start -> answer_start + len(answer_text)
- We validated that the first answerable example maps correctly back to the context:
      Gold answer: "DISTRIBUTOR AGREEMENT"
      Gold span: 44-65
      Recovered context slice matched exactly.

Current dataset statistics:
- Total QA examples: 20910
- Answerable QA examples: 6702

Current design decision:
- For the first version of the project, we mainly use:
      load_cuad(path, only_answerable=True)
- This keeps only examples where is_impossible is False.
- This is appropriate because source attribution requires a real supporting clause
  to compare retrieved/cited/attributed chunks against.

How this connects to the rest of the pipeline:
- dataset.py loads CUAD examples.
- chunking.py splits each contract context into chunks while preserving character offsets.
- CUAD gold answer spans are then mapped to chunk IDs.
- Those gold chunk IDs become ground truth for evaluating attribution methods such as:
      retrieval-score attribution,
      LLM-generated citations,
      leave-one-out attribution,
      optional sampled Shapley/surrogate attribution.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CuadAnswer:
    text: str
    answer_start: int

    @property
    def answer_end(self) -> int:
        return self.answer_start + len(self.text)


@dataclass
class CuadExample:
    contract_id: str
    contract_title: str
    context: str
    question_id: str
    question: str
    clause_type: str
    is_impossible: bool
    answers: list[CuadAnswer]


def extract_clause_type(question_id: str) -> str:
    """
    CUAD ids look like:
    WHITESMOKE_INC_...__Governing Law

    We extract the part after the final double underscore.
    """
    if "__" in question_id:
        return question_id.split("__")[-1]
    return question_id.split("_")[-1]


def load_cuad(
    path: str | Path,
    only_answerable: bool = False,
) -> list[CuadExample]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    examples: list[CuadExample] = []

    for contract_idx, contract in enumerate(raw["data"]):
        contract_title = contract["title"]

        for paragraph in contract["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                is_impossible = qa.get("is_impossible", False)

                if only_answerable and is_impossible:
                    continue

                answers = [
                    CuadAnswer(
                        text=answer["text"],
                        answer_start=answer["answer_start"],
                    )
                    for answer in qa.get("answers", [])
                ]

                example = CuadExample(
                    contract_id=str(contract_idx),
                    contract_title=contract_title,
                    context=context,
                    question_id=qa["id"],
                    question=qa["question"],
                    clause_type=extract_clause_type(qa["id"]),
                    is_impossible=is_impossible,
                    answers=answers,
                )

                examples.append(example)

    return examples


def main():
    path = "data/raw/data/CUADv1.json"

    all_examples = load_cuad(path, only_answerable=False)
    answerable_examples = load_cuad(path, only_answerable=True)

    print("Total QA examples:", len(all_examples))
    print("Answerable QA examples:", len(answerable_examples))

    first = answerable_examples[0]

    print("\nFirst answerable example")
    print("Contract ID:", first.contract_id)
    print("Contract title:", first.contract_title)
    print("Question ID:", first.question_id)
    print("Clause type:", first.clause_type)
    print("Question:", first.question)
    print("Is impossible:", first.is_impossible)
    print("Number of answers:", len(first.answers))

    if first.answers:
        answer = first.answers[0]

        print("\nGold answer text:")
        print(answer.text)

        print("\nGold span:")
        print("start:", answer.answer_start)
        print("end:", answer.answer_end)

        recovered = first.context[answer.answer_start : answer.answer_end]

        print("\nRecovered text from context:")
        print(recovered)

        print("\nExact match between answer and context slice:")
        print(answer.text == recovered)


if __name__ == "__main__":
    main()
