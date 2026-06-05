import json
from pathlib import Path
from pprint import pprint


def load_json(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    path = Path("data/raw/data/CUADv1.json")
    cuad = load_json(path)

    print("Top-level type:", type(cuad))
    print("Top-level keys:", list(cuad.keys()))

    contracts = cuad["data"]
    print("Number of contracts:", len(contracts))

    first = contracts[0]
    print("\nFirst contract keys:")
    print(list(first.keys()))

    print("\nFirst contract title:")
    print(first["title"])

    paragraphs = first["paragraphs"]
    print("\nNumber of paragraphs in first contract:", len(paragraphs))

    first_paragraph = paragraphs[0]
    print("\nFirst paragraph keys:")
    print(list(first_paragraph.keys()))

    context = first_paragraph["context"]
    qas = first_paragraph["qas"]

    print("\nContract text length, characters:", len(context))
    print("Number of questions for first contract:", len(qas))

    print("\nFirst 500 characters of contract:")
    print(context[:500])

    print("\nFirst question:")
    pprint(qas[0])

    print("\nFirst 5 question names:")
    for qa in qas[:5]:
        print("-", qa["question"])


if __name__ == "__main__":
    main()
