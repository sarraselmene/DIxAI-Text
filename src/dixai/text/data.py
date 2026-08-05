from datasets import load_dataset


def load_sst2(split="validation", limit=None):
    dataset = load_dataset(
        "glue",
        "sst2",
        split=split
    )

    texts = [
        x["sentence"]
        for x in dataset
    ]

    if limit:
        texts = texts[:limit]

    return texts