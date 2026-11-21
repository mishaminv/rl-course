"""Download the main configuration of GSM8K as train/test CSV files."""

import csv
from datasets import load_dataset

from config import PRJ_ROOT



def download_dataset():
    """Save only GSM8K main, preserving the original order and full answers."""
    dataset = load_dataset("openai/gsm8k", "main")
    output_dir = PRJ_ROOT / "data" / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        path = output_dir / f"{split}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["question", "answer"])
            writer.writeheader()
            for row in dataset[split]:
                writer.writerow({"question": row["question"], "answer": row["answer"]})
        print(f"{split}: {len(dataset[split])} примеров → {path}")



if __name__ == "__main__":
    download_dataset()
    print("done")

