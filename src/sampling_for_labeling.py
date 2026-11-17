"""Collect GSM8K answers via Ollama's OpenAI-compatible API."""

import csv
from pathlib import Path
from uuid import uuid4
from openai import OpenAI

from config import PRJ_ROOT
from evaluators import exact_match


def get_model_answer(client: OpenAI, question: str, model: str) -> tuple[str, str]:
    """Return final output for evaluation and the full answer for labeling."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Solve the math problem step by step. Enclose every intermediate "
                    "calculation in << >> as in GSM8K, for example: "
                    "2 + 3 = <<2+3=5>>5. End your response with "
                    "a separate line in the exact format: #### <number>. "
                    "Write only the numeric answer after ####, without units."
                ),
            },
            {"role": "user", "content": question},
        ],
        stream=False,
    )
    message = response.choices[0].message
    content = message.content or ""
    if not content.strip():
        raise ValueError("Ollama вернула пустой ответ модели.")
    thinking = (
        getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None)
        or ""
    )
    full_answer = f"<think>\n{thinking}\n</think>\n\n{content}" if thinking else content
    return content, full_answer


def sample_for_labeling(
    dataset_path: Path = PRJ_ROOT / "data" / "dataset" / "train.csv",
    output_dir: Path = PRJ_ROOT / "data" / "labeling",
    model: str = "gpt-oss:20b",
    host: str = "http://localhost:11434",
    per_label: int = 75,
) -> Path:
    """Walk the CSV in order and save examples only while their quota is open.

    Each question is sent independently, without the reference answer.
    Accepted rows are flushed immediately, including before an interruption.
    API failures propagate rather than being treated as incorrect solutions.
    """
    if per_label <= 0:
        raise ValueError("per_label должен быть положительным числом.")
    base_url = host.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    client = OpenAI(base_url=base_url, api_key="ollama", timeout=600.0)
    counts = {True: 0, False: 0}
    output_dir = Path(output_dir)
    with Path(dataset_path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"question", "answer"}.issubset(reader.fieldnames or []):
            raise ValueError("В датасете должны быть колонки question и answer.")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{uuid4().hex}.csv"
        with output_path.open("x", encoding="utf-8-sig", newline="") as output:
            writer = csv.writer(output)
            writer.writerow([
                "задача", "ответ модели", "результат exact_match", "эталонный ответ",
                "ручная проверка"
            ])
            output.flush()
            try:
                for index, row in enumerate(reader, start=1):
                    content, full_answer = get_model_answer(client, row["question"], model)
                    label = exact_match(content, row["answer"])
                    if counts[label] < per_label:
                        writer.writerow([row["question"], full_answer, label, row["answer"], "null"])
                        output.flush()
                        counts[label] += 1
                    print(
                        f"Обработано: {index}; True: {counts[True]}/{per_label}; "
                        f"False: {counts[False]}/{per_label}",
                        flush=True,
                    )
                    if all(count == per_label for count in counts.values()):
                        break
            finally:
                print(f"Собрано: True={counts[True]}, False={counts[False]}")
                print(f"Сохранено {sum(counts.values())} примеров: {output_path}")
    return output_path



if __name__ == "__main__":
    sample_for_labeling()
