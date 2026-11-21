"""Подготовить slice.jsonl с размышлениями, решением модели и эталоном."""

import argparse
import csv
import json
from pathlib import Path


from config import PRJ_ROOT
from evaluators import split_model_answer


def make_slice_json(
    source_path: Path = PRJ_ROOT / "data" / "labeling" / "selected_tasks.csv",
    output_dir: Path = PRJ_ROOT / "audit",
    team_id: str = "K09",
) -> Path:
    """Взять первые 10 + 16 примеров каждой метки в исходном порядке.

    Первые 10 True и 10 False относятся к core, следующие 16 каждой
    метки — к second. ID содержит номер записи исходного CSV (без заголовка),
    поэтому остаётся прежним, если новые записи добавляются в конец CSV.
    model_thinking содержит содержимое <think>, model_answer — решение без
    этого блока, reference_answer — полный эталон из датасета.
    """
    quotas = {"core": 10, "second": 16}
    counts = {group: {True: 0, False: 0} for group in quotas}
    objects = []
    required = {"задача", "ответ модели", "результат exact_match", "эталонный ответ"}

    with Path(source_path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"В CSV отсутствуют колонки: {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=1):
            label_text = (row["результат exact_match"] or "").strip().lower()
            if label_text not in {"true", "false"}:
                raise ValueError(f"Запись {row_number}: некорректная метка {label_text!r}")
            label = label_text == "true"
            group = next(
                (group for group, quota in quotas.items() if counts[group][label] < quota),
                None,
            )
            if group is None:
                continue

            for column in ("задача", "ответ модели", "эталонный ответ"):
                if not row[column] or not row[column].strip():
                    raise ValueError(f"Запись {row_number}: пустое поле {column!r}")

            thinking, solution = split_model_answer(row["ответ модели"])
            if not solution:
                raise ValueError(f"Запись {row_number}: нет решения вне блока <think>.")
            obj = {
                "id": f"{team_id}-{row_number:03d}",
                "group": group,
                "question": row["задача"],
                "model_thinking": thinking,
                "model_answer": solution,
                "reference_answer": row["эталонный ответ"],
            }
            objects.append(obj)
            counts[group][label] += 1
            if all(
                counts[group][label] == quota
                for group, quota in quotas.items()
                for label in (True, False)
            ):
                break

    if any(
        counts[group][label] < quota
        for group, quota in quotas.items()
        for label in (True, False)
    ):
        raise ValueError(
            "Недостаточно примеров: нужны 26 True и 26 False. "
            f"Набрано: {counts}. Выходные файлы не изменены."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_path = output_dir / "slice.jsonl"
    with slice_path.open("w", encoding="utf-8", newline="\n") as output:
        for obj in objects:
            output.write(json.dumps(obj, ensure_ascii=False) + "\n")

    for group in quotas:
        print(f"{group}: True={counts[group][True]}, False={counts[group][False]}")
    print(f"Сохранено {len(objects)} примеров: {slice_path}")
    return slice_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=PRJ_ROOT / "data/labeling/selected_tasks.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=PRJ_ROOT / "audit")
    parser.add_argument("--team-id", default="K09")
    args = parser.parse_args()
    make_slice_json(args.source, args.output_dir, args.team_id)


if __name__ == "__main__":
    main()
