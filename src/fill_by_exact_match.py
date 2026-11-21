"""Заполнить audit/baseline_predictions.csv сохранёнными метками Exact Match."""

import argparse
from pathlib import Path

import pandas as pd

from config import PRJ_ROOT


def fill_by_exact_match(
    source_path: Path = PRJ_ROOT / "data" / "labeling" / "selected_tasks.csv",
    baseline_path: Path = PRJ_ROOT / "audit" / "baseline_predictions.csv",
    team_id: str = "K09",
) -> Path:
    """Перенести метки по ID, сохранив порядок и столбцы шаблона.

    Как в make_slice_json.py, K09-001 соответствует первой записи исходного
    CSV без заголовка. Используются уже рассчитанные результаты exact_match.
    """
    source = pd.read_csv(source_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    baseline = pd.read_csv(
        baseline_path, encoding="utf-8-sig", dtype=str, keep_default_na=False
    )
    if "результат exact_match" not in source.columns:
        raise ValueError("В исходном CSV отсутствует колонка 'результат exact_match'.")
    if not {"id", "verdict"}.issubset(baseline.columns):
        raise ValueError("В baseline_predictions.csv нужны колонки id и verdict.")
    if baseline["id"].eq("").any() or baseline["id"].duplicated().any():
        raise ValueError("ID в baseline_predictions.csv должны быть непустыми и уникальными.")

    labels = source["результат exact_match"].str.strip().str.lower()
    labels.index = [f"{team_id}-{number:03d}" for number in range(1, len(source) + 1)]
    verdicts = baseline["id"].map(labels)
    if verdicts.isna().any():
        missing = baseline.loc[verdicts.isna(), "id"].tolist()
        raise ValueError(f"В исходном CSV не найдены ID: {missing}")
    if not verdicts.isin(["true", "false"]).all():
        raise ValueError("Результаты exact_match должны быть true или false.")

    baseline["verdict"] = verdicts
    baseline_path = Path(baseline_path)
    baseline.to_csv(baseline_path, index=False, encoding="utf-8-sig")
    print(f"Заполнено {len(baseline)} строк: {baseline_path}")
    print(baseline["verdict"].value_counts().to_string())
    return baseline_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=PRJ_ROOT / "data/labeling/selected_tasks.csv"
    )
    parser.add_argument("--baseline", type=Path, default=PRJ_ROOT / "audit/baseline_predictions.csv")
    parser.add_argument("--team-id", default="K09")
    args = parser.parse_args()
    fill_by_exact_match(args.source, args.baseline, args.team_id)


if __name__ == "__main__":
    main()
