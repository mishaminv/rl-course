"""Проверить два зонда через Exact Match и LLM, дописать probe_results.csv."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import PRJ_ROOT
from evaluators import LLM_as_evaluator, exact_match


RESULT_COLUMNS = ["probe", "n", "metric", "value", "interpretation"]
PROBES = [
    ("answer_in_words", "slice_broken_1.jsonl", "scorer_broken_1.jsonl"),
    ("swapped_reasoning", "slice_broken_2.jsonl", "scorer_broken_2.jsonl"),
]


def read_objects(path: Path) -> dict[str, dict]:
    """Прочитать JSONL, проверив уникальность непустых строковых ID."""
    objects = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict) or not isinstance(obj.get("id"), str) or not obj["id"].strip():
            raise ValueError(f"В {path} нужна JSON-запись с непустым строковым id.")
        if obj["id"] in objects:
            raise ValueError(f"В {path} повторяется ID {obj['id']}.")
        objects[obj["id"]] = obj
    return objects


def evaluate_modified(objects: list[dict], cache_path: Path) -> list[bool]:
    """Оценить изменённые решения, сохраняя LLM-вердикты после каждого ответа.

    input_hash связывает кеш с конкретными задачей, решением и эталоном.
    После изменения самого оценщика кеш нужно удалить вручную.
    """
    cache = read_objects(cache_path) if cache_path.exists() else {}
    verdicts = []
    for obj in tqdm(objects, desc=cache_path.stem, unit="пример", dynamic_ncols=True):
        payload = {field: obj[field] for field in ("question", "model_answer", "reference_answer")}
        fingerprint = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        saved = cache.get(obj["id"], {})
        if saved.get("input_hash") == fingerprint and type(saved.get("verdict")) is bool:
            verdict = saved["verdict"]
        else:
            verdict = LLM_as_evaluator(
                math_task=obj["question"], solution=obj["model_answer"],
                reference_solution=obj["reference_answer"],
            )
            if type(verdict) is not bool:
                raise ValueError(f"LLM вернула не bool для {obj['id']}.")
            cache[obj["id"]] = {
                "id": obj["id"], "verdict": verdict, "input_hash": fingerprint,
            }
            temporary = cache_path.with_name(cache_path.name + ".tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as output:
                for result in cache.values():
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
            temporary.replace(cache_path)
        verdicts.append(verdict)
    return verdicts


def calculate_metrics(probe: str, before: list[bool], after: list[bool]) -> list[dict]:
    """Дельта доли True и доля True -> False среди исходно принятых."""
    if not before or len(before) != len(after):
        raise ValueError("Нужны непустые списки одинаковой длины до и после зонда.")
    rows = [{
        "probe": probe, "n": len(before), "metric": "acceptance_rate_delta",
        "value": (sum(after) - sum(before)) / len(before), "interpretation": "",
    }]
    positives = sum(before)
    if positives:
        rejected = sum(old and not new for old, new in zip(before, after))
        rows.append({
            "probe": probe, "n": positives, "metric": "true_to_false_rate",
            "value": rejected / positives, "interpretation": "",
        })
    else:
        print(f"{probe}: исходных True нет, true_to_false_rate не определена и пропущена.")
    return rows


def check_exact_match_robustness(audit_dir: Path = PRJ_ROOT / "audit") -> Path:
    """Проверить оба оценщика на двух изменённых срезах и ДОБАВИТЬ строки CSV.

    Исходные оценки читаются из baseline_predictions.csv и scorer.jsonl.
    В каждом зонде учитываются только ID с изменённым model_answer.
    Повторный запуск добавляет новые строки, даже если предыдущие уже есть.
    """
    audit_dir = Path(audit_dir)
    results_path = audit_dir / "probe_results.csv"
    has_results = results_path.exists() and results_path.stat().st_size > 0
    if has_results:
        columns = pd.read_csv(results_path, encoding="utf-8-sig", nrows=0).columns.tolist()
        if columns != RESULT_COLUMNS:
            raise ValueError(f"В probe_results.csv ожидаются столбцы {RESULT_COLUMNS}.")

    original = read_objects(audit_dir / "slice.jsonl")
    baseline = pd.read_csv(
        audit_dir / "baseline_predictions.csv", encoding="utf-8-sig", dtype=str,
        keep_default_na=False,
    )
    if not {"id", "verdict"}.issubset(baseline.columns) or baseline["id"].duplicated().any():
        raise ValueError("В baseline_predictions.csv нужны уникальные id и столбец verdict.")
    labels = baseline["verdict"].str.strip().str.lower()
    if not labels.isin(["true", "false"]).all():
        raise ValueError("В baseline_predictions.csv verdict должен быть true или false.")
    exact_before = dict(zip(baseline["id"], labels.eq("true")))
    llm_original = read_objects(audit_dir / "scorer.jsonl")
    if any(type(obj.get("verdict")) is not bool for obj in llm_original.values()):
        raise ValueError("В scorer.jsonl verdict должен быть логическим значением.")
    llm_before = {key: obj["verdict"] for key, obj in llm_original.items()}

    # Проверяем оба среза до первого запроса к модели.
    prepared = []
    for name, filename, cache_name in PROBES:
        broken = read_objects(audit_dir / filename)
        if broken.keys() != original.keys():
            raise ValueError(f"Набор ID в {filename} отличается от исходного среза.")
        changed = []
        for key, obj in broken.items():
            for field in ("question", "model_answer", "reference_answer"):
                if not isinstance(obj.get(field), str) or not obj[field].strip():
                    raise ValueError(f"{filename}, {key}: нужно непустое поле {field}.")
            if {k: v for k, v in obj.items() if k != "model_answer"} != {
                k: v for k, v in original[key].items() if k != "model_answer"
            }:
                raise ValueError(f"{filename}, {key}: изменены поля помимо model_answer.")
            if obj["model_answer"] != original[key]["model_answer"]:
                if key not in exact_before or key not in llm_before:
                    raise ValueError(f"Для {key} не хватает исходных вердиктов.")
                changed.append(obj)
        if not changed:
            raise ValueError(f"В {filename} нет изменённых решений.")
        prepared.append((name, changed, audit_dir / cache_name))

    rows = []
    for name, changed, cache_path in prepared:
        print(f"{name}: проверяется {len(changed)} изменённых примеров.")
        exact_after = [exact_match(obj["model_answer"], obj["reference_answer"]) for obj in changed]
        llm_after = evaluate_modified(changed, cache_path)
        for evaluator, before, after in (
            ("exact_match", exact_before, exact_after),
            ("LLM_as_evaluator", llm_before, llm_after),
        ):
            rows.extend(calculate_metrics(
                f"{name}/{evaluator}", [before[obj["id"]] for obj in changed], after,
            ))

    # Записываем итоги только после успешной проверки всех зондов.
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    needs_newline = False
    if has_results:
        with results_path.open("rb") as existing:
            existing.seek(-1, 2)
            needs_newline = existing.read(1) != b"\n"
    with results_path.open(
        "a", encoding="utf-8" if has_results else "utf-8-sig", newline=""
    ) as output:
        if needs_newline:
            output.write("\n")
        results.to_csv(output, index=False, header=not has_results)
    print(results.to_string(index=False))
    print(f"Добавлено {len(results)} строк: {results_path}")
    return results_path


if __name__ == "__main__":
    check_exact_match_robustness()
