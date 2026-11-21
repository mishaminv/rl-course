"""Оценить audit/slice.jsonl через LLM и записать audit/scorer.jsonl."""

import json
from pathlib import Path

from tqdm import tqdm

from config import PRJ_ROOT, OLLAMA_BASE_URL
from evaluators import LLM_as_evaluator


def run_llm_evaluator(
    slice_path: Path = PRJ_ROOT / "audit" / "slice.jsonl",
    scorer_path: Path = PRJ_ROOT / "audit" / "scorer.jsonl",
) -> Path:
    """Сохранить по одному булевому verdict на ID, продолжая прерванный запуск.

    В оценщик передаются только задача, решение и эталон, без model_thinking.
    Уже записанные оценки не пересчитываются. При изменении содержимого среза
    прежний scorer.jsonl следует убрать перед повторным запуском.
    """
    slice_path, scorer_path = Path(slice_path), Path(scorer_path)
    if slice_path.resolve() == scorer_path.resolve():
        raise ValueError("Пути среза и результатов должны различаться.")

    objects = [
        json.loads(line)
        for line in slice_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not objects:
        raise ValueError("Срез пуст.")
    ids = []
    for obj in objects:
        for field in ("id", "question", "model_answer", "reference_answer"):
            if not isinstance(obj, dict) or not isinstance(obj.get(field), str) or not obj[field].strip():
                raise ValueError(f"Каждому объекту среза нужно непустое строковое поле {field}.")
        ids.append(obj["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("В срезе повторяются ID.")

    completed = []
    if scorer_path.exists():
        completed = [
            json.loads(line)
            for line in scorer_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for result in completed:
            if not isinstance(result, dict) or type(result.get("verdict")) is not bool:
                raise ValueError("В scorer.jsonl ожидаются объекты с булевым verdict.")
        if [result.get("id") for result in completed] != ids[:len(completed)]:
            raise ValueError("ID или порядок в scorer.jsonl не совпадают с началом среза.")

    scorer_path.parent.mkdir(parents=True, exist_ok=True)
    # Добавляем перевод строки, если существующий файл завершён без него.
    needs_newline = scorer_path.exists() and bool(scorer_path.stat().st_size)
    if needs_newline:
        with scorer_path.open("rb") as existing:
            existing.seek(-1, 2)
            needs_newline = existing.read(1) != b"\n"

    with scorer_path.open("a", encoding="utf-8", newline="\n") as output, tqdm(
        total=len(objects), initial=len(completed), desc="LLM-оценка",
        unit="пример", dynamic_ncols=True,
    ) as progress:
        if needs_newline:
            output.write("\n")
        for obj in objects[len(completed):]:
            progress.set_postfix_str(obj["id"])
            verdict = LLM_as_evaluator(
                math_task=obj["question"],
                solution=obj["model_answer"],
                reference_solution=obj["reference_answer"],
            )
            if type(verdict) is not bool:
                raise ValueError(f"Оценщик вернул не bool для {obj['id']}.")
            output.write(json.dumps({"id": obj["id"], "verdict": verdict}, ensure_ascii=False) + "\n")
            output.flush()
            progress.update(1)

    print(f"Сохранено {len(objects)} оценок: {scorer_path}")
    return scorer_path


def main() -> None:
    from openai import APIConnectionError, APITimeoutError

    try:
        run_llm_evaluator()
    except APITimeoutError:
        raise SystemExit(
            "Ollama не ответила за отведённое время. Проверьте её состояние и повторите запуск. "
            "Уже полученные оценки сохранены в scorer.jsonl."
        ) from None
    except APIConnectionError:
        raise SystemExit(
            f"Не удалось подключиться к Ollama: {OLLAMA_BASE_URL}. "
            "Запустите `ollama serve` в отдельном терминале, затем повторите запуск скрипта. "
            "Уже полученные оценки сохранены в scorer.jsonl."
        ) from None


if __name__ == "__main__":
    main()
