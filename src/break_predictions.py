"""Создать два среза для зондов: ответ словами и чужой ход решения."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from num2words import num2words
from openai import OpenAI
from tqdm import tqdm

from config import PRJ_ROOT, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, OLLAMA_BASE_URL
from evaluators import normalize_final_answer, split_model_answer


BROKEN_FRACTION = 0.5



def split_solution(answer: str) -> tuple[str, str]:
    """Вернуть ход решения и неизменённый финал, начиная с последнего ####."""
    _, answer = split_model_answer(answer)
    reasoning, marker, final = answer.rpartition("####")
    if not marker or not final.strip() or not reasoning.strip():
        raise ValueError("Нужны непустой ход решения и финальный ответ после ####.")
    return reasoning, marker + final


def reasoning_hash(reasoning: str) -> str:
    """Одинаковый ключ для текстов, различающихся только пробелами."""
    normalized = re.sub(r"\s+", " ", reasoning).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _cosine(left: list[float] | None, right: list[float] | None) -> float:
    """Вычисляем cosine similarity двух векторов"""
    if not left or not right or len(left) != len(right):
        return 0.0
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    return float(np.dot(left_array, right_array) / denominator) if denominator else 0.0


def number_to_words(answer: str) -> str:
    """Поменять только финальное число на английские слова, сохранив ####."""
    reasoning, final = split_solution(answer)
    number = normalize_final_answer(final)
    if number is None:
        raise ValueError(f"Не удалось извлечь финальное число: {final!r}")
    suffix = final[len("####"):]
    leading = suffix[:len(suffix) - len(suffix.lstrip())]
    trailing = suffix[len(suffix.rstrip()):]
    return reasoning + "####" + leading + num2words(str(number), lang="en") + trailing


def validate_embedding(vector) -> list[float]:
    array = np.asarray(vector, dtype=float)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all() or not np.linalg.norm(array):
        raise ValueError("Эмбеддинг должен быть непустым конечным ненулевым вектором.")
    return array.tolist()


def cache_embeddings(
    source: pd.DataFrame, source_path: Path, reasonings: dict[int, str],
) -> dict[str, list[float]]:
    """Дополнить selected_tasks.csv векторами; сохранять после каждого батча.

    Кеш действителен только для совпадающих модели и хеша хода решения.
    Ни задача, ни <think>, ни финальная строка не входят в эмбеддинг.
    """
    columns = ["embedding", "embedding_model", "embedding_text_hash"]
    for column in columns:
        if column not in source.columns:
            source[column] = ""

    texts = {reasoning_hash(text): text.strip() for text in reasonings.values()}
    embeddings = {}
    for index, text in reasonings.items():
        key = reasoning_hash(text)
        row = source.loc[index]
        if row["embedding_model"] == EMBEDDING_MODEL and row["embedding_text_hash"] == key:
            try:
                embeddings[key] = validate_embedding(json.loads(row["embedding"]))
            except (ValueError, TypeError):
                pass  # Некорректный кеш пересчитывается.

    missing = [key for key in texts if key not in embeddings]
    print(f"Эмбеддинги: в кеше {len(embeddings)}, нужно получить {len(missing)}")
    if missing:
        with OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=600.0) as client:
            with tqdm(total=len(missing), desc="Эмбеддинги", unit="текст") as progress:
                for offset in range(0, len(missing), EMBEDDING_BATCH_SIZE):
                    keys = missing[offset:offset + EMBEDDING_BATCH_SIZE]
                    response = client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=[texts[key] for key in keys],
                        encoding_format="float",
                    )
                    items = sorted(response.data, key=lambda item: item.index)
                    if [item.index for item in items] != list(range(len(keys))):
                        raise ValueError("Ollama вернула неверное число или индексы эмбеддингов.")
                    for key, item in zip(keys, items):
                        embeddings[key] = validate_embedding(item.embedding)
                    for index, text in reasonings.items():
                        key = reasoning_hash(text)
                        if key in embeddings:
                            source.loc[index, columns] = [
                                json.dumps(embeddings[key]), EMBEDDING_MODEL, key,
                            ]
                    temporary = source_path.with_name(source_path.name + ".tmp")
                    source.to_csv(temporary, index=False, encoding="utf-8-sig")
                    temporary.replace(source_path)
                    progress.update(len(keys))
    return embeddings


def replace_reasonings(
    objects: list[dict], count: int, source_path: Path,
) -> list[dict]:
    """Для каждого объекта взять наиболее похожее несовпадающее решение из CSV.

    Пропускается только текст текущего решения (с нормализацией пробелов).
    Остальные кандидаты разрешены, включая решения других заменяемых объектов.
    Одного донора можно использовать несколько раз.
    """
    targets = [split_solution(obj["model_answer"]) for obj in objects[:count]]
    target_keys = [reasoning_hash(reasoning) for reasoning, _ in targets]
    source = pd.read_csv(source_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if "ответ модели" not in source.columns:
        raise ValueError("В selected_tasks.csv нужна колонка 'ответ модели'.")
    reasonings = {}
    for index, row in source.iterrows():
        try:
            reasoning, _ = split_solution(row["ответ модели"])
        except ValueError:
            continue
        reasonings[index] = reasoning
    if len(reasonings) < len(source):
        print(f"Пропущено записей CSV без корректного разделения решения и ответа: {len(source) - len(reasonings)}")
    known = {reasoning_hash(text) for text in reasonings.values()}
    if not set(target_keys).issubset(known):
        raise ValueError("Некоторые заменяемые решения не найдены в selected_tasks.csv.")
    candidates = list(reasonings)
    # Для каждой строки исключаем только совпадения с её собственным решением.
    allowed = np.array([
        [key != reasoning_hash(reasonings[index]) for index in candidates]
        for key in target_keys
    ])
    if not allowed.any(axis=1).all():
        raise ValueError("Не для всех выбранных объектов найдено отличающееся решение.")
    embeddings = cache_embeddings(source, Path(source_path), reasonings)
    queries = np.array([embeddings[reasoning_hash(text)] for text, _ in targets], dtype=float)
    donors = np.array([embeddings[reasoning_hash(reasonings[index])] for index in candidates], dtype=float)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    donors /= np.linalg.norm(donors, axis=1, keepdims=True)
    similarity = queries @ donors.T
    similarity[~allowed] = -np.inf

    broken = [obj.copy() for obj in objects]
    for position, donor_position in enumerate(similarity.argmax(axis=1)):
        index = candidates[int(donor_position)]
        _, final = targets[position]
        broken[position]["model_answer"] = reasonings[index].rstrip() + "\n\n" + final
        print(
            f"{objects[position]['id']} <- запись CSV {index + 1}; "
            f"cosine={similarity[position, donor_position]:.4f}"
        )
    return broken


def break_predictions(
    slice_path: Path = PRJ_ROOT / "audit" / "slice.jsonl",
    source_path: Path = PRJ_ROOT / "data" / "labeling" / "selected_tasks.csv",
    output_dir: Path = PRJ_ROOT / "audit",
    fraction: float = BROKEN_FRACTION,
) -> tuple[Path, Path]:
    """Создать два полных среза, изменив первые floor(len(slice)*fraction) записей."""
    if not 0 < fraction <= 1:
        raise ValueError("Доля должна быть больше 0 и не больше 1.")
    objects = [json.loads(line) for line in Path(slice_path).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    count = int(len(objects) * fraction)
    if not count:
        raise ValueError("При выбранной доле N=0: нечего изменять.")
    ids = []
    for obj in objects:
        for field in ("id", "question", "model_answer"):
            if not isinstance(obj, dict) or not isinstance(obj.get(field), str) or not obj[field].strip():
                raise ValueError(f"Объекту нужно непустое строковое поле {field}.")
        ids.append(obj["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("ID среза должны быть уникальными.")

    first = [obj.copy() for obj in objects]
    for obj in first[:count]:
        obj["model_answer"] = number_to_words(obj["model_answer"])
    second = replace_reasonings(objects, count, Path(source_path))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (output_dir / "slice_broken_1.jsonl", output_dir / "slice_broken_2.jsonl")
    if Path(slice_path).resolve() in [path.resolve() for path in paths]:
        raise ValueError("Выходной файл не должен совпадать с исходным срезом.")
    for path, rows in zip(paths, (first, second)):
        with path.open("w", encoding="utf-8", newline="\n") as output:
            for obj in rows:
                output.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"{path}: всего {len(rows)}, изменено {count} ({count / len(rows):.1%})")
    return paths


if __name__ == "__main__":
    break_predictions()
