"""Exact Match for the final numeric answer in GSM8K format."""

import re
from decimal import Decimal, InvalidOperation


def normalize_final_answer(answer: str) -> Decimal | None:
    """Read only the last #### answer; never search the solution for numbers.

    Commas denote thousands separators (GSM8K uses English notation).
    The #### marker is required for both model and reference answers.
    Missing or malformed final answers return None.
    """
    _, marker, final = answer.rpartition("####")
    if not marker:
        return None
    final = final.strip()
    final = re.sub(r"\s+", "", final).replace("−", "-")
    pattern = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    if not re.fullmatch(pattern, final):
        return None
    try:
        return Decimal(final.replace(",", ""))
    except InvalidOperation:
        return None


def exact_match(model_answer: str, reference_answer: str) -> bool:
    """Compare final numbers exactly, without checking reasoning or rounding."""
    predicted = normalize_final_answer(model_answer)
    reference = normalize_final_answer(reference_answer)
    return (predicted is not None) and (reference is not None) and (predicted == reference)

