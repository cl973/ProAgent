import json
import re
from typing import Dict

from .types import ACTION_SPACE, CoTDict


def _extract_json_block(text: str) -> Dict:
    candidates = []
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    candidates.extend(fenced)
    inline = re.findall(r"(\{.*\})", text, flags=re.S)
    candidates.extend(inline)

    for block in candidates:
        try:
            data = json.loads(block)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


def _extract_action_fallback(text: str) -> str:
    for act in ACTION_SPACE:
        if re.search(rf"\b{act}\b", text, flags=re.I):
            return act
    m = re.search(r"Action\s*[:=]\s*([A-Za-z_]+)", text, flags=re.I)
    if m:
        return m.group(1).upper().strip()
    return "WAIT"


def normalize_action(action: str) -> str:
    a = (action or "").strip().upper()
    if a in ACTION_SPACE:
        return a
    return "WAIT"


def parse_cot_response(text: str) -> CoTDict:
    payload = _extract_json_block(text)

    analysis = str(payload.get("Analysis", "")).strip()
    belief = str(payload.get("Belief", "")).strip()
    plan = str(payload.get("Plan", "")).strip()
    action = payload.get("Action", "")

    if not action:
        action = _extract_action_fallback(text)

    return CoTDict(
        Analysis=analysis or "N/A",
        Belief=belief or "N/A",
        Plan=plan or "N/A",
        Action=normalize_action(str(action)),
    )
