from collections import deque
from dataclasses import dataclass
from typing import Deque

from .types import ACTION_SPACE


@dataclass
class BeliefRecord:
    turn: int
    predicted_action: str
    actual_action: str
    corrected: bool


def infer_action_from_belief(belief_text: str) -> str:
    text = (belief_text or "").upper()
    for act in ACTION_SPACE:
        if act in text:
            return act
    return "WAIT"


class BeliefMemory:
    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self.records: Deque[BeliefRecord] = deque(maxlen=capacity)

    def compare_and_record(self, turn: int, predicted_action: str, actual_action: str) -> str:
        pa = (predicted_action or "WAIT").upper()
        aa = (actual_action or "WAIT").upper()
        corrected = pa != aa
        self.records.append(BeliefRecord(turn=turn, predicted_action=pa, actual_action=aa, corrected=corrected))
        if corrected:
            return (
                f"Belief revision: last turn your belief predicted teammate action '{pa}', "
                f"but actual action was '{aa}'. Update your teammate model."
            )
        return f"Belief check: last turn prediction '{pa}' matched actual teammate action '{aa}'."

    def summarize(self) -> str:
        if not self.records:
            return "No prior belief records."
        lines = []
        for rec in list(self.records):
            flag = "corrected" if rec.corrected else "matched"
            lines.append(
                f"- Turn {rec.turn}: predicted={rec.predicted_action}, actual={rec.actual_action}, {flag}"
            )
        return "\n".join(lines)
