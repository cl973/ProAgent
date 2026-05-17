from dataclasses import dataclass
from typing import Optional


ACTION_SPACE = ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "WAIT"]


@dataclass
class CoTDict:
    Analysis: str
    Belief: str
    Plan: str
    Action: str


@dataclass
class AgentDecision:
    action: str
    cot: CoTDict
    retries: int
    valid: bool
    validator_reason: Optional[str] = None
