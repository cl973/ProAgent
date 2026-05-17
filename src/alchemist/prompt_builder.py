from typing import Optional


def build_system_prompt(player_id: str) -> str:
    return f"""You are {player_id.upper()} in a cooperative Alchemist workshop game.
You must output a JSON object with exactly these keys:
- Analysis
- Belief
- Plan
- Action

Rules:
1. Action must be one of: UP, DOWN, LEFT, RIGHT, INTERACT, WAIT.
2. Keep reasoning concise and grounded in the current scene.
3. Belief must predict teammate's likely next action.
4. If validator feedback says your last action is invalid, fix your plan and choose a new legal action.
5. Output JSON only, with no markdown fence.
6. The grid is 5x5 (positions 0-4). If you reach edge, WAIT or move back. Do not try to move out of bounds."""


def build_user_prompt(
    state_description: str,
    belief_memory_summary: str,
    belief_revision_hint: Optional[str] = None,
    validator_feedback: Optional[str] = None,
) -> str:
    chunks = [
        "Current scene description:",
        state_description,
        "",
        "Belief memory (recent records):",
        belief_memory_summary,
    ]
    if belief_revision_hint:
        chunks += ["", "Belief revision hint:", belief_revision_hint]
    if validator_feedback:
        chunks += ["", "Validator feedback:", validator_feedback]
    chunks += ["", "Now provide your next decision JSON."]
    return "\n".join(chunks)
