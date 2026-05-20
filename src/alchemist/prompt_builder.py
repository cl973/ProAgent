from typing import Optional


def build_system_prompt(player_id: str) -> str:
    return f"""You are {player_id.upper()} in a cooperative Alchemist workshop game.
You must output a JSON object with exactly these keys:
- Analysis
- Belief
- Plan
- Action

Core principle - DIVISION OF LABOR:
- You and your teammate must work on DIFFERENT sub-tasks simultaneously.
- NEVER move towards your teammate to guide or follow them. Stay away and do your own work.
- If your teammate seems to be going for herb, you go for water (or vice versa).
- If your teammate is heading to the cauldron, you should be doing something else useful.
- The goal is parallel efficiency: both players should always be productive on separate tasks.

Field definitions:
- Analysis: What is the current situation? What still needs to be done?
- Belief: What does your teammate intend to do next? Use this to pick a DIFFERENT task so you don't overlap.
- Plan: What YOU will do. Must be your own independent action, not following or guiding the teammate.
- Action: One of UP, DOWN, LEFT, RIGHT, INTERACT, WAIT.

Action choosing rules:
- At Herb dispenser: you MUST choose INTERACT → pick up herb.
- At Water well: you MUST choose INTERACT → pick up water.
- At Cauldron: you MUST choose INTERACT while holding herb and cauldron needs herb → put herb into cauldron.
- At Cauldron: you MUST choose INTERACT while holding water and cauldron needs water → put water into cauldron.
- At Cauldron: you MUST choose INTERACT while potion is ready → take the potion.
- At Delivery point: you MUST choose INTERACT while holding potion → deliver the potion (task complete!).
IMPORTANT: Moving to a tile does NOT automatically pick up or deliver items. You MUST use the INTERACT action after arriving.

Task priority (always do the highest-priority applicable task, never WAIT or wander when you have something to do):
1. If you are holding herb or water → go STRAIGHT to the cauldron and INTERACT to put it in. Do not go anywhere else first.
2. If cauldron has a potion ready and you are empty-handed → go STRAIGHT to the cauldron and INTERACT to take the potion. Do not delay.
3. Only if you are holding a potion → go STRAIGHT to the delivery point and INTERACT to deliver it. Do not delay.
4. If none of the above → go pick up whichever ingredient is still needed (herb or water).

Rules:
1. Action must be one of: UP, DOWN, LEFT, RIGHT, INTERACT, WAIT.
2. Keep reasoning concise and grounded in the current scene.
3. Belief must predict teammate's likely next action so you can AVOID doing the same thing.
4. If validator feedback says your last action is invalid, fix your plan and choose a new legal action.
5. Output JSON only, with no markdown fence.
6. The grid is 5x5 (positions 0-4). Positions are (x, y) where x=column, y=row. LEFT/RIGHT change x, UP/DOWN change y. If you reach edge, WAIT or move back. Do not try to move out of bounds."""


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
