from typing import Optional

from .llm_client import OpenAICompatibleClient
from .memory import BeliefMemory, infer_action_from_belief
from .parser import normalize_action, parse_cot_response
from .prompt_builder import build_system_prompt, build_user_prompt
from .types import ACTION_SPACE, AgentDecision, CoTDict


class ProAgentAlchemist:
    def __init__(
        self,
        player_id: str,
        model: str = "llama-3.1-8b",
        api_base: str = "https://yunwu.ai/v1",
        key_file: str = "src/openai_key.txt",
        memory_k: int = 5,
        max_reflection_retry: int = 3,
        max_tokens: int = 16384,
    ):
        self.player_id = player_id
        self.system_prompt = build_system_prompt(player_id)
        self.memory = BeliefMemory(capacity=memory_k)
        self.max_reflection_retry = max_reflection_retry
        self.client = OpenAICompatibleClient(model=model, api_base=api_base, key_file=key_file, max_tokens=max_tokens)

        self.last_predicted_teammate_action = "WAIT"

    def _build_messages(
        self,
        state_description: str,
        teammate_last_action: Optional[str],
        validator_feedback: Optional[str] = None,
        turn: int = 0,
    ):
        revision_hint = None
        if teammate_last_action is not None:
            revision_hint = self.memory.compare_and_record(
                turn=turn,
                predicted_action=self.last_predicted_teammate_action,
                actual_action=teammate_last_action,
            )

        user_prompt = build_user_prompt(
            state_description=state_description,
            belief_memory_summary=self.memory.summarize(),
            belief_revision_hint=revision_hint,
            validator_feedback=validator_feedback,
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def decide_action(
        self,
        env,
        state_description: str,
        teammate_last_action: Optional[str],
        turn: int,
    ) -> AgentDecision:
        feedback = None
        retries = 0
        parsed: CoTDict = CoTDict("N/A", "N/A", "N/A", "WAIT")
        last_reason = None

        while retries <= self.max_reflection_retry:
            messages = self._build_messages(
                state_description=state_description,
                teammate_last_action=teammate_last_action if retries == 0 else None,
                validator_feedback=feedback,
                turn=turn,
            )
            raw_text = self.client.chat(messages)
            print(f"[{self.player_id}] LLM raw response: {raw_text}")
            if not raw_text.strip():
                print(f"[{self.player_id}] Empty LLM response, falling back to WAIT.")
                parsed = CoTDict("API timeout", "unknown", "wait", "WAIT")
            else:
                parsed = parse_cot_response(raw_text)
            parsed.Action = normalize_action(parsed.Action)

            valid, reason = env.check_action_valid(self.player_id, parsed.Action)
            if valid:
                self.last_predicted_teammate_action = infer_action_from_belief(parsed.Belief)
                return AgentDecision(
                    action=parsed.Action,
                    cot=parsed,
                    retries=retries,
                    valid=True,
                )

            last_reason = reason
            retries += 1
            feedback = (
                f"Your last Action='{parsed.Action}' is invalid because: {reason}. "
                f"Please rethink and output a new legal action in JSON."
            )

        fallback = "WAIT"
        valid, reason = env.check_action_valid(self.player_id, fallback)
        return AgentDecision(
            action=fallback if valid else ACTION_SPACE[0],
            cot=parsed,
            retries=retries,
            valid=valid,
            validator_reason=last_reason or reason,
        )
