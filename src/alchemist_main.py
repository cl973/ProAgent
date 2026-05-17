import datetime
import json
import os
import random
import sys
import time
from argparse import ArgumentParser
from distutils.util import strtobool
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from AlchemistEnv import AlchemistEnv
from alchemist.proagent_alchemist import ProAgentAlchemist


def boolean_argument(value):
    return bool(strtobool(value))


def _move_towards(pos, target):
    r, c = pos
    tr, tc = target
    if r < tr:
        return "DOWN"
    if r > tr:
        return "UP"
    if c < tc:
        return "RIGHT"
    if c > tc:
        return "LEFT"
    return "INTERACT"


def greedy_action(env: AlchemistEnv, player_id: str) -> str:
    pos = tuple(env.players_pos[player_id])
    holding = env.players_holding[player_id]
    cauldron = env.cauldron

    herb_pos = (0, 0)
    water_pos = (4, 1)
    cauldron_pos = (2, 2)
    delivery_pos = (4, 4)

    if holding == "potion":
        target = delivery_pos
    elif holding == "herb" or holding == "water":
        target = cauldron_pos
    elif cauldron["has_potion"]:
        target = cauldron_pos
    elif not cauldron["has_herb"]:
        target = herb_pos
    elif not cauldron["has_water"]:
        target = water_pos
    else:
        target = cauldron_pos

    action = _move_towards(pos, target)
    valid, _ = env.check_action_valid(player_id, action)
    return action if valid else "WAIT"


def random_action(env: AlchemistEnv, player_id: str) -> str:
    actions = ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "WAIT"]
    random.shuffle(actions)
    for act in actions:
        valid, _ = env.check_action_valid(player_id, act)
        if valid:
            return act
    return "WAIT"


def choose_action(algo, env, player_id, proagent, teammate_last_action, turn):
    if algo == "ProAgent":
        state_desc = env.get_state_description(player_id)
        decision = proagent.decide_action(
            env=env,
            state_description=state_desc,
            teammate_last_action=teammate_last_action,
            turn=turn,
        )
        return decision.action, {
            "cot": decision.cot.__dict__,
            "retries": decision.retries,
            "valid": decision.valid,
            "validator_reason": decision.validator_reason,
        }
    if algo == "Greedy":
        act = greedy_action(env, player_id)
        return act, {"cot": None, "retries": 0, "valid": True, "validator_reason": None}
    if algo == "Random":
        act = random_action(env, player_id)
        return act, {"cot": None, "retries": 0, "valid": True, "validator_reason": None}
    return "WAIT", {"cot": None, "retries": 0, "valid": True, "validator_reason": None}


def main(variant):
    random.seed(variant["seed"])
    np.random.seed(variant["seed"])

    p0_algo = variant["p0"]
    p1_algo = variant["p1"]
    horizon = variant["horizon"]
    episode = variant["episode"]

    p0_agent = None
    p1_agent = None
    if p0_algo == "ProAgent":
        p0_agent = ProAgentAlchemist(
            player_id="p1",
            model=variant["model"],
            api_base=variant["api_base"],
            key_file=variant["key_file"],
            memory_k=variant["memory_k"],
            max_reflection_retry=variant["max_retry"],
        )
    if p1_algo == "ProAgent":
        p1_agent = ProAgentAlchemist(
            player_id="p2",
            model=variant["model"],
            api_base=variant["api_base"],
            key_file=variant["key_file"],
            memory_k=variant["memory_k"],
            max_reflection_retry=variant["max_retry"],
        )

    raw_results = []
    trajectories = []
    start_time = time.time()

    for ep in range(episode):
        env = AlchemistEnv()
        env.reset()
        prev_actions = {"p1": None, "p2": None}
        step_logs = []

        for _ in range(horizon):
            turn = env.turn + 1
            a1, info1 = choose_action(
                p0_algo, env, "p1", p0_agent, teammate_last_action=prev_actions["p2"], turn=turn
            )
            a2, info2 = choose_action(
                p1_algo, env, "p2", p1_agent, teammate_last_action=prev_actions["p1"], turn=turn
            )

            state = env.step(a1, a2)
            step_logs.append(
                {
                    "turn": state["turn"],
                    "actions": {"p1": a1, "p2": a2},
                    "errors": env.last_errors,
                    "score": env.score,
                    "cauldron_status": state["cauldron_status"],
                    "p1_info": info1,
                    "p2_info": info2,
                }
            )
            prev_actions = {"p1": a1, "p2": a2}

        raw_results.append(env.score)
        trajectories.append({"episode": ep + 1, "steps": step_logs})

    result_dict = {
        "input": variant,
        "raw_results": raw_results,
        "mean_result": float(np.mean(raw_results)) if raw_results else 0.0,
        "trajectories": trajectories,
        "cost_seconds": round(time.time() - start_time, 3),
    }

    print(f"raw_results: {raw_results}")
    print(f"mean_result: {result_dict['mean_result']}")

    if variant["save"]:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        log_dir = variant["log_dir"] or f"experiments\\alchemist_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        json_file = (
            f"{log_dir}\\results_{episode}_{horizon}_{p0_algo}_{p1_algo}_{variant['model']}.json"
        )
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        print(f"saved: {json_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Alchemist experiment runner")
    parser.add_argument("--p0", type=str, default="ProAgent", choices=["ProAgent", "Greedy", "Random", "Stay"])
    parser.add_argument("--p1", type=str, default="Greedy", choices=["ProAgent", "Greedy", "Random", "Stay"])
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--model", type=str, default="llama-3.1-8b")
    parser.add_argument("--api_base", type=str, default="https://yunwu.ai/v1")
    parser.add_argument("--key_file", type=str, default="src\\openai_key.txt")
    parser.add_argument("--memory_k", type=int, default=5)
    parser.add_argument("--max_retry", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=boolean_argument, default=True)
    parser.add_argument("--log_dir", type=str, default=None)

    args = parser.parse_args()
    variant = vars(args)

    os.chdir(ROOT_DIR)
    main(variant)
