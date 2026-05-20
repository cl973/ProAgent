"""
AlchemistEnv - 炼金工坊协作环境
动作空间: ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "WAIT"]
状态JSON格式:
{
    "turn": int,
    "grid": 5x5 list, 元素为 "empty"/静态物体/"player1"/"player2",
    "players": {"p1": {"holding": "none"/"herb"/"water"/"potion"}, "p2": {...}},
    "cauldron_status": "empty" / "has_herb" / "has_water" / "has_both" / "has_potion"
}
"""

from typing import Dict, Tuple, List
import json

# 常量定义
class Tile:
    EMPTY = "empty"
    HERB_DISPENSER = "herb_dispenser"
    WATER_WELL = "water_well"
    CAULDRON = "cauldron"
    DELIVERY = "delivery"

class Holding:
    NONE = "none"
    HERB = "herb"
    WATER = "water"
    POTION = "potion"

ACTION_SPACE = ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "WAIT"]


class AlchemistEnv:
    def __init__(self):
        # 静态地图（不含玩家）
        self.static_grid = [
            [Tile.HERB_DISPENSER, Tile.EMPTY,      Tile.EMPTY,      Tile.EMPTY,      Tile.EMPTY],
            [Tile.EMPTY,          Tile.EMPTY,      Tile.EMPTY,      Tile.EMPTY,      Tile.EMPTY],
            [Tile.EMPTY,          Tile.EMPTY,      Tile.CAULDRON,   Tile.EMPTY,      Tile.EMPTY],
            [Tile.EMPTY,          Tile.EMPTY,      Tile.EMPTY,      Tile.EMPTY,      Tile.EMPTY],
            [Tile.EMPTY,          Tile.WATER_WELL, Tile.EMPTY,      Tile.EMPTY,      Tile.DELIVERY],
        ]
        self.grid_size = 5
        self.reset()

    def reset(self, p1_start: Tuple[int, int] = (1, 1), p2_start: Tuple[int, int] = (0, 1)) -> Dict:
        """重置环境，默认黄金剧本：P1在草药台旁，P2在水井旁"""
        self.players_pos = {"p1": list(p1_start), "p2": list(p2_start)}
        self.players_holding = {"p1": Holding.NONE, "p2": Holding.NONE}
        self.cauldron = {"has_herb": False, "has_water": False, "has_potion": False}
        self.score = 0
        self.turn = 0
        self.last_errors = {"p1": None, "p2": None}
        return self.get_state_dict()

    # ---------- 辅助 ----------
    def _get_tile(self, pos: Tuple[int, int]) -> str:
        r, c = pos
        if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
            return self.static_grid[r][c]
        return Tile.EMPTY

    # ---------- 动作合法性检查 ----------
    def check_action_valid(self, player_id: str, action: str) -> Tuple[bool, str]:
        if action not in ACTION_SPACE:
            return False, f"Invalid action '{action}'. Valid: {ACTION_SPACE}"

        pos = tuple(self.players_pos[player_id])
        holding = self.players_holding[player_id]
        tile = self._get_tile(pos)

        if action == "WAIT":
            return True, ""

        # 移动
        if action in ("UP", "DOWN", "LEFT", "RIGHT"):
            dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}[action]
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                return True, ""
            return False, f"Move out of bounds to ({nr},{nc})"

        # 交互
        if action == "INTERACT":
            # 草药台
            if tile == Tile.HERB_DISPENSER:
                if holding != Holding.NONE:
                    return False, f"Already holding {holding}, cannot pick herb"
                return True, ""
            # 水井
            if tile == Tile.WATER_WELL:
                if holding != Holding.NONE:
                    return False, f"Already holding {holding}, cannot pick water"
                return True, ""
            # 炼金锅
            if tile == Tile.CAULDRON:
                if self.cauldron["has_potion"]:
                    if holding != Holding.NONE:
                        return False, "Must be empty-handed to take potion"
                    return True, ""
                else:
                    if holding == Holding.HERB and not self.cauldron["has_herb"]:
                        return True, ""
                    if holding == Holding.WATER and not self.cauldron["has_water"]:
                        return True, ""
                    return False, f"Cannot put {holding} into cauldron (herb:{self.cauldron['has_herb']}, water:{self.cauldron['has_water']})"
            # 交付台
            if tile == Tile.DELIVERY:
                if holding == Holding.POTION:
                    return True, ""
                return False, f"Holding {holding}, need potion to deliver"
            return False, f"Nothing to interact at {tile}"

        return False, "Unreachable"

    # ---------- 执行单个动作 ----------
    def _execute_action(self, player_id: str, action: str):
        pos = tuple(self.players_pos[player_id])
        holding = self.players_holding[player_id]
        tile = self._get_tile(pos)

        if action == "WAIT":
            return
        if action in ("UP", "DOWN", "LEFT", "RIGHT"):
            dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}[action]
            self.players_pos[player_id][0] += dr
            self.players_pos[player_id][1] += dc
            return
        if action == "INTERACT":
            if tile == Tile.HERB_DISPENSER:
                self.players_holding[player_id] = Holding.HERB
            elif tile == Tile.WATER_WELL:
                self.players_holding[player_id] = Holding.WATER
            elif tile == Tile.CAULDRON:
                if self.cauldron["has_potion"]:
                    self.players_holding[player_id] = Holding.POTION
                    self.cauldron["has_potion"] = False
                else:
                    if holding == Holding.HERB and not self.cauldron["has_herb"]:
                        self.cauldron["has_herb"] = True
                        self.players_holding[player_id] = Holding.NONE
                    elif holding == Holding.WATER and not self.cauldron["has_water"]:
                        self.cauldron["has_water"] = True
                        self.players_holding[player_id] = Holding.NONE
                    # 自动合成药水
                    if self.cauldron["has_herb"] and self.cauldron["has_water"]:
                        self.cauldron["has_potion"] = True
                        self.cauldron["has_herb"] = False
                        self.cauldron["has_water"] = False
            elif tile == Tile.DELIVERY and holding == Holding.POTION:
                self.players_holding[player_id] = Holding.NONE
                self.score += 1

    # ---------- 主 step 函数 ----------
    def step(self, action1: str, action2: str) -> Dict:
        self.turn += 1
        self.last_errors = {"p1": None, "p2": None}

        valid1, err1 = self.check_action_valid("p1", action1)
        valid2, err2 = self.check_action_valid("p2", action2)
        if not valid1:
            self.last_errors["p1"] = err1
        if not valid2:
            self.last_errors["p2"] = err2

        if valid1:
            self._execute_action("p1", action1)
        if valid2:
            self._execute_action("p2", action2)

        return self.get_state_dict()

    # ---------- 获取简化 JSON 状态 ----------
    def get_state_dict(self) -> Dict:
        # 构建 grid：先复制静态网格，再覆盖玩家
        grid = [row[:] for row in self.static_grid]
        for pid, pos in self.players_pos.items():
            r, c = pos
            grid[r][c] = "player1" if pid == "p1" else "player2"

        # 合成 cauldron_status
        if self.cauldron["has_potion"]:
            cs = "has_potion"
        elif self.cauldron["has_herb"] and self.cauldron["has_water"]:
            cs = "has_both"
        elif self.cauldron["has_herb"]:
            cs = "has_herb"
        elif self.cauldron["has_water"]:
            cs = "has_water"
        else:
            cs = "empty"

        return {
            "turn": self.turn,
            "grid": grid,
            "players": {
                "p1": {"holding": self.players_holding["p1"]},
                "p2": {"holding": self.players_holding["p2"]}
            },
            "cauldron_status": cs
        }

    # ---------- 自然语言描述（给 LLM）----------
    def get_state_description(self, player_id: str) -> str:
        state = self.get_state_dict()
        players = state["players"]
        cs = state["cauldron_status"]

        p1_pos = tuple(self.players_pos["p1"])
        p2_pos = tuple(self.players_pos["p2"])

        herb_pos = (0, 0)
        water_pos = (4, 1)
        cauldron_pos = (2, 2)
        delivery_pos = (4, 4)

        if player_id == "p1":
            my_pos = p1_pos
            my_holding = players["p1"]["holding"]
            teammate_pos = p2_pos
            teammate_holding = players["p2"]["holding"]
        else:
            my_pos = p2_pos
            my_holding = players["p2"]["holding"]
            teammate_pos = p1_pos
            teammate_holding = players["p1"]["holding"]

        status_text = {
            "empty": "empty, needs herb and water",
            "has_herb": "has herb, needs water",
            "has_water": "has water, needs herb",
            "has_both": "has both herb and water (ready to brew)",
            "has_potion": "has a potion ready to take"
        }.get(cs, "unknown")

        desc = f"""Turn {state['turn']}. Grid size 5x5.

Coordinate system: positions are (x, y). x is column (0=LEFT, 4=RIGHT). y is row (0=TOP, 4=BOTTOM).
Action effects on (x, y):
- LEFT: x decreases by 1 (move toward left)
- RIGHT: x increases by 1 (move toward right)
- UP: y decreases by 1 (move toward top)
- DOWN: y increases by 1 (move toward bottom)
Example: to go from (2, 1) to (1, 4), you need x-1 and y+3, so go LEFT once then DOWN three times.

Static objects:
- Herb dispenser at ({herb_pos[1]}, {herb_pos[0]})
- Water well at ({water_pos[1]}, {water_pos[0]})
- Cauldron at ({cauldron_pos[1]}, {cauldron_pos[0]})
- Delivery point at ({delivery_pos[1]}, {delivery_pos[0]})

You are {player_id.upper()} at ({my_pos[1]}, {my_pos[0]}), holding {my_holding}.
Teammate is at ({teammate_pos[1]}, {teammate_pos[0]}), holding {teammate_holding}.

Cauldron status: {status_text}.

Goal: brew a potion by putting herb and water into the cauldron, wait for the potion to be ready, then take the potion and deliver it to ({delivery_pos[1]}, {delivery_pos[0]}).
IMPORTANT: Work in parallel with your teammate. Do NOT follow or guide your teammate. If your teammate is handling one ingredient, you handle the other. If your teammate is at the cauldron, do something else productive. Both players should always be working on DIFFERENT sub-tasks.

Valid actions: {ACTION_SPACE}.
Last step error: {self.last_errors.get(player_id, 'none')}

Please respond with a JSON containing Analysis, Belief, Plan, and Action."""
        return desc

    # ---------- 保存/加载 JSON ----------
    def save_state_to_json(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.get_state_dict(), f, indent=2)

    def load_state_from_json(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.turn = data["turn"]
        grid = data["grid"]
        self.players_pos = {}
        self.players_holding = {"p1": data["players"]["p1"]["holding"], "p2": data["players"]["p2"]["holding"]}
        for r in range(5):
            for c in range(5):
                cell = grid[r][c]
                if cell == "player1":
                    self.players_pos["p1"] = [r, c]
                elif cell == "player2":
                    self.players_pos["p2"] = [r, c]
        cs = data["cauldron_status"]
        self.cauldron = {"has_herb": False, "has_water": False, "has_potion": False}
        if cs == "has_herb":
            self.cauldron["has_herb"] = True
        elif cs == "has_water":
            self.cauldron["has_water"] = True
        elif cs == "has_both":
            self.cauldron["has_herb"] = True
            self.cauldron["has_water"] = True
        elif cs == "has_potion":
            self.cauldron["has_potion"] = True
        self.score = 0
        self.last_errors = {"p1": None, "p2": None}


# 简单测试
#if __name__ == "__main__":
#    env = AlchemistEnv()
#    env.reset()
#    print("=== Initial JSON ===")
#    print(json.dumps(env.get_state_dict(), indent=2))
#    print("\n=== Description for P1 ===")
#    print(env.get_state_description("p1"))