import functools
import random
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv


class CollaborativePickUpEnv(ParallelEnv):
    metadata = {
        "name": "collaborative_pickup_v0",
        "render_modes": ["human"],
        "render_fps": 4
    }

    def __init__(self, render_mode="human"):
        self.render_mode = render_mode
        self.agent1_x = None
        self.agent1_y = None
        self.agent2_x = None
        self.agent2_y = None
        self.objects = []  # Lista de objetos [(x, y), ...]
        self.collected = set()  # Índices de objetos ya levantados
        self.timestep = 0
        self.possible_agents = ["agent_1", "agent_2"]

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.collected = set()

        self.agent1_x, self.agent1_y = 0, 0
        self.agent2_x, self.agent2_y = 6, 6

        # Coloca 3 objetos aleatorios en el grid (sin repetir)
        self.objects = []
        while len(self.objects) < 3:
            pos = (random.randint(1, 5), random.randint(1, 5))
            if pos not in self.objects:
                self.objects.append(pos)

        obs = self._get_observation()
        observations = {
            "agent_1": {"observation": obs, "action_mask": self._action_mask(self.agent1_x, self.agent1_y)},
            "agent_2": {"observation": obs, "action_mask": self._action_mask(self.agent2_x, self.agent2_y)},
        }

        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        a1_action = actions["agent_1"]
        a2_action = actions["agent_2"]

        # Ejecutar movimiento de ambos agentes
        self._move_agent("agent_1", a1_action)
        self._move_agent("agent_2", a2_action)

        # Recompensas
        rewards = {"agent_1": 0, "agent_2": 0}

        # Verificar acción PICK_UP (acción 4)
        if a1_action == 4 and a2_action == 4:
            for i, (ox, oy) in enumerate(self.objects):
                if i in self.collected:
                    continue
                if (
                    self._is_adjacent(self.agent1_x, self.agent1_y, ox, oy)
                    and self._is_adjacent(self.agent2_x, self.agent2_y, ox, oy)
                ):
                    self.collected.add(i)
                    rewards = {"agent_1": 1, "agent_2": 1}
                    break

        # Termina si levantaron todos los objetos
        done = len(self.collected) == len(self.objects)
        terminations = {a: done for a in self.agents}
        truncations = {a: self.timestep > 100 for a in self.agents}

        if done or self.timestep > 100:
            self.agents = []

        obs = self._get_observation()
        observations = {
            "agent_1": {"observation": obs, "action_mask": self._action_mask(self.agent1_x, self.agent1_y)},
            "agent_2": {"observation": obs, "action_mask": self._action_mask(self.agent2_x, self.agent2_y)},
        }

        infos = {a: {} for a in self.possible_agents}
        self.timestep += 1

        return observations, rewards, terminations, truncations, infos

    def _move_agent(self, agent, action):
        if agent == "agent_1":
            x, y = self.agent1_x, self.agent1_y
        else:
            x, y = self.agent2_x, self.agent2_y

        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < 6:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < 6:
            y += 1
        # acción 4 es PICK_UP → no mueve

        if agent == "agent_1":
            self.agent1_x, self.agent1_y = x, y
        else:
            self.agent2_x, self.agent2_y = x, y

    def _action_mask(self, x, y):
        mask = np.ones(5, dtype=np.int8)  # 5 acciones: 4 mov + 1 pick_up
        if x == 0: mask[0] = 0
        if x == 6: mask[1] = 0
        if y == 0: mask[2] = 0
        if y == 6: mask[3] = 0
        return mask

    def _get_observation(self):
        # Codificamos todas las posiciones en valores únicos
        agent1_pos = self.agent1_x + 7 * self.agent1_y
        agent2_pos = self.agent2_x + 7 * self.agent2_y
        object_pos = [
            (x + 7 * y) if i not in self.collected else 48  # marcamos recogido como 48 (fuera de rango)
            for i, (x, y) in enumerate(self.objects)
        ]
        return (agent1_pos, agent2_pos, *object_pos)

    def _is_adjacent(self, ax, ay, ox, oy):
        return abs(ax - ox) + abs(ay - oy) == 1

    def render(self):
        grid = np.full((7, 7), ".", dtype=str)
        for i, (x, y) in enumerate(self.objects):
            if i not in self.collected:
                grid[y, x] = "O"
        a1 = self.agent1_y, self.agent1_x
        a2 = self.agent2_y, self.agent2_x

        if grid[a1] == ".":
            grid[a1] = "A"
        else:
            grid[a1] = "X"

        if grid[a2] == ".":
            grid[a2] = "a"
        else:
            grid[a2] = "X"

        print("\n".join([" ".join(row) for row in grid]))
        print()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # 2 agentes + 3 objetos codificados como celdas (0–48)
        return MultiDiscrete([49, 49, 49, 49, 49])

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)  # 0:left, 1:right, 2:down, 3:up, 4:pick_up
