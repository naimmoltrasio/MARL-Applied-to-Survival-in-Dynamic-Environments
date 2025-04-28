import functools
import random
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv
import pygame


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
            other_x, other_y = self.agent2_x, self.agent2_y
        else:
            x, y = self.agent2_x, self.agent2_y
            other_x, other_y = self.agent1_x, self.agent1_y

        new_x, new_y = x, y

        if action == 0 and x > 0:
            new_x -= 1
        elif action == 1 and x < 6:
            new_x += 1
        elif action == 2 and y > 0:
            new_y -= 1
        elif action == 3 and y < 6:
            new_y += 1
        # acción 4 es PICK_UP → no mueve

        # Verifica si la nueva posición está ocupada por un objeto no recogido
        if (new_x, new_y) in [pos for i, pos in enumerate(self.objects) if i not in self.collected]:
            return  # movimiento bloqueado, se queda donde está

        # Verifica si nueva posición está ocupada por el otro agente
        if (new_x, new_y) == (other_x, other_y):
            return  # Bloqueado por el otro agente

        # Aplica el movimiento
        if agent == "agent_1":
            self.agent1_x, self.agent1_y = new_x, new_y
        else:
            self.agent2_x, self.agent2_y = new_x, new_y

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
        return agent1_pos, agent2_pos, *object_pos

    def _is_adjacent(self, ax, ay, ox, oy):
        return abs(ax - ox) + abs(ay - oy) == 1

    def render(self):
        cell_size = 80
        width, height = 7 * cell_size, 7 * cell_size

        if not hasattr(self, "screen"):
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Collaborative Pick Up")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))  # blanco

        # Dibujar celdas del grid
        for y in range(7):
            for x in range(7):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)  # borde gris

        # Dibujar objetos
        for i, (x, y) in enumerate(self.objects):
            if i not in self.collected:
                center = (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2)
                pygame.draw.circle(self.screen, (0, 0, 255), center, 20)  # objeto azul

        # Dibujar agentes
        agent1_pos = (self.agent1_x * cell_size + 20, self.agent1_y * cell_size + 20)
        agent2_pos = (self.agent2_x * cell_size + 20, self.agent2_y * cell_size + 20)

        pygame.draw.rect(self.screen, (255, 0, 0), (*agent1_pos, 40, 40))  # Agente 1 rojo
        pygame.draw.rect(self.screen, (0, 255, 0), (*agent2_pos, 40, 40))  # Agente 2 verde

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        """
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
        """


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # 2 agentes + 3 objetos codificados como celdas (0–48)
        return MultiDiscrete([49, 49, 49, 49, 49])

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)  # 0:left, 1:right, 2:down, 3:up, 4:pick_up
