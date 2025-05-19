import functools
import random
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box
from pettingzoo import ParallelEnv
import pygame


class CollaborativePickUpEnv(ParallelEnv):
    metadata = {
        "name": "collaborative_pickup_v0",
        "render_modes": ["human"],
        "render_fps": 4,
        "is_parallelizable": True
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

        def _are_positions_valid(obj_positions):
            for i, (x1, y1) in enumerate(obj_positions):
                for j, (x2, y2) in enumerate(obj_positions):
                    if i != j and abs(x1 - x2) + abs(y1 - y2) == 1:
                        return False  # Son adyacentes
            return True

        # Coloca 3 objetos aleatorios en el grid (no adyacentes entre sí)
        self.objects = []
        attempts = 0
        while len(self.objects) < 3 and attempts < 1000:
            candidate = (random.randint(1, 5), random.randint(1, 5))
            if candidate in self.objects:
                continue
            temp = self.objects + [candidate]
            if _are_positions_valid(temp):
                self.objects.append(candidate)
            attempts += 1

        if len(self.objects) < 3:
            raise RuntimeError("No se pudieron colocar 3 objetos no adyacentes después de muchos intentos.")

        # Coloca a los agentes aleatoriamente (en celdas libres)
        occupied_positions = set(self.objects)
        while True:
            pos1 = (random.randint(0, 6), random.randint(0, 6))
            if pos1 not in occupied_positions:
                occupied_positions.add(pos1)
                self.agent1_x, self.agent1_y = pos1
                break

        while True:
            pos2 = (random.randint(0, 6), random.randint(0, 6))
            if pos2 not in occupied_positions:
                self.agent2_x, self.agent2_y = pos2
                break

        obs1 = self._get_observation_for_agent("agent_1")
        obs2 = self._get_observation_for_agent("agent_2")

        observations = {
            "agent_1": obs1,
            "agent_2": obs2
        }

        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        infos = {a: {} for a in self.agents}
        return observations, infos

    def observe(self, agent):
        return self._get_observation_for_agent(agent)

    def step(self, actions):
        a1_action = actions["agent_1"]
        a2_action = actions["agent_2"]

        prev_a1_x, prev_a1_y = self.agent1_x, self.agent1_y
        prev_a2_x, prev_a2_y = self.agent2_x, self.agent2_y

        self._move_agent("agent_1", a1_action)
        self._move_agent("agent_2", a2_action)

        rewards = {"agent_1": -0.01, "agent_2": -0.01}  # penalización leve por paso

        # Pick-up automático si ambos están adyacentes al mismo objeto
        for i, (ox, oy) in enumerate(self.objects):
            if i in self.collected:
                continue
            if (
                    self._is_adjacent(self.agent1_x, self.agent1_y, ox, oy)
                    and self._is_adjacent(self.agent2_x, self.agent2_y, ox, oy)
            ):
                self.collected.add(i)
                rewards["agent_1"] += 5.0
                rewards["agent_2"] += 5.0
                break

        # Recompensas intermedias

        for agent_name, (x_now, y_now, x_prev, y_prev) in zip(
                ["agent_1", "agent_2"],
                [(self.agent1_x, self.agent1_y, prev_a1_x, prev_a1_y),
                 (self.agent2_x, self.agent2_y, prev_a2_x, prev_a2_y)]
        ):
            remaining_objects = [(ox, oy) for i, (ox, oy) in enumerate(self.objects) if i not in self.collected]
            if remaining_objects:
                best_prev_dist = min(abs(x_prev - ox) + abs(y_prev - oy) for (ox, oy) in remaining_objects)
                best_new_dist = min(abs(x_now - ox) + abs(y_now - oy) for (ox, oy) in remaining_objects)

                if best_new_dist < best_prev_dist:
                    rewards[agent_name] += 0.1
                elif best_new_dist > best_prev_dist:
                    rewards[agent_name] -= 0.02

        # Recompensa por estar adyacente a un objeto
        for i, (ox, oy) in enumerate(self.objects):
            if i in self.collected:
                continue
            if self._is_adjacent(self.agent1_x, self.agent1_y, ox, oy):
                rewards["agent_1"] += 0.1
            if self._is_adjacent(self.agent2_x, self.agent2_y, ox, oy):
                rewards["agent_2"] += 0.1

        # Recompensa extra si ambos están adyacentes al mismo objeto
        for i, (ox, oy) in enumerate(self.objects):
            if i in self.collected:
                continue
            if (
                    self._is_adjacent(self.agent1_x, self.agent1_y, ox, oy)
                    and self._is_adjacent(self.agent2_x, self.agent2_y, ox, oy)
            ):
                rewards["agent_1"] += 0.5
                rewards["agent_2"] += 0.5

        done = len(self.collected) == len(self.objects)
        terminations = {a: done for a in self.agents}
        truncations = {a: self.timestep > 100 for a in self.agents}

        self.terminations = terminations
        self.truncations = truncations
        self.infos = {a: {} for a in self.possible_agents}

        if done or self.timestep > 100:
            self.agents = []

        observations = {
            "agent_1": self._get_observation_for_agent("agent_1"),
            "agent_2": self._get_observation_for_agent("agent_2")
        }

        self.timestep += 1
        return observations, rewards, terminations, truncations, self.infos

    def _get_observation_for_agent(self, agent):
        # Posiciones de ambos agentes
        a1 = (self.agent1_x, self.agent1_y)
        a2 = (self.agent2_x, self.agent2_y)

        if agent == "agent_1":
            my_x, my_y = a1
            other_x, other_y = a2
        else:
            my_x, my_y = a2
            other_x, other_y = a1

        # Coordenadas normalizadas
        my_pos = [my_x / 6, my_y / 6]
        other_pos = [other_x / 6, other_y / 6]

        object_coords = []
        object_distances = []

        for i, (ox, oy) in enumerate(self.objects):
            if i in self.collected:
                # Objeto recogido → usa marcador especial
                object_coords.extend([-1.0, -1.0])
                object_distances.append(2.0)  # máximo valor de distancia normalizada (12/6)
            else:
                object_coords.extend([ox / 6, oy / 6])
                dist = abs(my_x - ox) + abs(my_y - oy)
                object_distances.append(dist / 12)  # normalizamos distancia máxima (manhattan) posible

        # Distancia al otro agente
        dist_to_other = abs(my_x - other_x) + abs(my_y - other_y)
        dist_to_other /= 12  # normalización máxima

        obs = my_pos + other_pos + object_coords + object_distances + [dist_to_other]
        return np.array(obs, dtype=np.float32)

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
        return np.array([agent1_pos, agent2_pos, *object_pos], dtype=np.float32)

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


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # 18 valores: 2 agentes (x,y) + 3 objetos (x,y) + 4 distancias (3 objetos + otro agente)
        low = np.array([0.0] * 4 + [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0] + [0.0] * 4, dtype=np.float32)
        high = np.array([1.0] * 4 + [1.0] * 6 + [1.0] * 4, dtype=np.float32)
        return Box(low=low, high=high, dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)  # 0:left, 1:right, 2:down, 3:up, 4:pick_up
