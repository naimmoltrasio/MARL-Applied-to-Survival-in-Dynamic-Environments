import functools
import random
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv
import pygame


class CollaborativePickUpEnv(ParallelEnv):
    metadata = {
        "name": "collaborative_pickup_v0",
        "render_modes": ["human"],
        "render_fps": 4,
        "is_parallelizable": True
    }

    def __init__(self, render_mode="human", config="config1", reward_mode="basic"):
        self.render_mode = render_mode
        self.config = config
        self.reward_mode = reward_mode
        self.agent1_x = None
        self.agent1_y = None
        self.agent2_x = None
        self.agent2_y = None
        self.objects = []  # [(x, y, type)] with type in {"coop", "solo"}
        self.collected = set()
        self.timestep = 0
        self.possible_agents = ["agent_1", "agent_2"]

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.collected = set()

        occupied = set()
        self.objects = []

        def is_adjacent(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1

        def is_adjacent_to_any(new_pos, positions):
            return any(is_adjacent(new_pos, pos) for pos in positions)

        def place_safe_object(obj_type):
            while True:
                x, y = random.randint(1, 5), random.randint(1, 5)
                new_pos = (x, y)
                existing_positions = [(ox, oy) for ox, oy, _ in self.objects]
                if new_pos not in occupied and not is_adjacent_to_any(new_pos, existing_positions):
                    self.objects.append((x, y, obj_type))
                    occupied.add(new_pos)
                    break

        if self.config == "config1":
            for _ in range(5):
                place_safe_object("solo")
        elif self.config == "config2":
            for _ in range(3):
                place_safe_object("coop")
            for _ in range(2):
                place_safe_object("solo")
        elif self.config == "config3":
            for _ in range(5):
                place_safe_object("coop")

        while True:
            pos1 = (random.randint(0, 6), random.randint(0, 6))
            if pos1 not in occupied:
                self.agent1_x, self.agent1_y = pos1
                occupied.add(pos1)
                break

        while True:
            pos2 = (random.randint(0, 6), random.randint(0, 6))
            if pos2 not in occupied:
                self.agent2_x, self.agent2_y = pos2
                break

        obs1 = self._get_observation_for_agent("agent_1")
        obs2 = self._get_observation_for_agent("agent_2")

        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        return {"agent_1": obs1, "agent_2": obs2}, self.infos

    def _place_object(self, typ, occupied):
        while True:
            x, y = random.randint(1, 5), random.randint(1, 5)
            if (x, y) not in occupied:
                self.objects.append((x, y, typ))
                occupied.add((x, y))
                break

    def apply_reward_logic(self, rewards, i, ox, oy, typ):
        if self.reward_mode == "basic":
            if typ == "coop":
                if (self._is_adjacent(self.agent1_x, self.agent1_y, ox, oy) and
                        self._is_adjacent(self.agent2_x, self.agent2_y, ox, oy)):
                    self.collected.add(i)
                    rewards["agent_1"] += 5.0
                    rewards["agent_2"] += 5.0
            elif typ == "solo":
                if (self._is_adjacent(self.agent1_x, self.agent1_y, ox, oy)):
                    self.collected.add(i)
                    rewards["agent_1"] += 2.0
                elif (self._is_adjacent(self.agent2_x, self.agent2_y, ox, oy)):
                    self.collected.add(i)
                    rewards["agent_2"] += 2.0

        elif self.reward_mode == "dense":
            for agent, (ax, ay) in zip(["agent_1", "agent_2"],
                                       [(self.agent1_x, self.agent1_y), (self.agent2_x, self.agent2_y)]):
                dist = abs(ax - ox) + abs(ay - oy)
                rewards[agent] += max(0, 1 - dist / 6) * 0.1  # reward por cercanía

            if typ == "coop":
                if (self._is_adjacent(self.agent1_x, self.agent1_y, ox, oy) and
                        self._is_adjacent(self.agent2_x, self.agent2_y, ox, oy)):
                    self.collected.add(i)
                    rewards["agent_1"] += 4.0
                    rewards["agent_2"] += 4.0
            elif typ == "solo":
                if (self._is_adjacent(self.agent1_x, self.agent1_y, ox, oy)):
                    self.collected.add(i)
                    rewards["agent_1"] += 1.5
                elif (self._is_adjacent(self.agent2_x, self.agent2_y, ox, oy)):
                    self.collected.add(i)
                    rewards["agent_2"] += 1.5


        elif self.reward_mode == "coordination":
            # Recompensa adicional por cercanía al otro agente (solo si el objeto es cooperativo)
            if typ == "coop":
                d = abs(self.agent1_x - self.agent2_x) + abs(self.agent1_y - self.agent2_y)
                if d < 4:
                    rewards["agent_1"] += 0.2
                    rewards["agent_2"] += 0.2
            # Recompensa por cercanía al objeto (aplicado siempre)
            for agent, (ax, ay) in zip(["agent_1", "agent_2"],
                                       [(self.agent1_x, self.agent1_y), (self.agent2_x, self.agent2_y)]):
                dist = abs(ax - ox) + abs(ay - oy)
                rewards[agent] += max(0, 1 - dist / 6) * 0.1
            # Recompensa por recogida del objeto
            if typ == "coop":
                if (self._is_adjacent(self.agent1_x, self.agent1_y, ox, oy) and
                        self._is_adjacent(self.agent2_x, self.agent2_y, ox, oy)):
                    self.collected.add(i)
                    rewards["agent_1"] += 6.0
                    rewards["agent_2"] += 6.0
            elif typ == "solo":
                if (self._is_adjacent(self.agent1_x, self.agent1_y, ox, oy)):
                    self.collected.add(i)
                    rewards["agent_1"] += 1.0
                elif (self._is_adjacent(self.agent2_x, self.agent2_y, ox, oy)):
                    self.collected.add(i)
                    rewards["agent_2"] += 1.0

    def step(self, actions):
        a1_action = actions["agent_1"]
        a2_action = actions["agent_2"]

        self._move_agent("agent_1", a1_action)
        self._move_agent("agent_2", a2_action)

        rewards = {"agent_1": -0.05, "agent_2": -0.05}

        for i, (ox, oy, typ) in enumerate(self.objects):
            if i in self.collected:
                continue
            self.apply_reward_logic(rewards, i, ox, oy, typ)

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
        a1 = (self.agent1_x, self.agent1_y)
        a2 = (self.agent2_x, self.agent2_y)

        if agent == "agent_1":
            my_x, my_y = a1
            other_x, other_y = a2
        else:
            my_x, my_y = a2
            other_x, other_y = a1

        my_pos = [my_x / 6, my_y / 6]
        other_pos = [other_x / 6, other_y / 6]

        object_coords = []
        object_distances = []

        for i, (ox, oy, _) in enumerate(self.objects):
            if i in self.collected:
                object_coords.extend([-1.0, -1.0])
                object_distances.append(2.0)
            else:
                object_coords.extend([ox / 6, oy / 6])
                dist = abs(my_x - ox) + abs(my_y - oy)
                object_distances.append(dist / 12)

        dist_to_other = abs(my_x - other_x) + abs(my_y - other_y)
        dist_to_other /= 12

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

        if (new_x, new_y) in [(ox, oy) for i, (ox, oy, _) in enumerate(self.objects) if i not in self.collected]:
            return
        if (new_x, new_y) == (other_x, other_y):
            return

        if agent == "agent_1":
            self.agent1_x, self.agent1_y = new_x, new_y
        else:
            self.agent2_x, self.agent2_y = new_x, new_y

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

            # Cargar imágenes una sola vez
            self.apple_img = pygame.image.load("apple.png")
            self.apple_img = pygame.transform.scale(self.apple_img, (50, 50))
            self.deer_img = pygame.image.load("deer.png")
            self.deer_img = pygame.transform.scale(self.deer_img, (60, 60))
            self.human_img = pygame.image.load("human.png")
            self.human_img = pygame.transform.scale(self.human_img, (60, 60))

        self.screen.fill((255, 255, 255))

        for y in range(7):
            for x in range(7):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        font = pygame.font.SysFont(None, 24)
        for i, (x, y, typ) in enumerate(self.objects):
            if i in self.collected:
                continue
            center = (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2)
            if typ == "coop":
                img_rect = self.deer_img.get_rect(center=center)
                self.screen.blit(self.deer_img, img_rect)

                label = font.render("2", True, (0, 255, 0))
                label_rect = label.get_rect(center=center)
                self.screen.blit(label, label_rect)
            else:
                img_rect = self.apple_img.get_rect(center=center)
                self.screen.blit(self.apple_img, img_rect)
                label = font.render("1", True, (255, 255, 255))
                label_rect = label.get_rect(center=center)
                self.screen.blit(label, label_rect)

        agent1_pos = (self.agent1_x * cell_size + 10, self.agent1_y * cell_size + 10)
        agent2_pos = (self.agent2_x * cell_size + 10, self.agent2_y * cell_size + 10)

        self.screen.blit(self.human_img, agent1_pos)
        self.screen.blit(self.human_img, agent2_pos)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if self.config == "config1":
            n_objects = 5
        elif self.config == "config2":
            n_objects = 5
        elif self.config == "config3":
            n_objects = 5
        else:
            n_objects = 3  # fallback

        dim = 2 + 2 + n_objects * 2 + n_objects + 1
        return Box(low=np.full(dim, -1.0), high=np.full(dim, 1.0), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)



