import random
from config import *
from state_encoder import create_state
from reward import calculate_reward


class Environment:
    def __init__(self, grid_size=10, max_steps=100):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.step_count = 0
        self.battery = 100
        self.terminated = False

        # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_deltas = {
            0: (-1, 0),  # row - 1
            1: (0, 1),   # col + 1
            2: (1, 0),   # row + 1
            3: (0, -1)   # col - 1
        }

        self.robot = [0, 0]
        self.obstacles = set()
        self.hazards = set()
        self.survivors = []
        self.survivor_healths = {}
        self.rescued_survivors = []
        self.visited_tiles = []

    def get_closest_survivor(self):
        active = [s for s in self.survivors if tuple(s) not in self.rescued_survivors]
        if not active:
            return None
        return min(active, key=lambda s: abs(s[0] - self.robot[0]) + abs(s[1] - self.robot[1]))

    def get_local_observation(self):
        r, c = self.robot
        obs = [[0 for _ in range(3)] for _ in range(3)]
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                    obs[dr + 1][dc + 1] = 1
                elif (nr, nc) in self.obstacles:
                    obs[dr + 1][dc + 1] = 1
                elif (nr, nc) in self.hazards:
                    obs[dr + 1][dc + 1] = 2
                else:
                    obs[dr + 1][dc + 1] = 0
        return obs

    def get_state(self):
        target = self.get_closest_survivor()
        target_health = self.survivor_healths.get(tuple(target), 100) if target else 0
        local_obs = self.get_local_observation()
        return create_state(local_obs, self.robot, target, target_health, self.battery)

    def reset(self, random_layout=True):
        self.step_count = 0
        self.battery = 100
        self.terminated = False
        self.rescued_survivors = []
        self.visited_tiles = []

        all_coords = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        samples = random.sample(all_coords, 18)

        self.robot = list(samples[0])
        self.survivors = [list(samples[1]), list(samples[2]), list(samples[3])]
        self.survivor_healths = {tuple(s): 100 for s in self.survivors}
        self.obstacles = set(samples[4:8])
        self.hazards = set(samples[8:18])

        self.visited_tiles.append(tuple(self.robot))
        return self.get_state()

    def spread_fire(self, spread_chance=0.05, max_total_fires=20):
        """
        Slow, controlled fire spread:
        - Only checks every 3 steps.
        - Picks at most 2 fire fronts to spread outwards.
        - 5% chance per open neighbor.
        - Hard-capped at 20 fire tiles total.
        """
        # Spread only every 3 steps
        if self.step_count % 3 != 0:
            return

        if len(self.hazards) >= max_total_fires:
            return

        new_fires = set()
        active_survivors = set(tuple(s) for s in self.survivors if tuple(s) not in self.rescued_survivors)
        forbidden = self.obstacles | active_survivors | {tuple(self.robot)}

        current_hazards = list(self.hazards)
        sample_size = min(len(current_hazards), 2)
        candidate_sources = random.sample(current_hazards, sample_size)

        for r, c in candidate_sources:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if (nr, nc) not in self.hazards and (nr, nc) not in forbidden:
                        if random.random() < spread_chance:
                            new_fires.add((nr, nc))

        self.hazards.update(new_fires)

    def step(self, action):
        self.step_count += 1
        self.battery = max(0, self.battery - 1)
        dr, dc = self.action_deltas[action]
        new_r = self.robot[0] + dr
        new_c = self.robot[1] + dc

        done = False
        info = {"reason": "ongoing"}

        # 1. Out of bounds or obstacle collision
        if not (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size) or (new_r, new_c) in self.obstacles:
            reward = calculate_reward(hit_obstacle=True)
            info["reason"] = "obstacle_collision"

        # 2. Stepped into existing fire
        elif (new_r, new_c) in self.hazards:
            reward = calculate_reward(entered_hazard=True)
            done = True
            self.terminated = True
            self.robot = [new_r, new_c]
            info["reason"] = "burned_in_fire"

        # 3. Survivor rescued
        elif [new_r, new_c] in self.survivors and tuple([new_r, new_c]) not in self.rescued_survivors:
            reward = calculate_reward(rescued_survivor=True)
            self.robot = [new_r, new_c]
            self.rescued_survivors.append((new_r, new_c))

            if len(self.rescued_survivors) >= len(self.survivors):
                done = True
                self.terminated = True
                info["reason"] = "mission_complete"
            else:
                info["reason"] = "rescued_target"

        # 4. Standard move
        else:
            new_tile = (new_r, new_c)
            is_ping_pong = len(self.visited_tiles) >= 2 and new_tile == self.visited_tiles[-2]
            reward = calculate_reward(moved=True, is_ping_pong=is_ping_pong)
            self.robot = [new_r, new_c]

        self.visited_tiles.append(tuple(self.robot))

        # --- SLOW FIRE SPREAD ---
        if not done:
            self.spread_fire(spread_chance=0.05, max_total_fires=20)
            if tuple(self.robot) in self.hazards:
                done = True
                self.terminated = True
                reward = calculate_reward(entered_hazard=True)
                info["reason"] = "engulfed_in_fire"

        if (self.step_count >= self.max_steps or self.battery <= 0) and not done:
            done = True
            self.terminated = True
            info["reason"] = "battery_empty" if self.battery <= 0 else "max_steps_exceeded"

        return self.get_state(), reward, done, info