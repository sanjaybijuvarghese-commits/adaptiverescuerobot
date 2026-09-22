import random
import pickle
import pygame

from config import *
from environment import Environment
from visualization import Visualization


class SARSAAgent:
    def __init__(self, actions=[0, 1, 2, 3], lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.9992, min_epsilon=0.02):
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in self.actions]
        return self.q_table[state]

    def choose_action(self, state, allowed_actions=None):
        candidates = allowed_actions if (allowed_actions is not None and len(allowed_actions) > 0) else self.actions

        if random.random() < self.epsilon:
            return random.choice(candidates)

        q_vals = self.get_q_values(state)
        best_val = max(q_vals[a] for a in candidates)
        best_actions = [a for a in candidates if q_vals[a] == best_val]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_action, done):
        current_q = self.get_q_values(state)[action]
        if done:
            target = reward
        else:
            next_q = self.get_q_values(next_state)[next_action]
            target = reward + self.gamma * next_q

        self.q_table[state][action] = current_q + self.lr * (target - current_q)

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_policy(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_policy(self, filename="q_table.pkl"):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)


def train(episodes=6000, max_steps_per_episode=100):
    env = Environment()
    agent = SARSAAgent(lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.9992, min_epsilon=0.02)
    opposite = {0: 2, 2: 0, 1: 3, 3: 1}

    print(f"--- Starting SARSA Training ({episodes} episodes) ---")

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        info = {}

        action = agent.choose_action(state)

        while not done and steps < max_steps_per_episode:
            next_state, reward, done, info = env.step(action)

            avoid = opposite.get(action)
            allowed = [a for a in [0, 1, 2, 3] if a != avoid] if avoid is not None else [0, 1, 2, 3]
            next_action = agent.choose_action(next_state, allowed_actions=allowed)

            agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

        agent.decay_epsilon()

        if ep % 200 == 0:
            rescued_count = len(env.rescued_survivors)
            total_survivors = len(env.survivors)
            reason = info.get("reason", "done")
            print(f"Ep {ep:4d}/{episodes} | Epsilon: {agent.epsilon:.3f} | Steps: {steps:3d} | Reward: {total_reward:6.1f} | Rescued: {rescued_count}/{total_survivors} | Ended: {reason}")

    agent.save_policy("q_table.pkl")
    print("Training finished. Policy saved to 'q_table.pkl'.\n")
    return agent


def run_simulation(agent=None):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Adaptive Rescue Robot - SARSA Simulation")
    clock = pygame.time.Clock()

    env = Environment()
    viz = Visualization()

    if agent is None:
        agent = SARSAAgent(epsilon=0.0)
        try:
            agent.load_policy("q_table.pkl")
            print("Loaded trained policy from q_table.pkl")
        except FileNotFoundError:
            print("Warning: q_table.pkl not found!")

    agent.epsilon = 0.0

    state = env.reset()
    running = True
    step_delay_ms = 200
    last_step_time = pygame.time.get_ticks()

    deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    pos_history = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    state = env.reset()
                    pos_history.clear()

        now = pygame.time.get_ticks()
        if now - last_step_time >= step_delay_ms and not env.terminated:
            current_pos = tuple(env.robot)
            pos_history.append(current_pos)
            if len(pos_history) > 8:
                pos_history.pop(0)

            # 1. Target Snapping to ANY remaining unrescued survivor
            survivor_targets = [tuple(s) for s in env.survivors if tuple(s) not in env.rescued_survivors]
            immediate_rescue_action = None
            for a in [0, 1, 2, 3]:
                dr, dc = deltas[a]
                if (current_pos[0] + dr, current_pos[1] + dc) in survivor_targets:
                    immediate_rescue_action = a
                    break

            allowed = [0, 1, 2, 3]

            if immediate_rescue_action is not None:
                action = immediate_rescue_action
            else:
                # 2. Avoid moving into recently visited tiles
                candidate_allowed = []
                for a in [0, 1, 2, 3]:
                    dr, dc = deltas[a]
                    dest = (current_pos[0] + dr, current_pos[1] + dc)
                    if dest not in pos_history[-3:]:
                        candidate_allowed.append(a)

                if candidate_allowed:
                    allowed = candidate_allowed

                action = agent.choose_action(state, allowed_actions=allowed)

            next_state, reward, done, info = env.step(action)
            state = next_state
            new_pos = tuple(env.robot)

            q_vals = [round(v, 1) for v in agent.get_q_values(state)]
            print(f"Pos: {current_pos} -> Action: {action_names[action]} -> New Pos: {new_pos} | Rescued: {len(env.rescued_survivors)}/{len(env.survivors)} | Q: {q_vals}")

            last_step_time = now

        viz.draw(screen, env)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    trained_agent = train(episodes=6000, max_steps_per_episode=100)
    run_simulation(trained_agent)