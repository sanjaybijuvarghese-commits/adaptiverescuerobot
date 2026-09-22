import random
import pickle
import pygame

from config import *
from environment import Environment
from visualization import Visualization


class QLearningAgent:
    def __init__(self, actions=[0, 1, 2, 3], lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05):
        self.actions = actions
        self.lr = lr                      # Learning rate (alpha)
        self.gamma = gamma                # Discount factor
        self.epsilon = epsilon            # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Tabular Q-values stored as: {state_tuple: [q0, q1, q2, q3]}
        self.q_table = {}

    def get_q_values(self, state):
        """Returns Q-values for a state, initializing with zeros if unvisited."""
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in self.actions]
        return self.q_table[state]

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        q_values = self.get_q_values(state)
        max_val = max(q_values)
        # Random tie-breaking among best actions
        best_actions = [a for a, val in enumerate(q_values) if val == max_val]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """Bellman equation update for Q-Learning."""
        current_q = self.get_q_values(state)[action]

        if done:
            target = reward
        else:
            max_next_q = max(self.get_q_values(next_state))
            target = reward + self.gamma * max_next_q

        # Temporal difference update
        self.q_table[state][action] = current_q + self.lr * (target - current_q)

    def decay_epsilon(self):
        """Decays exploration probability after each episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_policy(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_policy(self, filename="q_table.pkl"):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)


# ==========================================================
# FAST TRAINING PIPELINE (No Graphics)
# ==========================================================

def train(episodes=2000):
    env = Environment()
    agent = QLearningAgent(lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.997, min_epsilon=0.05)

    print(f"--- Starting Training ({episodes} episodes) ---")

    for ep in range(1, episodes + 1):
        # env.reset() already returns the compact encoded tuple directly
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        if ep % 100 == 0:
            rescued_count = len(env.rescued_survivors)
            print(f"Ep {ep:4d}/{episodes} | Epsilon: {agent.epsilon:.3f} | Reward: {total_reward:4d} | Rescued: {rescued_count}/{env.num_survivors} | Ended: {info.get('reason')}")

    agent.save_policy("q_table.pkl")
    print("Training finished. Q-table saved to 'q_table.pkl'.\n")
    return agent


# ==========================================================
# TEST SIMULATION (Live Pygame Window)
# ==========================================================

def run_simulation(agent=None):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Adaptive Rescue Robot - Q-Learning Simulation")
    clock = pygame.time.Clock()

    env = Environment()
    viz = Visualization()

    if agent is None:
        agent = QLearningAgent(epsilon=0.0)  # Pure exploitation
        try:
            agent.load_policy("q_table.pkl")
            print("Loaded trained policy from q_table.pkl")
        except FileNotFoundError:
            print("Warning: q_table.pkl not found! Robot will act randomly.")

    agent.epsilon = 0.0

    state = env.reset()
    running = True
    step_delay_ms = 200  # Delay per step in ms so movement is visible
    last_step_time = pygame.time.get_ticks()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Press R to reset scenario
                    state = env.reset()

        # Step agent on timer
        now = pygame.time.get_ticks()
        if now - last_step_time >= step_delay_ms and not env.terminated:
            action = agent.choose_action(state)
            next_state, _, done, _ = env.step(action)
            state = next_state
            last_step_time = now

        viz.draw(screen, env)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    # 1. Train agent headlessly
    trained_agent = train(episodes=2000)

    # 2. Watch learned policy in Pygame
    run_simulation(trained_agent)