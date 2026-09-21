import random
import numpy as np

ACTIONS = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

# Policy configuration

INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.995

def choose_action(q_values, epsilon):
    # Exploration
    if random.random() < epsilon:
        return random.choice(list(ACTIONS.keys()))

    # Exploitation (greedy action)
    max_q = np.max(q_values)

    # Get all actions having the maximum Q-value
    best_actions = [
        action for action, q_value in enumerate(q_values)
        if q_value == max_q
    ]

    # Randomly choose among equally good actions
    return random.choice(best_actions)

def decay_epsilon(epsilon):
    epsilon = epsilon * EPSILON_DECAY
    return max(MIN_EPSILON, epsilon)

def calculate_reward(
    rescued=False,
    survivor_health=None,
    hit_obstacle=False,
    entered_hazard=False,
    survivor_lost=False,
    battery_depleted=False
):
    reward = -1

    if hit_obstacle:
        reward -= 10

    if entered_hazard:
        reward -= 30

    if rescued:
        if survivor_health is not None:
            if survivor_health <= 20:
                reward += 150
            elif survivor_health <= 50:
                reward += 120
            else:
                reward += 100

    if survivor_lost:
        reward -= 100

    if battery_depleted:
        reward -= 100

    return reward
