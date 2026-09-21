from environment import Environment
from state_encoder import create_state


env = Environment()

# Get current local observation
observation = env.get_local_observation()

# Select target using the world model
target_survivor = env.select_target_survivor()

if target_survivor is not None:

    target_health = env.known_survivors[target_survivor]

else:

    target_health = 0


# Create compact RL state
state = create_state(
    local_observation=observation,
    robot=env.robot,
    target_survivor=target_survivor,
    target_health=target_health,
    battery=env.battery
)


print("\n========== STATE TEST ==========")

print("Robot:")
print(env.robot)

print("\n3x3 Sensor:")
for row in observation:
    print(row)

print("\nTarget survivor:")
print(target_survivor)

print("\nTarget health:")
print(target_health)

print("\nBattery:")
print(env.battery)

print("\nCompact RL state:")
print(state)