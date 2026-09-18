def calculate_reward(
    moved,
    hit_obstacle,
    entered_hazard,
    discovered_survivor,
    rescued_survivor,
    survivor_lost,
    battery_empty,
    mission_complete
):

    reward = 0

    # 1. Normal movement
    if moved:
        reward -= 1

    # 2. Hit obstacle / invalid movement
    if hit_obstacle:
        reward -= 5

    # 3. Entered hazard
    if entered_hazard:
        reward -= 20

    # 4. Discovered a survivor
    if discovered_survivor:
        reward += 10

    # 5. Rescued a survivor
    if rescued_survivor:
        reward += 50

    # 6. Survivor was lost
    if survivor_lost:
        reward -= 50

    # 7. Battery became empty
    if battery_empty:
        reward -= 30

    # 8. All survivors rescued
    if mission_complete:
        reward += 100

    return reward