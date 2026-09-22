def calculate_reward(
    moved=False,
    hit_obstacle=False,
    entered_hazard=False,
    rescued_survivor=False,
    is_ping_pong=False,
    **kwargs
):
    """
    Robust reward function.
    Accepts **kwargs so custom environmental flags
    (e.g., discovered_survivor, battery_low, etc.) don't raise TypeError.
    """
    # 1. Catastrophic failures
    if entered_hazard:
        return -100.0

    # 2. Collisions
    if hit_obstacle:
        return -15.0

    # 3. High-value achievements
    if rescued_survivor:
        return 150.0

    if kwargs.get("discovered_survivor", False):
        return 20.0

    # 4. Anti-loop penalty
    if is_ping_pong:
        return -10.0

    # 5. Regular movement step cost
    if moved:
        return -1.0

    return 0.0