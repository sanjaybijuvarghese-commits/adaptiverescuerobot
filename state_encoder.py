# ==================================================
# STATE ENCODER
# Discrete state for tabular SARSA
# ==================================================

def get_battery_category(battery):
    if battery >= 75:
        return 0
    elif battery >= 50:
        return 1
    elif battery >= 25:
        return 2
    else:
        return 3


def get_health_category(health):
    if health >= 75:
        return 0
    elif health >= 50:
        return 1
    elif health >= 25:
        return 2
    else:
        return 3


def get_survivor_direction(robot, survivor):
    if survivor is None:
        return (0, 0)

    robot_row, robot_col = robot
    survivor_row, survivor_col = survivor

    r_diff = survivor_row - robot_row
    c_diff = survivor_col - robot_col

    if r_diff < 0:
        r_dir = 1   # UP
    elif r_diff > 0:
        r_dir = 2   # DOWN
    else:
        r_dir = 0   # ALIGNED

    if c_diff < 0:
        c_dir = 1   # LEFT
    elif c_diff > 0:
        c_dir = 2   # RIGHT
    else:
        c_dir = 0   # ALIGNED

    return (r_dir, c_dir)


def create_state(local_observation, robot, target_survivor, target_health, battery):
    up = local_observation[0][1]
    down = local_observation[2][1]
    left = local_observation[1][0]
    right = local_observation[1][2]

    if target_survivor is None:
        target_r_dir, target_c_dir = 0, 0
        target_health_cat = 4
    else:
        target_r_dir, target_c_dir = get_survivor_direction(robot, target_survivor)
        target_health_cat = get_health_category(target_health)

    battery_cat = get_battery_category(battery)

    return (
        up,
        down,
        left,
        right,
        target_r_dir,
        target_c_dir,
        target_health_cat,
        battery_cat
    )