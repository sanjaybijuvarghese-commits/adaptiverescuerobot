# ==================================================
# STATE ENCODER
# Compact discrete state for tabular Q-learning
# ==================================================


def get_battery_category(battery):
    """
    Convert battery percentage into a discrete category.

    0 = HIGH
    1 = MEDIUM
    2 = LOW
    3 = CRITICAL
    """

    if battery >= 75:
        return 0

    elif battery >= 50:
        return 1

    elif battery >= 25:
        return 2

    else:
        return 3


def get_health_category(health):
    """
    Convert survivor health into a discrete category.

    0 = HIGH
    1 = MEDIUM
    2 = LOW
    3 = CRITICAL
    """

    if health >= 75:
        return 0

    elif health >= 50:
        return 1

    elif health >= 25:
        return 2

    else:
        return 3


def get_survivor_direction(robot, survivor):
    """
    Find the main direction of the target survivor
    relative to the robot.

    0 = NO TARGET
    1 = UP
    2 = DOWN
    3 = LEFT
    4 = RIGHT
    5 = SAME CELL
    """

    if survivor is None:
        return 0

    robot_row, robot_col = robot
    survivor_row, survivor_col = survivor

    if robot == survivor:
        return 5

    row_difference = survivor_row - robot_row
    col_difference = survivor_col - robot_col

    # Choose the direction with the larger distance
    if abs(row_difference) >= abs(col_difference):

        if row_difference < 0:
            return 1       # UP
        else:
            return 2       # DOWN

    else:

        if col_difference < 0:
            return 3       # LEFT
        else:
            return 4       # RIGHT


def create_state(
    local_observation,
    robot,
    target_survivor,
    target_health,
    battery
):
    """
    Create a compact discrete state for Q-learning.

    Only the four cells immediately around the robot
    are used from the 3x3 sensor.
    """

    # --------------------------------------------------
    # Extract the four neighbouring cells
    # --------------------------------------------------

    up = local_observation[0][1]

    down = local_observation[2][1]

    left = local_observation[1][0]

    right = local_observation[1][2]

    # --------------------------------------------------
    # Target information
    # --------------------------------------------------

    if target_survivor is None:

        target_direction = 0
        target_health_category = 4

    else:

        target_direction = get_survivor_direction(
            robot,
            target_survivor
        )

        target_health_category = get_health_category(
            target_health
        )

    # --------------------------------------------------
    # Battery information
    # --------------------------------------------------

    battery_category = get_battery_category(
        battery
    )

    # --------------------------------------------------
    # Final discrete state
    # --------------------------------------------------

    state = (
        up,
        down,
        left,
        right,
        target_direction,
        target_health_category,
        battery_category
    )

    return state