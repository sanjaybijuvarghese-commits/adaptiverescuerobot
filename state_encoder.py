# ==================================================
# STATE ENCODER
# Converts environment information into a
# compact discrete state for tabular Q-learning
# ==================================================


def encode_sensor(observation):
    """
    Convert the 3x3 sensor matrix into a tuple.
    """

    return tuple(
        value
        for row in observation
        for value in row
    )


def get_battery_category(battery):
    """
    Convert battery percentage into 4 categories.
    """

    if battery >= 75:
        return 0          # HIGH

    elif battery >= 50:
        return 1          # MEDIUM

    elif battery >= 25:
        return 2          # LOW

    else:
        return 3          # CRITICAL


def get_health_category(health):
    """
    Convert survivor health into 4 categories.
    """

    if health >= 75:
        return 0          # HIGH

    elif health >= 50:
        return 1          # MEDIUM

    elif health >= 25:
        return 2          # LOW

    else:
        return 3          # CRITICAL


def get_survivor_direction(robot, survivor):
    """
    Find the approximate direction of a survivor
    relative to the robot.

    Returns:
        0 = SAME CELL
        1 = UP
        2 = DOWN
        3 = LEFT
        4 = RIGHT
        5 = UNKNOWN
    """

    robot_row, robot_col = robot
    survivor_row, survivor_col = survivor

    # Same position
    if robot == survivor:
        return 0

    # Compare vertical and horizontal distance
    row_difference = survivor_row - robot_row
    col_difference = survivor_col - robot_col

    # Choose the dominant direction
    if abs(row_difference) >= abs(col_difference):

        if row_difference < 0:
            return 1     # UP

        else:
            return 2     # DOWN

    else:

        if col_difference < 0:
            return 3     # LEFT

        else:
            return 4     # RIGHT


def create_state(
    robot,
    local_observation,
    target_survivor,
    target_health,
    battery
):
    """
    Create the final discrete state for Q-learning.
    """

    # Convert 3x3 observation into a tuple
    sensor_state = encode_sensor(local_observation)

    # Battery category
    battery_state = get_battery_category(
        battery
    )

    # Target information
    if target_survivor is None:

        target_direction = 5
        health_state = 3

    else:

        target_direction = get_survivor_direction(
            robot,
            target_survivor
        )

        health_state = get_health_category(
            target_health
        )

    # Final state
    state = (
        sensor_state,
        target_direction,
        health_state,
        battery_state
    )

    return state