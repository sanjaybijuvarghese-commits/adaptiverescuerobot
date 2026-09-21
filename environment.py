import random
from reward import calculate_reward

from config import (
    ROWS,
    COLS,
    NUM_SURVIVORS,
    NUM_HAZARDS,
    NUM_OBSTACLES,
    MAX_BATTERY,
    HAZARD_SPREAD_INTERVAL
)


class Environment:

    def __init__(self):

        self.rows = ROWS
        self.cols = COLS

        self.max_battery = MAX_BATTERY

        self.num_survivors = NUM_SURVIVORS
        self.num_hazards = NUM_HAZARDS
        self.num_obstacles = NUM_OBSTACLES

        self.hazard_spread_interval = HAZARD_SPREAD_INTERVAL

                # -----------------------------
        # Robot's world model / memory
        # -----------------------------

        self.known_cells = {}

        self.known_survivors = {}

        self.known_hazards = set()

        self.known_obstacles = set()

        # Start first episode
        self.reset()

    # ==================================================
    # RESET
    # Starts a completely new episode
    # ==================================================

    def reset(self):

        # Episode step counter
        self.steps = 0

        # Robot
        self.robot = None

        # Environment objects
        self.survivors = []
        self.hazards = []
        self.obstacles = []

        # Survivor health
        self.survivor_health = {}

        # Survivor status
        self.rescued_survivors = set()
        self.lost_survivors = set()

                # Reset robot's world model / memory
        self.known_cells = {}
        self.known_survivors = {}
        self.known_hazards = set()
        self.known_obstacles = set()

        # Robot battery
        self.battery = self.max_battery

        # Episode status
        self.terminated = False

        # Generate new disaster
        self.generate_environment()

        self.update_world_model()

        return self.get_state()

    # ==================================================
    # FIND RANDOM EMPTY CELL
    # ==================================================

    def random_empty_cell(self):

        while True:

            row = random.randint(0, self.rows - 1)
            col = random.randint(0, self.cols - 1)

            cell = (row, col)

            if cell == self.robot:
                continue

            if cell in self.survivors:
                continue

            if cell in self.hazards:
                continue

            if cell in self.obstacles:
                continue

            return cell

    # ==================================================
    # CHECK WHETHER CELL IS FREE
    # ==================================================

    def is_free(self, cell):

        if cell == self.robot:
            return False

        if cell in self.survivors:
            return False

        if cell in self.hazards:
            return False

        if cell in self.obstacles:
            return False

        return True

    # ==================================================
    # GENERATE DISASTER ENVIRONMENT
    # ==================================================

    def generate_environment(self):

        # -----------------------------
        # Robot
        # -----------------------------

        self.robot = self.random_empty_cell()

        # -----------------------------
        # Survivors
        # -----------------------------

        for _ in range(self.num_survivors):

            position = self.random_empty_cell()

            self.survivors.append(position)

            # Random health
            self.survivor_health[position] = random.randint(
                50,
                100
            )

        # -----------------------------
        # Hazards
        # -----------------------------

        for _ in range(self.num_hazards):

            position = self.random_empty_cell()

            self.hazards.append(position)

        # -----------------------------
        # Two adjacent obstacles
        # -----------------------------

        while True:

            row = random.randint(
                0,
                self.rows - 1
            )

            col = random.randint(
                0,
                self.cols - 2
            )

            obstacle1 = (row, col)
            obstacle2 = (row, col + 1)

            if (
                self.is_free(obstacle1)
                and self.is_free(obstacle2)
            ):

                self.obstacles.append(obstacle1)
                self.obstacles.append(obstacle2)

                break

        # -----------------------------
        # Remaining obstacles
        # -----------------------------

        while len(self.obstacles) < self.num_obstacles:

            obstacle = self.random_empty_cell()

            self.obstacles.append(obstacle)

    # ==================================================
    # MOVE ROBOT
    # ==================================================

    def move_robot(self, action):

        row, col = self.robot

        # 0 = UP
        if action == 0:

            new_row = row - 1
            new_col = col

        # 1 = DOWN
        elif action == 1:

            new_row = row + 1
            new_col = col

        # 2 = LEFT
        elif action == 2:

            new_row = row
            new_col = col - 1

        # 3 = RIGHT
        elif action == 3:

            new_row = row
            new_col = col + 1

        else:

            raise ValueError(
                "Action must be 0, 1, 2, or 3."
            )

        new_position = (
            new_row,
            new_col
        )

        # -----------------------------
        # Boundary check
        # -----------------------------

        if (
            new_row < 0
            or new_row >= self.rows
            or new_col < 0
            or new_col >= self.cols
        ):

            return False, True, False

        # -----------------------------
        # Obstacle check
        # -----------------------------

        if new_position in self.obstacles:

            return False, True, False

        # -----------------------------
        # Move robot
        # -----------------------------

        self.robot = new_position

        # Check hazard
        entered_hazard = (
            self.robot in self.hazards
        )

        return True, False, entered_hazard


        # ==================================================
    # GET 3x3 LOCAL SENSOR OBSERVATION
    # ==================================================

    def get_local_observation(self):

        row, col = self.robot

        observation = []

        # Check 3x3 area around robot
        for r in range(row - 1, row + 2):

            row_observation = []

            for c in range(col - 1, col + 2):

                cell = (r, c)

                # Outside the grid
                if (
                    r < 0
                    or r >= self.rows
                    or c < 0
                    or c >= self.cols
                ):
                    row_observation.append(1)
                    continue

                # Robot
                if cell == self.robot:
                    row_observation.append(4)

                # Obstacle
                elif cell in self.obstacles:
                    row_observation.append(1)

                # Hazard
                elif cell in self.hazards:
                    row_observation.append(2)

                # Survivor
                elif cell in self.survivors:
                    row_observation.append(3)

                # Empty
                else:
                    row_observation.append(0)

            observation.append(row_observation)

        return observation

        # ==================================================
    # UPDATE WORLD MODEL
    # ==================================================

    def update_world_model(self):

        row, col = self.robot

        # Check the 3x3 area around the robot
        for r in range(row - 1, row + 2):

            for c in range(col - 1, col + 2):

                # Ignore cells outside the grid
                if (
                    r < 0
                    or r >= self.rows
                    or c < 0
                    or c >= self.cols
                ):
                    continue

                cell = (r, c)

                # Obstacle
                if cell in self.obstacles:

                    self.known_cells[cell] = "obstacle"
                    self.known_obstacles.add(cell)

                # Hazard
                elif cell in self.hazards:

                    self.known_cells[cell] = "hazard"
                    self.known_hazards.add(cell)

                # Survivor
                elif cell in self.survivors:

                    self.known_cells[cell] = "survivor"

                    if cell in self.survivor_health:

                        self.known_survivors[cell] = (
                            self.survivor_health[cell]
                        )

                # Robot
                elif cell == self.robot:

                    self.known_cells[cell] = "robot"

                # Empty
                else:

                    self.known_cells[cell] = "empty"

    # ==================================================
    # UPDATE SURVIVOR HEALTH
    # ==================================================

    def update_survivor_health(self):

        newly_lost = []

        for survivor in self.survivors:

            # Already rescued
            if survivor in self.rescued_survivors:
                continue

            # Already lost
            if survivor in self.lost_survivors:
                continue

            # Health decreases
            self.survivor_health[survivor] -= 1

            # Survivor dies
            if self.survivor_health[survivor] <= 0:

                self.survivor_health[survivor] = 0

                self.lost_survivors.add(
                    survivor
                )

                newly_lost.append(
                    survivor
                )

        return newly_lost

    # ==================================================
    # SPREAD HAZARDS
    # ==================================================

    def spread_hazards(self):

        new_hazards = []

        # Copy current hazards
        current_hazards = self.hazards.copy()

        for hazard in current_hazards:

            row, col = hazard

            neighbours = [
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1)
            ]

            for cell in neighbours:

                r, c = cell

                # Boundary
                if r < 0 or r >= self.rows:
                    continue

                if c < 0 or c >= self.cols:
                    continue

                # Cannot spread into obstacle
                if cell in self.obstacles:
                    continue

                # Already a hazard
                if cell in self.hazards:
                    continue

                # Already selected for spreading
                if cell in new_hazards:
                    continue

                # Don't place hazard on robot
                if cell == self.robot:
                    continue

                # Don't place hazard on rescued survivor
                if cell in self.rescued_survivors:
                    continue

                # 25% chance
                if random.random() < 0.25:

                    new_hazards.append(cell)

        self.hazards.extend(new_hazards)

        return new_hazards

    # ==================================================
    # RESCUE SURVIVOR
    # ==================================================

    def rescue_survivor(self):

        # Robot is not on a survivor
        if self.robot not in self.survivors:

            return False

        survivor = self.robot

        # Already rescued
        if survivor in self.rescued_survivors:

            return False

        # Already lost
        if survivor in self.lost_survivors:

            return False

        # Rescue
        self.rescued_survivors.add(
            survivor
        )

        return True

    # ==================================================
    # CHECK WHETHER EPISODE IS OVER
    # ==================================================

    def check_episode_end(self):

        # -----------------------------
        # All survivors rescued
        # -----------------------------

        if len(self.rescued_survivors) == self.num_survivors:

            return True, "mission_complete"

        # -----------------------------
        # Battery empty
        # -----------------------------

        if self.battery <= 0:

            return True, "battery_empty"

        # -----------------------------
        # All survivors lost
        # -----------------------------

        if len(self.lost_survivors) == self.num_survivors:

            return True, "all_survivors_lost"

        return False, None

    # ==================================================
    # ONE ENVIRONMENT STEP
    # ==================================================
        # ==================================================
    # ONE ENVIRONMENT STEP
    # ==================================================

    def step(self, action):

        # Episode already finished
        if self.terminated:

            return (
                self.get_state(),
                0,
                True,
                {
                    "reason": "episode_already_finished"
                }
            )

        # -----------------------------
        # Increase step counter
        # -----------------------------

        self.steps += 1

        # -----------------------------
        # Move robot
        # -----------------------------

        moved, hit_obstacle, entered_hazard = self.move_robot(action)

        self.update_world_model()

        # -----------------------------
        # Battery decreases
        # -----------------------------

        self.battery -= 1

        if self.battery < 0:
            self.battery = 0

        # -----------------------------
        # Survivor health decreases
        # -----------------------------

        newly_lost = self.update_survivor_health()

        survivor_lost = len(newly_lost) > 0

        # -----------------------------
        # Rescue survivor
        # -----------------------------

        rescued = self.rescue_survivor()

        # -----------------------------
        # Hazard spreading
        # -----------------------------

        hazards_spread = []

        if self.steps % self.hazard_spread_interval == 0:

            hazards_spread = self.spread_hazards()

        # -----------------------------
        # Check episode termination
        # -----------------------------

        done, reason = self.check_episode_end()

        self.terminated = done

        # -----------------------------
        # Calculate reward
        # -----------------------------

        reward = calculate_reward(
            moved=moved,
            hit_obstacle=hit_obstacle,
            entered_hazard=entered_hazard,
            discovered_survivor=False,
            rescued_survivor=rescued,
            survivor_lost=survivor_lost,
            battery_empty=(self.battery == 0),
            mission_complete=(reason == "mission_complete")
        )

        # -----------------------------
        # Information about this step
        # -----------------------------

        info = {
            "moved": moved,
            "hit_obstacle": hit_obstacle,
            "entered_hazard": entered_hazard,
            "rescued": rescued,
            "survivors_lost": newly_lost,
            "hazards_spread": hazards_spread,
            "battery": self.battery,
            "reason": reason
        }

        # -----------------------------
        # Get next state
        # -----------------------------

        next_state = self.get_state()

        # -----------------------------
        # Return
        # -----------------------------

        return (
            next_state,
            reward,
            done,
            info
        )
        # ==================================================
    # GET CURRENT STATE
    # ==================================================

    def get_state(self):

        return {

            # Robot
            "robot": self.robot,

            # Actual environment information
            "survivors": self.survivors.copy(),

            "survivor_health":
                self.survivor_health.copy(),

            "rescued_survivors":
                self.rescued_survivors.copy(),

            "lost_survivors":
                self.lost_survivors.copy(),

            "hazards":
                self.hazards.copy(),

            "obstacles":
                self.obstacles.copy(),

            # Battery
            "battery":
                self.battery,

            # Episode step
            "steps":
                self.steps,

            # -----------------------------
            # Robot's local 3x3 observation
            # -----------------------------

            "local_observation":
                self.get_local_observation(),

            # -----------------------------
            # Robot's world model / memory
            # -----------------------------

            "known_cells":
                self.known_cells.copy(),

            "known_survivors":
                self.known_survivors.copy(),

            "known_hazards":
                self.known_hazards.copy(),

            "known_obstacles":
                self.known_obstacles.copy()
        }