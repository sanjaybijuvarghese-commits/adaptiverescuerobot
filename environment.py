import random


ROWS = 10
COLS = 10


class Environment:

    def __init__(self):

        self.rows = ROWS
        self.cols = COLS

        # -----------------------------
        # Environment settings
        # -----------------------------

        self.max_battery = 100

        self.num_survivors = 3
        self.num_hazards = 4
        self.num_obstacles = 6

        # How often hazards spread
        self.hazard_spread_interval = 5

        # -----------------------------
        # Start first episode
        # -----------------------------

        self.reset()

    # ==================================================
    # RESET - START A NEW EPISODE
    # ==================================================

    def reset(self):

        # Episode counter/time
        self.steps = 0

        # Robot
        self.robot = None

        # Environment objects
        self.survivors = []
        self.hazards = []
        self.obstacles = []

        # Survivor information
        self.survivor_health = {}

        # Survivor status
        self.rescued_survivors = set()
        self.lost_survivors = set()

        # Battery
        self.battery = self.max_battery

        # Episode status
        self.terminated = False

        # Generate new disaster
        self.generate_environment()

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
    # GENERATE NEW DISASTER ENVIRONMENT
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

            # Random health between 50 and 100
            self.survivor_health[position] = random.randint(50, 100)

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

            row = random.randint(0, self.rows - 1)
            col = random.randint(0, self.cols - 2)

            obstacle1 = (row, col)
            obstacle2 = (row, col + 1)

            if self.is_free(obstacle1) and self.is_free(obstacle2):

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
                "Invalid action. Use 0, 1, 2, or 3."
            )

        new_position = (new_row, new_col)

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
        # Move
        # -----------------------------

        self.robot = new_position

        # Check whether robot entered hazard
        entered_hazard = self.robot in self.hazards

        return True, False, entered_hazard

    # ==================================================
    # SURVIVOR HEALTH DECAY
    # ==================================================

    def update_survivor_health(self):

        newly_lost = []

        for survivor in self.survivors:

            # Ignore rescued or already lost survivors
            if survivor in self.rescued_survivors:
                continue

            if survivor in self.lost_survivors:
                continue

            # Health decreases
            self.survivor_health[survivor] -= 1

            # Check death
            if self.survivor_health[survivor] <= 0:

                self.survivor_health[survivor] = 0

                self.lost_survivors.add(survivor)

                newly_lost.append(survivor)

        return newly_lost

    # ==================================================
    # HAZARD SPREADING
    # ==================================================

    

    # ==================================================
    # RESCUE SURVIVOR
    # ==================================================

    def rescue_survivor(self):

        if self.robot not in self.survivors:

            return False

        survivor = self.robot

        # Already rescued?
        if survivor in self.rescued_survivors:

            return False

        # Already lost?
        if survivor in self.lost_survivors:

            return False

        # Rescue survivor
        self.rescued_survivors.add(survivor)

        return True

    # ==================================================
    # CHECK EPISODE TERMINATION
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
    # ONE RL STEP
    # ==================================================

    def step(self, action):

        # If episode already ended
        if self.terminated:

            return (
                self.get_state(),
                0,
                True,
                "episode_already_finished"
            )

        # -----------------------------
        # Step counter
        # -----------------------------

        self.steps += 1

        # -----------------------------
        # Robot movement
        # -----------------------------

        moved, hit_obstacle, entered_hazard = self.move_robot(action)

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
        # Basic reward
        # -----------------------------

        reward = -1

        if hit_obstacle:

            reward -= 5

        if entered_hazard:

            reward -= 20

        if rescued:

            reward += 50

        if survivor_lost:

            reward -= 50

        if self.battery == 0:

            reward -= 30

        if reason == "mission_complete":

            reward += 100

        # -----------------------------
        # Return information
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

        return (
            self.get_state(),
            reward,
            done,
            info
        )

    # ==================================================
    # GET CURRENT STATE
    # ==================================================

    def get_state(self):

        state = {

            "robot": self.robot,

            "survivors": self.survivors.copy(),

            "survivor_health": self.survivor_health.copy(),

            "rescued_survivors": self.rescued_survivors.copy(),

            "lost_survivors": self.lost_survivors.copy(),

            "hazards": self.hazards.copy(),

            "obstacles": self.obstacles.copy(),

            "battery": self.battery,

            "steps": self.steps
        }

        return state