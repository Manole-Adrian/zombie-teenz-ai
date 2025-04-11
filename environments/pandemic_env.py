import numpy as np
import gymnasium as gym
from gymnasium import spaces
from environments.cities import adjacency_list


class PandemicEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.num_cities = len(adjacency_list)
        self.max_infection = 4
        self.player_count = 3

        # Fixed action space includes all cities
        self.action_space = spaces.Discrete(self.num_cities)
        self.state = np.zeros(self.num_cities, dtype=np.int32)
        self.step_count = 0
        self.infection_step_count = 0
        self.max_steps = 200
        self.current_city = np.zeros(self.player_count, dtype=np.int32)  # Start in city 0
        self.active_player = 0
        self.neighbors = adjacency_list

        self.observation_space = spaces.Dict({
            "infection_cubes": spaces.Box(low=0, high=self.max_infection, shape=(self.num_cities,), dtype=np.int32),
            "current_cities": spaces.Box(low=0, high=self.max_infection, shape=(self.player_count,), dtype=np.int32)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.randint(0, high=2, size=self.num_cities)  # Example initial state
        self.step_count = 0
        self.infection_step_count = 0
        self.current_city = np.zeros(self.player_count, dtype=np.int32)  # Reset to starting city
        # print(f"state ${self.state} current_city ${self.current_city}")
        return {"infection_cubes": self.state.copy(), "current_cities": self.current_city.copy()}, {}

    def step(self, action):
        self.step_count += 1
        # Validate the action
        if action < 0 or action >= self.num_cities:
            raise ValueError(f"Invalid action: {action}. Must be between 0 and {self.num_cities - 1}")

        # Check if action is valid based on the current city
        if action not in self.neighbors[self.current_city[self.active_player]]:
            return {"infection_cubes": self.state.copy(),
                    "current_cities": self.current_city.copy()}, -0.1, False, False, {}  # Invalid action penalty

        self.infection_step_count += 1
        # Treat selected city
        chosen_city = action
        if self.state[chosen_city] > 0:
            self.state[chosen_city] -= 1
            if self.state[chosen_city] == 0:
                reward = 5.0
            else:
                reward = 1.0
        else:
            reward = -0.1  # Small penalty for wasting a turn

        # Infection spreads to a random city every 4 steps
        if self.infection_step_count % 4 == 0:
            self.active_player = (self.active_player + 1) % self.player_count

        if self.infection_step_count % 16 == 0:
            infection_city = self.np_random.integers(0, self.num_cities)
            if self.state[infection_city] < self.max_infection:
                self.state[infection_city] += 1
            if self.state[infection_city] == self.max_infection - 1:
                reward -= 2

        # Update the current city after moving
        self.current_city[self.active_player] = chosen_city

        won = np.all(self.state == 0)
        if won:
            reward += self.max_steps - self.step_count

        lost = np.any(self.state >= self.max_infection)

        terminated = lost or won
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated

        # print(f"state ${self.state} current_city ${self.current_city}")
        return {"infection_cubes": self.state.copy(),
                "current_cities": self.current_city.copy()}, reward, done, truncated, {}

    def render(self):
        print(f"Step {self.step_count}")
        for i, level in enumerate(self.state):
            bar = "🦠" * level
            marker1 = "😀" if self.current_city[0] == i else ""
            marker2 = "😍" if self.current_city[1] == i else ""
            marker3 = "😂" if self.current_city[1] == i else ""
            print(f"City {i}: Infection {level} {bar} {marker1} {marker2} {marker3}")
        print("-" * 30)
