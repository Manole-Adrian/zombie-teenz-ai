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
        self.player_count = 2

        # Flattened action space: 0-47 = move to city, 48 = treat city
        self.action_space = spaces.Discrete(self.num_cities + 1)

        self.observation_space = spaces.Dict({
            "infection_cubes": spaces.Box(low=0, high=self.max_infection, shape=(self.num_cities,), dtype=np.int32),
            "current_cities": spaces.Box(low=0, high=self.num_cities - 1, shape=(self.player_count,), dtype=np.int32)
        })

        self.state = np.zeros(self.num_cities, dtype=np.int32)
        self.step_count = 0
        self.max_steps = 200
        self.current_city = [0 for _ in range(self.player_count)]
        self.active_player = 0
        self.neighbors = adjacency_list

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.randint(1, high=2, size=self.num_cities)
        self.step_count = 0
        self.current_city = [0 for _ in range(self.player_count)]
        self.active_player = 0
        return {
            "infection_cubes": self.state.copy(),
            "current_cities": self.current_city.copy()
        }, {}

    def step(self, action):
        self.step_count += 1
        player = self.active_player
        current_city = self.current_city[player]

        # Decode action
        if action < self.num_cities:
            action_type = 0  # move
            target_city = action
        else:
            action_type = 1  # treat
            target_city = self.current_city[player]

        reward = 0
        terminated = False
        truncated = False
        infection_before = np.sum(self.state)

        if action_type == 0:  # Move
            if target_city in self.neighbors[current_city]:
                self.current_city[player] = target_city
                reward = -0.05
            else:
                reward = -0.2  # Illegal move
        elif action_type == 1:  # Treat
            if self.state[target_city] > 0:
                self.state[target_city] -= 1
                reward = 1.0
            else:
                reward = -0.5


        # Spread infection
        if self.step_count % 4 == 0:
            infection_city = self.np_random.integers(0, self.num_cities)
            if self.state[infection_city] < self.max_infection:
                self.state[infection_city] += 1

        # Reward shaping
        infection_after = np.sum(self.state)
        reward += (infection_before - infection_after) * 0.5

        won = np.all(self.state == 0)
        if won:
            reward += self.max_steps - self.step_count

        terminated = np.any(self.state >= self.max_infection) or won
        truncated = self.step_count >= self.max_steps

        self.active_player = (self.active_player + 1) % self.player_count

        return {
            "infection_cubes": self.state.copy(),
            "current_cities": self.current_city.copy()
        }, reward, terminated or truncated, truncated, {}

    def render(self):
        print(f"Step {self.step_count}")
        for i, level in enumerate(self.state):
            bar = "🦠" * level
            markers = "".join([
                "😀" if self.current_city[p] == i else "" for p in range(self.player_count)
            ])
            print(f"City {i}: Infection {level} {bar} {markers}")
        print("-" * 30)
