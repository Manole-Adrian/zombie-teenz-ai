import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from environments.locations import adjacency_list
from environments.zombie_tracks import zombie_tracks

class ZteenzEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.num_locations = len(adjacency_list)
        self.zombie_count = 4
        self.player_count = 3
        # Flattened action space: 0-12 - Move, 13 Move Crate, 14 Attack here, 15 Skip
        self.action_space = spaces.Discrete(self.num_locations + 3)

        self.observation_space = spaces.Dict({
            "crate_locations": spaces.Box(low=0, high=self.num_locations - 1, shape=(4,), dtype=np.int32),
            "current_locations": spaces.Box(low=0, high=self.num_locations - 1, shape=(self.player_count,), dtype=np.int32),
            "active_player": spaces.Discrete(self.player_count),
            "zombies_locations": spaces.Box(low=0, high=self.num_locations -1, shape=(self.zombie_count,), dtype=np.int32),
            "zombies_progress": spaces.Box(low=0,high=6, shape=(self.zombie_count,), dtype=np.int32),
            "covered_locations": spaces.Box(low=0,high=1, shape=(4,), dtype=np.int32)
        })

        self.crate_locations = [0,0,0,0]
        self.buildings = [2,5,8,11]
        self.step_count = 0
        self.max_steps = 200
        self.current_city = np.zeros(self.player_count, dtype=np.int32)
        self.zombies_progress = np.zeros(self.zombie_count, dtype=np.int32)
        self.zombies_locations = np.zeros(self.zombie_count, dtype=np.int32)
        self.covered_locations = np.zeros(4, dtype=np.int32)
        self.active_player = 1
        self.neighbors = adjacency_list
        self.dice_throw = 1
        self.defeated_zombies_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.current_city = [0 for _ in range(self.player_count)]
        self.crate_locations = np.array([2,5,8,11], dtype=np.int32)  # <- NumPy array!
        self.zombies_progress = [0 for _ in range(self.zombie_count)]
        self.zombies_locations = [0 for _ in range(self.zombie_count)]
        self.covered_locations = [0 for _ in range(4)]
        self.defeated_zombies_count = 0

        initial_zombies = np.random.choice(np.arange(1, 4), size=2, replace=False)
        for zombie_index in initial_zombies:
            self.zombies_progress[zombie_index] = 1
            self.zombies_locations[zombie_index] = zombie_tracks[zombie_index][1]
        self.active_player = 1
        self.dice_throw = 1
        return {
            "crate_locations": self.crate_locations.copy(),
            "current_locations": self.current_city.copy(),
            "active_player": self.active_player,
            "zombies_locations": self.zombies_locations.copy(),
            "zombies_progress": self.zombies_progress.copy(),
            "covered_locations": self.covered_locations.copy()
        }, {}

    def step(self, action):
        player = self.active_player
        current_city = self.current_city[player]
        reward = -0.1

        if self.step_count % 2 == 0:
            # Throw dice
            self.dice_throw = self.np_random.integers(0, 4)

            # Do dice action
            if self.dice_throw <= 4:
                if self.zombies_locations[self.dice_throw] in self.buildings:
                    coverIndex = self.buildings.index(self.zombies_locations[self.dice_throw])
                    if self.covered_locations[coverIndex] == 1:
                        self.zombies_progress[self.dice_throw] += 1
                        self.zombies_locations[self.dice_throw] = zombie_tracks[self.dice_throw][self.zombies_progress[self.dice_throw]]
                        # reward -= 1
                    else:
                        self.covered_locations[coverIndex] = 1
                        # reward -= 2
                else:
                    self.zombies_progress[self.dice_throw] += 1
                    self.zombies_locations[self.dice_throw] = zombie_tracks[self.dice_throw][self.zombies_progress[self.dice_throw]]


        # Decode action
        if action < self.num_locations:
            action_type = 0  # move
            target_city = action
        elif action == 13:
            action_type = 1  # move crate
            target_city = None
            # print("Tried moving!")
            # Iterate through all players to find a neighbor
            for other_player_index in range(self.player_count):
                other_player_location = self.current_city[other_player_index]
                if other_player_location in self.neighbors[current_city]:
                    target_city = other_player_location
                    # print("Valid move!")
                    break  # Found a neighbor, no need to check further
            if target_city == None:
                target_city = current_city
                # print("Invalid!")
        elif action == 14:
            action_type = 2 # attack here
            target_city = current_city
        else:
            action_type = 3 # skip
            target_city = current_city


        terminated = False
        truncated = False

        if action_type == 0:  # Move
            if target_city in self.neighbors[current_city]:
                self.current_city[player] = target_city
                if target_city in self.crate_locations:
                    reward += 0.3
                if target_city in self.zombies_locations and target_city != 0:
                    reward += 0.05
                if target_city in self.current_city:
                    reward -= 0.05
                # crate_distances = [loc for loc in self.crate_locations if loc in self.neighbors[target_city]]
                # if crate_distances:
                #     reward += 0.01 * len(crate_distances)  # small bonus for being near crates

            else:
                reward -= 0.1  # Illegal move
        elif action_type == 1:  # Move Crate
            if current_city in self.crate_locations and target_city != current_city:
                if target_city == 0:
                    reward += 4
                    print("Moved to school")
                else:
                    current_dist = self._distance_to_school(current_city)
                    target_dist = self._distance_to_school(target_city)
                    if target_dist < current_dist:
                        reward += (current_dist - target_dist) * 0.5  # the closer the better
                    else:
                        reward -= 0.5
                crateIndex = np.where(self.crate_locations == current_city)[0][0]
                self.crate_locations[crateIndex] = target_city # move crate
            else:
                reward -= 0.1 # Illegal move
                # print(f"Couldnt move {current_city in self.crate_locations} or {target_city != current_city}")
        elif action_type == 2: # Attack
            if target_city in self.zombies_locations and target_city != 0:
                zombieIndex = self.zombies_locations.index(target_city)
                progress = self.zombies_progress[zombieIndex]
                self.zombies_progress[zombieIndex] = 0
                self.zombies_locations[zombieIndex] = zombie_tracks[zombieIndex][0]
                base_reward = 0.3
                urgency_reward = (progress) * 0.1
                diminishing_returns = 0.03 * self.defeated_zombies_count
                reward += base_reward + urgency_reward - diminishing_returns
                self.defeated_zombies_count += 1
            else:
                reward -= 0.2 # Illegal move
        elif action_type ==3:
            reward -= 0.2

        # if current_city in self.crate_locations:
        #     reward += 0.05

        won = np.all(np.array(self.crate_locations) == 0)
        if won:
            reward += self.max_steps - self.step_count

        self.step_count += 1

        if self.step_count % 2 == 0:
            self.active_player += 1
            self.active_player %= self.player_count
            # if self.active_player == 0:
            #     self.active_player = 1

        terminated = np.all(np.array(self.covered_locations) == 1) or won
        truncated = self.step_count >= self.max_steps
        return {
            "crate_locations": self.crate_locations.copy(),
            "current_locations": self.current_city.copy(),
            "active_player": self.active_player,
            "zombies_locations": self.zombies_locations.copy(),
            "zombies_progress": self.zombies_progress.copy(),
            "covered_locations": self.covered_locations.copy()
        }, reward, terminated or truncated, truncated, {}

    def _distance_to_school(self, location):
        # Rough distance heuristic: 0 if at school, 1 if adjacent, 2 if 2 steps away, etc.
        visited = set()
        frontier = [(location, 0)]
        while frontier:
            node, dist = frontier.pop(0)
            if node == 0:
                return dist
            visited.add(node)
            for neighbor in self.neighbors.get(node, []):
                if neighbor not in visited:
                    frontier.append((neighbor, dist + 1))
        return 999  # unreachable (shouldn't happen)

    def render(self):
        # Rendering the game board
        print("=== Zombie Teenz Evolution ===")

        # Show players' locations
        print("\nPlayer Locations:")
        for player_idx in range(self.player_count):
            print(f"Player {player_idx + 1} is at location {self.current_city[player_idx]}")

        # Show zombies' locations and progress
        print("\nZombie Locations and Progress:")
        for zombie_idx in range(self.zombie_count):
            print(
                f"Zombie {zombie_idx + 1} is at location {self.zombies_locations[zombie_idx]} with progress {self.zombies_progress[zombie_idx]}")

        # Show crate locations
        print("\nCrate Locations:")
        for i, crate_loc in enumerate(self.crate_locations):
            print(f"Crate {i + 1} is at location {crate_loc}")

        # Show covered locations
        print("\nCovered Locations:")
        for i, covered in enumerate(self.covered_locations):
            status = "Covered" if covered == 1 else "Not Covered"
            print(f"Location {self.buildings[i]}: {status}")

        # Show the active player
        print(f"\nActive Player: Player {self.active_player + 1}")

        # Show the step count
        print(f"\nStep Count: {self.step_count}/{self.max_steps}")

        # Check if the game is over
        if np.all(np.array(self.covered_locations) == 1):
            print("\nAll locations are covered! The game is lost!")
        elif np.all(np.array(self.crate_locations) == 0):
            print("\nAll crates are at the school! The game is won!")
        elif self.step_count >= self.max_steps:
            print("\nMaximum steps reached! Game Over!")

        print("-"*30)
