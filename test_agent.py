from stable_baselines3 import PPO
from environments.pandemic_env import PandemicEnv

def run_trained_agent():
    env = PandemicEnv()
    model = PPO.load("models/pandemic_ppo")

    obs, info = env.reset()
    done = False

    while not done:
        env.render()
        action, _states = model.predict(obs)

        # Ensure the action is an integer
        action = int(action)  # Convert to integer if it's not already

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Action: {action}, Reward: {reward}")

if __name__ == "__main__":
    run_trained_agent()
