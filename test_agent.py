from stable_baselines3 import PPO
from environments.zteenz_env import ZteenzEnv

def run_trained_agent():
    env = ZteenzEnv()
    model = PPO.load("models/zteenz_ppo")

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
