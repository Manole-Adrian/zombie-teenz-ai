from stable_baselines3 import PPO
from environments.pandemic_env import PandemicEnv
import os


def train():
    env = PandemicEnv()
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/", learning_rate=0.0006)

    total_timesteps = 500_000
    model.learn(total_timesteps=total_timesteps)

    # Save the model after training
    os.makedirs("models", exist_ok=True)
    model.save("models/pandemic_ppo")
    print("✅ Model saved to /models")

    return env, model


def render_full_playthrough(env, model):
    obs, _ = env.reset()
    done = False

    while not done:
        env.render()  # Render the current state
        action, _states = model.predict(obs)  # Get action from the model

        # Ensure action is an integer
        action = int(action)  # Convert the action to an integer if necessary

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.render()  # Render the final state



if __name__ == "__main__":
    env, model = train()
    print("Starting full playthrough...")
    render_full_playthrough(env, model)
