from environments.zteenz_env import ZteenzEnv
from IQN.IQNAgent import IQNAgent
import torch
import os

def train():
    env = ZteenzEnv()
    agent = IQNAgent(observation_space=env.observation_space["current_locations"].shape[0],
                     action_space=env.action_space.n, num_quantiles=32)

    total_timesteps = 50_000
    epsilon = 0.1  # Epsilon for epsilon-greedy policy
    print(f"Training started with {total_timesteps} total timesteps...")

    for timestep in range(total_timesteps):
        state = env.reset()[0]["current_locations"]
        done = False
        total_reward = 0  # Track total reward for the episode

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Store experience and train agent
            agent.store_experience(state, action, reward, next_state["current_locations"], terminated or truncated)
            loss = agent.train()

            # Accumulate total reward
            total_reward += reward

            state = next_state["current_locations"]
            if terminated or truncated:
                break

        # Every 1000 timesteps, update target model
        if timestep % 1000 == 0:
            agent.update_target_model()
            print(f"Timestep {timestep}/{total_timesteps} - Target model updated")

        # Print episode progress (only if loss is valid)
        if timestep % 100 == 0:
            if loss is not None:
                print(f"Timestep {timestep}/{total_timesteps} - Total Reward: {total_reward:.4f}, Loss: {loss:.4f}")
            else:
                print(f"Timestep {timestep}/{total_timesteps} - Total Reward: {total_reward:.4f}, Loss: None")

    # Save the model after training
    os.makedirs("models", exist_ok=True)
    torch.save(agent.model.state_dict(), "models/zteenz_iqn.pth")
    print("✅ Model saved to /models")

    return env, agent

def render_full_playthrough(env, agent):
    obs, _ = env.reset()
    done = False

    while not done:
        env.render()  # Render the current state
        action = agent.select_action(obs["current_locations"], epsilon=0)  # Greedy action
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.render()  # Render the final state

if __name__ == "__main__":
    env, agent = train()
    print("Training complete! Starting full playthrough...")
    render_full_playthrough(env, agent)
