from environments.pandemic_env import PandemicEnv

env = PandemicEnv()
obs, info = env.reset()
done = False

while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Action: {action}, Reward: {reward}")
