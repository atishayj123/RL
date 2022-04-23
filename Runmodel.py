import gym
from stable_baselines3 import PPO
from learnig import SnekEnv

models_dir = "models/PPO_euclidean2"

env = SnekEnv()
env.reset()

model_path = f"{models_dir}/20000.zip"
model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        #env.render()
        print(rewards)