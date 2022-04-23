from stable_baselines3 import PPO,A2C,DQN
import os
from learnig import SnekEnv
import time



models_dir = f"models/A2C_euclidean3/"
logdir = f"logs/A2C_euclidean3/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = SnekEnv()
env.reset()

model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"A2C")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")