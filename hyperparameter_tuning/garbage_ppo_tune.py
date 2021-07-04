
from config import config
import pandas as pd
from trade.time_series_validation import TimeSeriesValidation

from torch import nn as nn
import gym
import numpy as np
import os
import optuna
import joblib
import pickle
import json

import time

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# Stable baselines 3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback, EveryNTimesteps, \
	EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnMaxEpisodes

# ======================================================================== Enviorment settings
n_trials = 2
total_timesteps_model= 1e4#1e4
model_name = "ppo"
timestamp = str(int(time.time()))
run_path = "/"+timestamp+"_"+model_name
logs_base_dir = f"{config.LOG_DIR_HYPERPARAMETER_TUNING}{run_path}"
log_tensorboard = f"{logs_base_dir}/log_tensorboard"
study_name = f"{timestamp}_{model_name}"

###########df load the preprocessed dataset
df = pd.read_pickle(config.DATA_SAVE_DIR+"/preprocess_df.pkl")
###########env_params
stock_dimension = len(config.MULTIPLE_TICKER_8)
state_space = stock_dimension
features = config.TECHNICAL_INDICATORS_LIST + ["open", "close", 'high', 'low']
features += [f"{feature}_diff" for feature in features]
features += [feature for feature in df.columns if feature.startswith("cov_")]

env_params = {
	"initial_amount": 1000000, 
	"technical_indicator_list": features, 
	"max_assets_amount_per_trade": 100, 
	"main_tickers": df.tic.unique(),
	"all_tickers": df.tic.unique(),
	"reward_scaling": 1e-4,
	"comission_value": 0.01
}
###########model_name
model_name = "ppo"

# ======================================================================== Optuna Loop
def objective(trial):
	print(f"!!!!!!!!!!!!!!!!!!!!!Trial number: {trial.number}!!!!!!!!!!!!!!!!!!!!!")
	# Parallel environments
	# env = make_vec_env(gym.make(env_id), n_envs=4)
	os.makedirs(logs_base_dir, exist_ok=True)

	batch_size = trial.suggest_categorical(
		"batch_size", [8, 16, 32, 64, 128, 256, 512])
	n_steps = trial.suggest_categorical(
		"n_steps", [256, 512, 1024, 2048, 4096])
	gamma = trial.suggest_categorical(
		"gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
	learning_rate = trial.suggest_loguniform("lr", 2e-4, 6e-4)
	lr_schedule = "constant"

	ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
	clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
	n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
	gae_lambda = trial.suggest_categorical(
		"gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
	max_grad_norm = trial.suggest_categorical(
		"max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
	vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
	net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
	log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
	sde_sample_freq = trial.suggest_categorical(
		"sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
	ortho_init = False
	ortho_init = trial.suggest_categorical('ortho_init', [False, True])
	activation_fn = trial.suggest_categorical(
		'activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])

	net_arch = {
		"small": [dict(pi=[64, 64], vf=[64, 64])],
		"medium": [dict(pi=[128, 128], vf=[128, 128])],
		"large": [dict(pi=[256, 256], vf=[256, 256])],
	}[net_arch]

	activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU,
					 "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]


	#Create the policy_kwargs
	#Create the model_params where its included policy_kwargs 

	#Store the policy_kwargs into log_tensorboard
	#Store the model_kwargs into log_tensorboard


	tb_log_name = f"trial_{trial.number}_{model_name}"
	PPO_PARAMS = {
	"n_steps": n_steps,
	"ent_coef": ent_coef,
	"learning_rate": learning_rate,
	"batch_size": batch_size,
	}


	with open(f"{logs_base_dir}/{tb_log_name}.pkl", 'wb') as fp:
		pickle.dump(PPO_PARAMS, fp, protocol=pickle.HIGHEST_PROTOCOL)
  
	tsv = TimeSeriesValidation(num_splits=2, with_graphs=False, total_timesteps_model=total_timesteps_model)
	summary = tsv.run(df, env_params, model_name, PPO_PARAMS, log_tensorboard, tb_log_name)
	
	# model = PPO(
	#     MlpPolicy,
	#     env,
	#     n_steps=n_steps,
	#     batch_size=batch_size,
	#     gamma=gamma,
	#     learning_rate=learning_rate,
	#     ent_coef=ent_coef,
	#     clip_range=clip_range,
	#     n_epochs=n_epochs,
	#     gae_lambda=gae_lambda,
	#     max_grad_norm=max_grad_norm,
	#     vf_coef=vf_coef,
	#     sde_sample_freq=sde_sample_freq,
	#     policy_kwargs=dict(
	#         log_std_init=log_std_init,
	#         net_arch=net_arch,
	#         activation_fn=activation_fn,
	#         ortho_init=ortho_init,

	#     ),
	#     tensorboard_log=log_tensorboard,
	#     verbose=0
	# )



	return summary['sharpe']


# storage = optuna.storages.RedisStorage(
#     url='redis://34.123.159.224:6379/DB1',
# )
#storage = 'mysql://root:@34.122.181.208/rl'

# storage = 'mysql://joan:@127.0.0.1/rl'

# study = optuna.create_study(study_name=study_name, storage=storage,
#                             pruner=optuna.pruners.MedianPruner(), load_if_exists=True)


study = optuna.create_study(study_name=study_name, load_if_exists=True, direction="maximize")

study.optimize(objective, n_trials=n_trials, n_jobs=1)


joblib.dump(study, f"{logs_base_dir}/study.pkl")


# with open(f"{logs_base_dir}/best_params_{str(int(best_reward))}.txt", 'w') as file:
#      file.write(json.dumps(study.best_params))

# df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
# print(df) , direction='maximize'

# print(study.best_value)  # Get best objective value.
# print(study.best_trial)  # Get best trial's information.
# print(study.trials)  # Get all trials' information.
# len(study.trials) # Get number of trails.

