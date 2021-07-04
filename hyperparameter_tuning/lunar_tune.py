from torch import nn as nn
import gym
import numpy as np
import base64
import IPython
import PIL.Image
import pyvirtualdisplay
import os
import optuna
import joblib
import pickle
import json

from datetime import datetime

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

env_id = "LunarLander-v2"
# env_id = 'CartPole-v1'

n_trials = 3
timesteps = 3e5
reward_threshold = 200
best_reward = -1000000
# episodes_threshold = 1000
study_name = "superlunarv0"
eval_env = gym.make(env_id)
video_folder = './videos'
video_length = 3000

datetime = dt_string = datetime.now().strftime("%H_%M_%S")
logs_base_dir = f"./log/{dt_string}"
log_tensorboard = f"{logs_base_dir}/log_tensorboard"


# ======================================================================== Optuna Loop
def objective(trial):
    # gym environment & variables

    env = gym.make(env_id)
    # Parallel environments
    # env = make_vec_env(gym.make(env_id), n_envs=4)
    os.makedirs(logs_base_dir, exist_ok=True)
    env = Monitor(env, logs_base_dir)

    global episodes
    global mean_reward
    episodes = 0
    mean_reward = 0

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
    #Create the callback

    #Store the policy_kwargs into log_tensorboard
    #Store the model_kwargs into log_tensorboard
    
    model = PPO(
        MlpPolicy,
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        clip_range=clip_range,
        n_epochs=n_epochs,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        vf_coef=vf_coef,
        sde_sample_freq=sde_sample_freq,
        policy_kwargs=dict(
            log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,

        ),
        tensorboard_log=log_tensorboard,
        verbose=0
    )

    # ======================================================================== Hyper Parameters

    # ======================================================================== Evaluation

    class RewardCallback(BaseCallback):

        """
        Callback for saving a model (the check is done every ``check_freq`` steps)
        based on the training reward (in practice, we recommend using ``EvalCallback``).

        :param check_freq: (int)
        :param log_dir: (str) Path to the folder where the model will be saved.
        It must contains the file created by the ``Monitor`` wrapper.
        :param verbose: (int)
        """

        def __init__(self, check_freq: int, log_dir: str, verbose=1):
            super(RewardCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.log_dir = log_dir
            self.save_path = os.path.join(log_dir, 'best_model')
            self.best_mean_reward = -np.inf

        def _init_callback(self) -> None:
            # Create folder if needed
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)

        def _on_step(self) -> bool:
            if self.n_calls % self.check_freq == 0:

                # Retrieve training reward
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                if len(x) > 0:
                    global episodes
                    global mean_reward
                    global best_reward
                    episodes = len(y)
                    # print(episodes)
                    mean_reward = np.mean(y[-50:])
                    mean_reward = round(mean_reward, 0)
                    if self.verbose > 0:
                        print(f"Episodes: {episodes}")
                        print(f"Num steps: {self.num_timesteps}")
                        print(f"Mean reward: {mean_reward:.2f} ")
                        print("=================================")
                    # Report intermediate objective value to Optima and Handle pruning
                    trial.report(mean_reward,self.num_timesteps)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                    # New best model, you could save the agent here
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        if mean_reward > reward_threshold:
                            print("REWARD ACHIVED")
                            model.save(f"{self.save_path}/reward_achived_{str(mean_reward)}")
                            return False
                        else:
                            model.save(f"{self.save_path}/best_model")
                        

                    # New best model, you could save the agent here
                    # if episodes > episodes_threshold:
                    #     print("REWARD ACHIVED")
                    #     model.save(self.save_path)
                    #     return False



            return True

    # ======================================================================== Training

    check_freq = int(timesteps/10) if int(timesteps/10)>0 else 1
    callback = RewardCallback(check_freq=check_freq, log_dir=logs_base_dir)
    model.learn(total_timesteps=int(timesteps), callback=callback)


    # ==== Rest environment
    del model
    env.reset()

    return mean_reward


# storage = optuna.storages.RedisStorage(
#     url='redis://34.123.159.224:6379/DB1',
# )
#storage = 'mysql://root:@34.122.181.208/rl'

# storage = 'mysql://joan:@127.0.0.1/rl'

# study = optuna.create_study(study_name=study_name, storage=storage,
#                             pruner=optuna.pruners.MedianPruner(), load_if_exists=True)


study = optuna.create_study(study_name=study_name,
                            pruner=optuna.pruners.MedianPruner(), load_if_exists=True, direction="maximize")

study.optimize(objective, n_trials=n_trials, n_jobs=1)
joblib.dump(study, f"{logs_base_dir}/study.pkl")
with open(f"{logs_base_dir}/best_params_{str(int(best_reward))}.txt", 'w') as file:
     file.write(json.dumps(study.best_params))
# df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
# print(df) , direction='maximize'
print(study.best_params)
# print(study.best_value)  # Get best objective value.
# print(study.best_trial)  # Get best trial's information.
# print(study.trials)  # Get all trials' information.
# len(study.trials) # Get number of trails.

