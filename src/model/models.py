from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

from src.config import config

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO, "dqn": DQN}

MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


class DRLAgent:
    def __init__(self, env):
        self.env = env

    def get_model(
            self,
            model_name,
            policy="MlpPolicy",
            policy_kwargs=None,
            model_kwargs=None,
            verbose=1,
            tensorboard_log=None,

    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            **model_kwargs,
        )
        return model

    def train_model(self, model, tb_log_name, total_timesteps=5000, callback=[]):
        model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, callback=callback)
        return model

    @staticmethod
    def DRL_prediction(model, environment):
        test_env, test_obs = environment.get_sb_env()
        test_env.reset()
        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs)
            test_obs, rewards, dones, info = test_env.step(action)
            if i == (len(environment.df.index.unique()) - 4):
                allocations = test_env.env_method(method_name="save_asset_memory")
                transactions = test_env.env_method(method_name="save_action_memory")
                allocation_values = test_env.env_method(method_name="save_asset_values_memory")

        return allocations[0], transactions[0], allocation_values[0]
