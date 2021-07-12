
from numpy.core.fromnumeric import argmax
from src.preprocessing.data import format_for_env

from src.environment.env_custom import CustomTradingEnv
from src.model.models import DRLAgent
from src.evaluate.backtest import BackTest
from src.preprocessing.data import build_features
from src.model.runner import test_model

"""
Class to perform time series validation. 
Splits dataset in num_splits, training and testing sequentially. Metrics are computed for each subset and then
averaged.
"""


class TimeSeriesValidation:
    def __init__(self, num_splits=5, test_proportion=0.2, gap_proportion=0.05, total_timesteps_model=1000,
                 with_graphs=True):
        self.num_splits = num_splits
        self.test_proportion = test_proportion
        self.gap_proportion = gap_proportion
        self.total_timesteps_model = total_timesteps_model
        self.with_graphs = with_graphs

    def next_part(self, df, n):
        split_length = max(df.index) // self.num_splits
        test_length = split_length * self.test_proportion
        gap_length = split_length * self.gap_proportion
        train_length = split_length - test_length - gap_length

        init_train = 0
        end_train = int(split_length * n + train_length)

        init_test = int(split_length * n + train_length + gap_length)
        end_test = int(split_length * n + train_length + gap_length + test_length)

        train = df.loc[init_train:end_train, :]
        test = df.loc[init_test:end_test, :]

        return train, test

    def run(self, df, env_params, model_name, model_params, log_tensorboard=None):
        total_results = []
        agent = DRLAgent(CustomTradingEnv(df=df, **env_params))
        model = agent.get_model(model_name=model_name, model_kwargs=model_params, tensorboard_log=log_tensorboard)        
        df = format_for_env(df)
        
        for n in range(self.num_splits):
            tb_name_train = f"split_{n}_train"
            df_train, df_test = self.next_part(df, n)            
            #Train
            print(f"Train from [{df_train['date'].iloc[0]}] to [{df_train['date'].iloc[-1]}]")
            env_train = CustomTradingEnv(df=df_train, **env_params)
            agent.env = env_train
            model = agent.train_model(model=model, tb_log_name=tb_name_train, total_timesteps=self.total_timesteps_model)
            #Test
            results = test_model(df_test, env_params, model, with_graphs=self.with_graphs)
            total_results.append(results)

        summary = {}
        for metric in results.keys():
            summary[metric] = sum(d[metric] for d in total_results) / len(total_results)

        return summary, model

