from env.env_custom import CustomTradingEnv
from model.models import DRLAgent
from preprocessing.data import format_for_env
from trade.backtest import BackTest


"""
Class to perform time series validation. 
Splits dataset in num_splits, training and testing sequentially. Metrics are computed for each subset and then
averaged.
"""


class TimeSeriesValidation:
    def __init__(self, num_splits=5, test_proportion=0.2, gap_proportion=0.05, total_timesteps_model=10000,
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

        init_train = int(split_length * n)
        end_train = int(split_length * n + train_length)

        init_test = int(split_length * n + train_length + gap_length)
        end_test = int(split_length * n + train_length + gap_length + test_length)

        train = df.loc[init_train:end_train, :]
        test = df.loc[init_test:end_test, :]

        return train, test

    def run(self, df, env_params, model_name, model_params, log_tensorboard=None, tb_log_name="tb_log_name"):
        total_results = []
        df = format_for_env(df)
        for n in range(self.num_splits):
            train, test = self.next_part(df, n)
            model = self.train_model(train, env_params, model_name, model_params, log_tensorboard, tb_log_name)
            print("Metrics training")
            _ = self.test_model(train, env_params, model)
            print("Metrics testing")
            results = self.test_model(test, env_params, model)
            total_results.append(results)

        summary = {}
        for metric in results.keys():
            summary[metric] = sum(d[metric] for d in total_results) / len(total_results)
        return summary

    def train_model(self, train, env_params, model_name, model_params, log_tensorboard=None, tb_log_name="tb_log_name"):
        env_train_gym = CustomTradingEnv(df=train, **env_params)
        env_train, _ = env_train_gym.get_sb_env()
        print(f"Train from [{train['date'].iloc[0]}] to [{train['date'].iloc[-1]}]")

        agent = DRLAgent(env=env_train)
        model = agent.get_model(model_name=model_name, model_kwargs=model_params, tensorboard_log=log_tensorboard)
        return agent.train_model(model=model,
                                 tb_log_name=tb_log_name,
                                 total_timesteps=self.total_timesteps_model)

    def test_model(self, test, env_params, model):
        print(f"Test from [{test['date'].iloc[0]}] to [{test['date'].iloc[-1]}]")
        env_test_gym = CustomTradingEnv(df=test, **env_params)
        _, _, allocation_values = DRLAgent.DRL_prediction(model=model, environment=env_test_gym)
        bat = BackTest(model, env_test_gym)
        results = bat.evaluate(allocation_values)
        if self.with_graphs:
            bat.plot_return_against_hold(allocation_values)
        return results
