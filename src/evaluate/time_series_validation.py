
from src.model.runner import train_model, test_model
from src.preprocessing.data import format_for_env

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
        df = format_for_env(df)
        for n in range(self.num_splits):
            tb_name_train = f"split_{n}_train"
            train, test = self.next_part(df, n)
            model = train_model(train, env_params, model_name, model_params, self.total_timesteps_model, log_tensorboard, tb_name=tb_name_train)
            print("Metrics testing")
            results = test_model(test, env_params, model, with_graphs=self.with_graphs)
            total_results.append(results)

        summary = {}
        for metric in results.keys():
            summary[metric] = sum(d[metric] for d in total_results) / len(total_results)
        return summary

