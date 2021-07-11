from src.hyperparameter_tuning.tune import Tune
from src.evaluate.time_series_validation import TimeSeriesValidation


class DQNTune(Tune):
    def __init__(self, n_trials, env_params, tsv_params, start_date, end_date):
        super().__init__("dqn", n_trials, env_params, tsv_params, start_date, end_date)

    def objective(self, trial):
        gamma = trial.suggest_loguniform("gamma", 0.9, 0.9999)
        tau = trial.suggest_loguniform("tau", 0.001, 0.1)
        learning_rate = trial.suggest_loguniform("learning_rate", 0.0005, 0.01)
        batch_size = trial.suggest_discrete_uniform("batch_size", 32, 128, 1)
        #learning_starts = trial.suggest_discrete_uniform("learning_starts", 1000, 10000)

        DQN_PARAMS = {
            "gamma": gamma,
            "tau": tau,
            #"learning_starts": learning_starts,
            "learning_rate": learning_rate,
            "batch_size": int(batch_size)
        }

        tsv = TimeSeriesValidation(**self.tsv_params)
        metrics = tsv.run(self.data, self.env_params, self.model_name, DQN_PARAMS)
        print(f"Metrics: {metrics}")
        self.save("hyperparameters", trial_number=trial.number, content=DQN_PARAMS)
        self.save("metrics", trial_number=trial.number, content=metrics)
        super().log_run("ppo", DQN_PARAMS, metrics)

        return metrics['sharpe']
