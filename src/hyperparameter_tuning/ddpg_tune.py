from src.evaluate.time_series_validation import TimeSeriesValidation
from src.hyperparameter_tuning.tune import Tune


class DDPGTune(Tune):
    def __init__(self, n_trials, env_params, tsv_params, start_date, end_date):
        super().__init__("ddpg", n_trials, env_params, tsv_params, start_date, end_date)

    def objective(self, trial):
        gamma = trial.suggest_loguniform("gamma", 0.1, 0.99)
        tau = trial.suggest_uniform("tau", 0.01, 0.5)
        exploration_fraction = trial.suggest_loguniform("exploration_fraction", 0.001, 0.1)
        learning_rate = trial.suggest_loguniform("learning_rate", 0.0005, 0.01)
        batch_size = trial.suggest_discrete_uniform("batch_size", 32, 128, 1)
        learning_starts = trial.suggest_discrete_uniform("learning_starts", 10, 10000, 2)

        DDPG_PARAMS = {
            "gamma": gamma,
            "tau": tau,
            "exploration_fraction": exploration_fraction,
            "learning_starts": learning_starts,
            "learning_rate": learning_rate,
            "batch_size": int(batch_size)
        }

        tsv = TimeSeriesValidation(**self.tsv_params)
        metrics = tsv.run(self.data, self.env_params, self.model_name, DDPG_PARAMS)
        print(f"Metrics: {metrics}")

        # Save hyperparameters with evaluation metrics
        self.save_hyperparameters_metrics(trial_number=trial.number, hyperparameters=DDPG_PARAMS, metrics=metrics)

        return metrics['sharpe']