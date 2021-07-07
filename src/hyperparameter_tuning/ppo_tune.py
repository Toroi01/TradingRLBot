from src.hyperparameter_tuning.tune import Tune
from src.evaluate.time_series_validation import TimeSeriesValidation


class PPOTune(Tune):
    def __init__(self, n_trials, env_params, tsv_params, start_date, end_date):
        super().__init__("ppo", n_trials, env_params, tsv_params, start_date, end_date)

    def objective(self, trial):
        n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048, 4096])
        ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
        learning_rate = trial.suggest_loguniform("lr", 2e-4, 6e-4)

        PPO_PARAMS = {
            "n_steps": n_steps,
            "ent_coef": ent_coef,
            "learning_rate": learning_rate,
        }

        tsv = TimeSeriesValidation(**self.tsv_params)
        metrics = tsv.run(self.data, self.env_params, self.model_name, PPO_PARAMS)
        print(f"Metrics: {metrics}")
        # Save hyperparameters with evaluation metrics
        self.save_hyperparameters_metrics(trial_number=trial.number, hyperparameters=PPO_PARAMS, metrics=metrics)

        return metrics['sharpe']
