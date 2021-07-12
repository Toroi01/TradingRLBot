from src.evaluate.time_series_validation import TimeSeriesValidation
from src.hyperparameter_tuning.tune import Tune


class DDPGTune(Tune):
    def __init__(self, n_trials, env_params, tsv_params, start_date, end_date):
        super().__init__("ddpg", n_trials, env_params, tsv_params, start_date, end_date)

    def objective(self, trial):
        gamma = trial.suggest_loguniform("gamma", 0.9, 0.99)
        tau = trial.suggest_uniform("tau", 0.001, 0.1)
        learning_rate = trial.suggest_loguniform("learning_rate", 0.001, 0.01)
        batch_size = trial.suggest_discrete_uniform("batch_size", 32, 128, 1)
        buffer_size= trial.suggest_discrete_uniform("buffer_size", 100000, 1500000, 100000)

        DDPG_PARAMS = {
            "gamma": gamma,
            "tau": tau,
            "learning_rate": learning_rate,
            "batch_size": int(batch_size),
            "buffer_size": int(buffer_size)
        }

        tsv = TimeSeriesValidation(**self.tsv_params)
        metrics, model = tsv.run(self.data, self.env_params, self.model_name, DDPG_PARAMS, log_tensorboard=self.log_tensorboard)
        print(f"Metrics: {metrics}")
        self.save("hyperparameters", trial_number=trial.number, content=DDPG_PARAMS)
        self.save("metrics", trial_number=trial.number, content=metrics)
        self.save("model", trial_number=trial.number, content=model)
        super().log_run("ddpg", DDPG_PARAMS, metrics, run_name = f"{self.timestamp}_trial_{trial.number}")

        return metrics['sharpe']
