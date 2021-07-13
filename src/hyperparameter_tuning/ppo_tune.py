from src.hyperparameter_tuning.tune import Tune
from src.evaluate.time_series_validation import TimeSeriesValidation


class PPOTune(Tune):
    def __init__(self, n_trials, env_params, tsv_params, start_date, end_date):
        super().__init__("ppo", n_trials, env_params, tsv_params, start_date, end_date)

    def sample_ppo_params(self, trial):
        n_steps = trial.suggest_categorical("n_steps", [100, 500, 1000, 2000, 3000, 10000, 20000])
        ent_coef = trial.suggest_loguniform("ent_coef", 0.01, 0.1)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 1e-2)

        batch_size = trial.suggest_categorical("batch_size", [2, 5, 10, 20, 50, 100]) 
        n_epochs = trial.suggest_categorical("n_epochs", [3, 5, 10])
        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
        gamma = trial.suggest_categorical("gamma", [0.99, 0.995, 0.999, 0.9999])
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99999)
        
        PPO_PARAMS = {
            "n_steps": n_steps,
            "ent_coef": ent_coef,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "clip_range": clip_range,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
        }

        return PPO_PARAMS

    def objective(self, trial):
        PPO_PARAMS = self.sample_ppo_params(trial)
        tsv = TimeSeriesValidation(**self.tsv_params)
        metrics, model  = tsv.run(self.data, self.env_params, self.model_name, PPO_PARAMS, log_tensorboard=self.log_tensorboard)
        print(f"Metrics: {metrics}")
        self.save("hyperparameters", trial_number=trial.number, content=PPO_PARAMS)
        self.save("metrics", trial_number=trial.number, content=metrics)
        self.save("model", trial_number=trial.number, content=model)
        super().log_run("ppo", PPO_PARAMS, metrics, run_name = f"{self.timestamp}_trial_{trial.number}")

        return metrics['sharpe']
