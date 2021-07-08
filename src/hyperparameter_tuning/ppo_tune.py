from src.hyperparameter_tuning.tune import Tune
from src.evaluate.time_series_validation import TimeSeriesValidation


class PPOTune(Tune):
    def __init__(self, n_trials, env_params, tsv_params, start_date, end_date):
        super().__init__("ppo", n_trials, env_params, tsv_params, start_date, end_date)

    def objective(self, trial):
        n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096, 8192])# (1/3) total_timesteps_model aprox
        ent_coef = trial.suggest_loguniform("ent_coef", 0.01, 0.1)
        learning_rate = trial.suggest_loguniform("lr", 5e-4, 1e-2)

        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 256]) 
        n_epochs = trial.suggest_categorical("n_epochs", [3, 5, 10])
        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
        #target_kl = trial.suggest_float("target_kl", 0.003, 0.03)
        gamma = trial.suggest_categorical("gamma", [0.99, 0.995, 0.999, 0.9999])
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99999)
        vf_coef = trial.suggest_uniform("vf_coef", 0.5, 1)

        PPO_PARAMS = {
            "n_steps": n_steps,
            "ent_coef": ent_coef,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "clip_range": clip_range,
            #"target_kl": target_kl,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "vf_coef": vf_coef,
        }

        tsv = TimeSeriesValidation(**self.tsv_params)
        metrics = tsv.run(self.data, self.env_params, self.model_name, PPO_PARAMS)
        print(f"Metrics: {metrics}")
        # Save hyperparameters with evaluation metrics
        self.save_hyperparameters_metrics(trial_number=trial.number, hyperparameters=PPO_PARAMS, metrics=metrics)

        return metrics['sharpe']
