from src.hyperparameter_tuning.tune import Tune
from src.evaluate.time_series_validation import TimeSeriesValidation


class PPOTune(Tune):
    def __init__(self, n_trials, env_params, tsv_params, start_date, end_date):
        super().__init__("ppo", n_trials, env_params, tsv_params, start_date, end_date)

    def sample_ppo_params(self, trial):
        n_steps = trial.suggest_categorical("n_steps", [2000, 3000, 5000, 10000])
        ent_coef = trial.suggest_loguniform("ent_coef", 0.001, 0.1)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 1e-2)

        batch_size = trial.suggest_categorical("batch_size", [5, 10, 20, 50]) 
        n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 15, 20])
        #clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
        #target_kl = trial.suggest_float("target_kl", 0.003, 0.03)
        gamma = trial.suggest_categorical("gamma", [0.999, 0.9999, 0.99999, 0.999999])
        # gae_lambda = trial.suggest_categorical("gae_lambda", [0.99, 0.999, 0.999, 0.9999])

        # ortho_init = trial.suggest_categorical("ortho_init", [False, True])
        # net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
        # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
        # net_arch = [
        #     {"pi": [256,128], "vf": [256,128]} if net_arch == "tiny" else {"pi": [512, 64], "vf": [512, 64]}
        # ]
        # activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]       
        
        PPO_PARAMS = {
            "n_steps": n_steps,
            "ent_coef": ent_coef,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            #"clip_range": clip_range,
            #"target_kl": target_kl,
            "gamma": gamma,
            # "gae_lambda": gae_lambda,
            # "policy_kwargs": {
            #     "net_arch": net_arch,
            #     "activation_fn": activation_fn,
            #     "ortho_init": ortho_init,
            # },
            "seed":8,
        }

        return PPO_PARAMS

    def objective(self, trial):
        PPO_PARAMS = self.sample_ppo_params(trial)
        print(PPO_PARAMS)
        tsv = TimeSeriesValidation(**self.tsv_params)
        metrics, model  = tsv.run(self.data, self.env_params, self.model_name, PPO_PARAMS, log_tensorboard=self.log_tensorboard)
        print(f"Metrics: {metrics}")
        self.save("hyperparameters", trial_number=trial.number, content=PPO_PARAMS)
        self.save("metrics", trial_number=trial.number, content=metrics)
        self.save("model", trial_number=trial.number, content=model)
        super().log_run("ppo", PPO_PARAMS, metrics, run_name = f"{self.timestamp}_trial_{trial.number}")

        return metrics['sharpe']
