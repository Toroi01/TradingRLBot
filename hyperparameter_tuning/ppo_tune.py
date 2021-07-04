from hyperparameter_tuning.tune import Tune
from trade.time_series_validation import TimeSeriesValidation
import pickle

class PPO_tune(Tune):
	def objective(self, trial):
		n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048, 4096])
		ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
		learning_rate = trial.suggest_loguniform("lr", 2e-4, 6e-4)

		PPO_PARAMS = {
			"n_steps": n_steps,
			"ent_coef": ent_coef,
			"learning_rate": learning_rate,
		}

		tb_log_name = f"trial_{trial.number}_{self.model_name}"
		tsv = TimeSeriesValidation(num_splits=2, with_graphs=False, total_timesteps_model=self.total_timesteps_model)
		summary = tsv.run(self.df, self.env_params, self.model_name, PPO_PARAMS, self.log_tensorboard, tb_log_name)

		with open(f"{self.logs_base_dir}/{tb_log_name}_{str(int(summary['sharpe']))}.pkl", 'wb') as fp:
			pickle.dump(PPO_PARAMS, fp, protocol=pickle.HIGHEST_PROTOCOL)

		return summary['sharpe']


ppo_tune = PPO_tune(model_name="ppo", n_trials=3, total_timesteps_model=1e4)

ppo_tune.run_study()