from hyperparameter_tuning.tune import Tune
from trade.time_series_validation import TimeSeriesValidation
import pickle

class DDPG_tune(Tune):
	def objective(self, trial):
		n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048, 4096])
		ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
		learning_rate = trial.suggest_loguniform("lr", 2e-4, 6e-4)

		DDPG_PARAMS = {
			"learning_rate": learning_rate,
		}

		tb_log_name = f"trial_{trial.number}_{self.model_name}"
		tsv = TimeSeriesValidation(num_splits=2, with_graphs=False, total_timesteps_model=self.total_timesteps_model)
		summary = tsv.run(self.df, self.env_params, self.model_name, DDPG_PARAMS, self.log_tensorboard, tb_log_name)

		with open(f"{self.logs_base_dir}/{tb_log_name}_{str(int(summary['sharpe']))}.pkl", 'wb') as fp:
			pickle.dump(DDPG_PARAMS, fp, protocol=pickle.HIGHEST_PROTOCOL)

		return summary['sharpe']


ddpg_tune = DDPG_tune(model_name="ddpg", n_trials=1, total_timesteps_model=1e4)

ddpg_tune.run_study()