from hyperparameter_tuning.tune import Tune
from trade.time_series_validation import TimeSeriesValidation
import pickle

class A2C_tune(Tune):
	def objective(self, trial):
		#A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
		n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048, 4096])
		ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
		learning_rate = trial.suggest_loguniform("lr", 2e-4, 6e-4)

		A2C_PARAMS = {
			"n_steps": n_steps,
			"ent_coef": ent_coef,
			"learning_rate": learning_rate,
		}

		tb_log_name = f"trial_{trial.number}_{self.model_name}"
		tsv = TimeSeriesValidation(num_splits=1, with_graphs=False, total_timesteps_model=self.total_timesteps_model)
		summary = tsv.run(self.df, self.env_params, self.model_name, A2C_PARAMS, self.log_tensorboard, tb_log_name)

		with open(f"{self.logs_base_dir}/{tb_log_name}_{str(int(summary['sharpe']))}.pkl", 'wb') as fp:
			pickle.dump(A2C_PARAMS, fp, protocol=pickle.HIGHEST_PROTOCOL)

		return summary['sharpe']


a2c_tune = A2C_tune(model_name="a2c", n_trials=1, total_timesteps_model=1e4)

a2c_tune.run_study()