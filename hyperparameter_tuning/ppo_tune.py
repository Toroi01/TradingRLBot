from hyperparameter_tuning.tune import Tune

class PPO_tune(Tune):
	def __init__(self, n_trials, total_timesteps_model):
		super().__init__("ppo", n_trials, total_timesteps_model)

	def objective(self, trial):
		n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048, 4096])
		ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
		learning_rate = trial.suggest_loguniform("lr", 2e-4, 6e-4)

		PPO_PARAMS = {
			"n_steps": n_steps,
			"ent_coef": ent_coef,
			"learning_rate": learning_rate,
		}

		#Get the model
		model = self.get_model(hyperparameters=PPO_PARAMS)

		#Train the model
		model_trained = self.train_model(model=model, trial_number=trial.number)

		#Evaluate the model
		metrics = self.test_model(model_trained)

		#Save hyperparameters with evaluation metrics
		self.save_hyperparameters_metrics(trial_number=trial.number, hyperparameters=PPO_PARAMS, metrics=metrics)

		return metrics['sharpe']
