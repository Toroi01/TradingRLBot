import optuna
from config import config
import time
import joblib
import pandas as pd
import os

class Tune():
	def __init__(self, model_name, n_trials, total_timesteps_model):
		self.model_name = model_name
		self.n_trials = n_trials
		self.total_timesteps_model = total_timesteps_model

		timestamp = str(int(time.time()))
		run_path = "/"+timestamp+"_"+model_name
		self.logs_base_dir = f"{config.LOG_DIR_HYPERPARAMETER_TUNING}{run_path}"
		self.log_tensorboard = f"{self.logs_base_dir}/log_tensorboard"
		self.study_name = f"{timestamp}_{model_name}"

		os.makedirs(self.logs_base_dir, exist_ok=True)

		#env_params
		self.df = pd.read_pickle(config.DATA_SAVE_DIR+"/preprocess_df.pkl")
		features = config.TECHNICAL_INDICATORS_LIST + ["open", "close", 'high', 'low']
		features += [f"{feature}_diff" for feature in features]
		features += [feature for feature in self.df.columns if feature.startswith("cov_")]

		self.env_params = {
			"initial_amount": 1000000, 
			"technical_indicator_list": features, 
			"max_assets_amount_per_trade": 100, 
			"main_tickers": self.df.tic.unique(),
			"all_tickers": self.df.tic.unique(),
			"reward_scaling": 1e-4,
			"comission_value": 0.01
		}

	def run_study(self):
		study = optuna.create_study(study_name=self.study_name, load_if_exists=True, direction="maximize")
		study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1)
		joblib.dump(study, f"{self.logs_base_dir}/study.pkl")
	
