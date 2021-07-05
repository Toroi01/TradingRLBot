import optuna
from config import config
import time
import joblib
import pandas as pd
import os
from model.models import DRLAgent
from env.env_custom import CustomTradingEnv
from preprocessing.data import data_split
from trade.backtest import BackTest
import pickle
from dataset.download_dataset.cryptodownloader_binance import CryptoDownloader_binance
from preprocessing.preprocessors import FeatureEngineer

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

		#Preprocessed df
		self.df = self.get_df()

		#Train env
		self.env_train, _ = self.get_env(config.HT_START_TRAIN_DATE, config.HT_END_TRAIN_DATE).get_sb_env()

		#Test env
		self.env_test = self.get_env(config.HT_START_TEST_DATE, config.HT_END_TEST_DATE)

		#Agent
		self.agent = DRLAgent(env = self.env_train)
	
	def get_df(self):
		#Check if it already exists
		df_path = os.path.join(config.DATA_SAVE_DIR, config.PREPROCESSED_DF_NAME)
		exists = os.path.isfile(df_path)
		if exists:
			return pd.read_pickle(df_path)
		else:
			#Download it
			data_downloader = CryptoDownloader_binance(config.START_DATE, config.END_DATE, config.ACTUAL_TICKERS, config.DATA_SAVE_DIR, config.DATA_GRANULARITY)
			data_downloader.download_data()
			df = data_downloader.load()

			#Preprocess it
			fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_turbulence=False,
                    user_defined_feature = True)
			df = fe.preprocess_data(df)

			#Save it
			df.to_pickle(df_path)
			return df

	def get_env(self, start_date, end_date):
		features =  config.TECHNICAL_INDICATORS_SHORTPERIOD + config.TECHNICAL_INDICATORS_LONGPERIOD  + ["open", "close", 'high', 'low']
		features += [f"{feature}_diff" for feature in features]
		features += [feature for feature in self.df.columns if feature.startswith("cov_")]
		env_kwargs = {
			"initial_amount": 1000000, 
			"technical_indicator_list": features, 
			"max_assets_amount_per_trade": 100, 
			"main_tickers": config.ACTUAL_TICKERS,
			"all_tickers": config.ACTUAL_TICKERS,
			"reward_scaling": 1e-4,
			"comission_value": 0.01		
		}
		data = data_split(self.df, start_date, end_date)

		env = CustomTradingEnv(df = data, **env_kwargs)

		return env
		

	def get_model(self, hyperparameters):
		model = self.agent.get_model(model_name=self.model_name, model_kwargs=hyperparameters, tensorboard_log=self.log_tensorboard)
		return model

	def train_model(self, model, trial_number):
		tb_log_name = f"trial_{trial_number}_{self.model_name}"
		trained_model = self.agent.train_model(model=model, tb_log_name=tb_log_name, total_timesteps=self.total_timesteps_model, callback=[])
		return trained_model
	
	def test_model(self, model):
		_, _, allocation_values = DRLAgent.DRL_prediction(model=model, environment=self.env_test)
		bat = BackTest(model, self.env_test)
		results = bat.evaluate(allocation_values)
		return results

	def save_hyperparameters_metrics(self, trial_number, hyperparameters, metrics):
		_return = metrics["return"]
		_sharpe = metrics["sharpe"]
		with open(f"{self.logs_base_dir}/trial_{trial_number}_{self.model_name}_{_return:0.2}_{_sharpe:0.2}.pkl", 'wb') as fp:
			pickle.dump(hyperparameters, fp, protocol=pickle.HIGHEST_PROTOCOL)

	def run_study(self):
		study = optuna.create_study(study_name=self.study_name, load_if_exists=True, direction="maximize")
		study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1)
		joblib.dump(study, f"{self.logs_base_dir}/study.pkl")		
		print(f"Best trial:\n{study.best_trial}")

	

	
	
