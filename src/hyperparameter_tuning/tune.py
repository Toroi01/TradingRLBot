import os
import pickle
import time

import joblib
import optuna
import random

from src.config import config
from src.preprocessing import data


class Tune:
    def __init__(self, model_name, n_trials, env_params, tsv_params, start_date, end_date):
        self.model_name = model_name
        self.n_trials = n_trials
        self.env_params = env_params
        self.tsv_params = tsv_params
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

        timestamp = str(random.randint(0, 1e6)) + str(int(time.time()))
        run_path = timestamp + "_" + model_name
        self.logs_base_dir = f"{config.LOG_DIR_HYPERPARAMETER_TUNING}/{run_path}"
        self.log_tensorboard = f"{self.logs_base_dir}/log_tensorboard"
        self.study_name = f"{timestamp}_{model_name}"

    def save(self, name, trial_number, content):
        os.makedirs(self.logs_base_dir, exist_ok=True)
        if name=="hyperparameters":
            file_name = f"trial_{trial_number}_HYP.pkl"
        elif name=="metrics":
            file_name = f"trial_{trial_number}_METRICS.pkl"
        elif name=="model":
            file_name = f"trial_{trial_number}_MODEL.pkl"
        with open(f"{self.logs_base_dir}/{file_name}.pkl",'wb') as fp:
            pickle.dump(content, fp, protocol=pickle.HIGHEST_PROTOCOL)


    def run_study(self, storage="memory"):
        self.init_data()
        if storage=="memory":   
            study = optuna.create_study(study_name=self.study_name, direction="maximize")
        elif storage=="mysql":
            study = optuna.create_study(study_name=self.study_name, direction="maximize", storage=config.MYSQL_DB)
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1)
        joblib.dump(study, f"{self.logs_base_dir}/study.pkl")
        print(f"Best trial:\n{study.best_trial}")
        return study.best_trial

    def init_data(self):
        print("Initializing data and features")
        self.data = data.load_processed_df()
        self.data = data.data_split(self.data, self.start_date, self.end_date)
        self.env_params["features"] = data.build_features(self.data)


class TuneBuilder:

    @staticmethod
    def load(model_name, n_trials, env_params, tsv_params, start_date, end_date):
        if model_name == "ppo":
            # Needs to be imported here to avoid circular dependency
            from src.hyperparameter_tuning.ppo_tune import PPOTune
            return PPOTune(n_trials, env_params, tsv_params, start_date, end_date)
        if model_name == "dqn":
            # Needs to be imported here to avoid circular dependency
            from src.hyperparameter_tuning.dqn_tune import DQNTune
            return DQNTune(n_trials, env_params, tsv_params, start_date, end_date)
        if model_name == "ddpg":
            # Needs to be imported here to avoid circular dependency
            from src.hyperparameter_tuning.ddpg_tune import DDPGTune
            return DDPGTune(n_trials, env_params, tsv_params, start_date, end_date)
        else:
            raise NotImplementedError(f"Model [{model_name}] not implemented")