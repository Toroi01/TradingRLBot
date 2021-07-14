from os.path import dirname, abspath
ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))

DATA_SAVE_DIR = f"{ROOT_DIR}/data/crypto_dataset_binance"
DATA_GRANULARITY = "1h"
TRAINED_MODEL_DIR = f"{ROOT_DIR}/trained_models"
TENSORBOARD_LOG_DIR = f"{ROOT_DIR}/logs/tensorboard"
RESULTS_DIR = f"{ROOT_DIR}/results"

LOG_DIR_HYPERPARAMETER_TUNING =f"{ROOT_DIR}/logs/hyp_tune"
MYSQL_DB = "mysql://root:0000@localhost/TradingRLBot"


## time_fmt = '%Y-%m-%d'
START_DATE = "2020-01-01"
START_TEST_DATE = "2021-01-01"
END_DATE = "2021-07-01"

HT_START_TRAIN_DATE = "2021-01-01"
HT_END_TRAIN_DATE = "2021-06-01"

HT_START_TEST_DATE = "2021-06-01"
HT_END_TEST_DATE = "2021-07-01"

START_TRAIN_DATE = "2020-01-01"
END_TRAIN_DATE = "2021-04-01"

START_TEST_DATE = "2021-04-01"
END_TEST_DATE = "2021-06-01"

PREPROCESSED_DF_NAME = "preprocess_df.pkl"


# This indicators describe what is appening during the day (up to 20 hours before)
TECHNICAL_INDICATORS_SHORTPERIOD = [
                                     'psar', 
                                     'ui', 
                                     'atr', 
                                     'bbw',
                                     'bbp', 
                                     'bbhi',
                                     'bbli',
                                     'kcp',
                                     'kchi',
                                     'kcli',
                                     'macd', 
                                     'macd_diff',
                                     'mass_index',
                                     'dpo',
                                     'kst',
                                     'aroon_up',
                                     'aroon_down',
                                     'ppo']


# This indicators describe what happen in the days before up to a month
TECHNICAL_INDICATORS_LONGPERIOD = [
                                     "rsi_24",
                                     "rsi_168",
                                     "rsi_720",
                                     "cci_24",
                                     "cci_168",
                                     "cci_720",
                                     "dx_24",
                                     "dx_168",
                                     "dx_720",
                                     "close_24_sma",
                                     "close_168_sma",
                                     "close_720_sma"]

SINGLE_TICKER = ['BTC']

TWO_TICKER = ['BTC', 'ETH']

MULTIPLE_TICKER_8 = ["BTC", "ETH", "BNB", "ADA", "XRP", "DOGE", "LINK", "LTC"]

ACTUAL_TICKERS = MULTIPLE_TICKER_8

## Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 512,
    "ent_coef": 2.17e-08,
    "learning_rate": 0.00027910,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
DQN_PARAMS = {'buffer_size': 10000,"learning_rate": 0.00025, "exploration_initial_eps": 1.0, "exploration_fraction": .999985, "exploration_final_eps": 0.02, "gamma": 0.99, "batch_size": 100}
SAC_PARAMS = {
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "batch_size": 64,
    "ent_coef": "auto_0.1",
}

BEST_DDPG_PARAMS = {
    "gamma": 0.9752730589768152,
    "tau": 0.012649506082360071,
    "learning_rate": 0.00297393374407611,
    "batch_size": 128,
    "buffer_size": 10000,
    "seed":8,
}
BEST_PPO_PARAMS = {
    "n_steps": 1000,
    "ent_coef": 0.0108486,
    "learning_rate": 0.00637956,
    "batch_size": 20,
    "n_epochs": 3,
    "gamma": 0.995,
    "gae_lambda": 0.969964,
    "seed":8,
}
BEST_DQN_PARAMS = {
    "batch_size": 100,
    "gamma": 0.965,
    "learning_rate": 0.002,
    "tau": 0.014,
}

BEST_MODEL_NAME = "ddpg"
BEST_MODEL_PARAMS = {}

if BEST_MODEL_NAME == "ddpg":
    BEST_MODEL_PARAMS = BEST_DDPG_PARAMS
if BEST_MODEL_NAME == "ppo":
    BEST_MODEL_PARAMS = BEST_PPO_PARAMS
if BEST_MODEL_NAME == "dqn":
    BEST_MODEL_PARAMS = BEST_DQN_PARAMS
