from os.path import dirname, abspath
ROOT_DIR = dirname(dirname(abspath(__file__))) 


DATA_SAVE_DIR = f"{ROOT_DIR}/dataset/crypto_dataset_binance"
DATA_GRANULARITY = "1h"
TRAINED_MODEL_DIR = f"{ROOT_DIR}/trained_models"
TENSORBOARD_LOG_DIR = f"{ROOT_DIR}/tensorboard_log_from_main"
RESULTS_DIR = f"{ROOT_DIR}/results"

LOG_DIR_HYPERPARAMETER_TUNING =f"{ROOT_DIR}/hyperparameter_tuning/log"


## time_fmt = '%Y-%m-%d'
START_DATE = "2020-01-01"
START_TEST_DATE = "2021-01-01"
END_DATE = "2021-07-01"

HT_START_TRAIN_DATE = "2021-01-01"
HT_END_TRAIN_DATE = "2021-06-01"

HT_START_TEST_DATE = "2021-06-01"
HT_END_TEST_DATE = "2021-07-01"

PREPROCESSED_DF_NAME = "preprocess_df.pkl"



TECHNICAL_INDICATORS_LIST = ['psar', 
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


TECHNICAL_INDICATORS_LIST2 = ["macd",
                             "boll_ub",
                             "boll_lb",
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
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "batch_size": 64,
    "ent_coef": "auto_0.1",
}

