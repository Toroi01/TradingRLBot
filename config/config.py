DATA_SAVE_DIR = f"./dataset/crypto_dataset_binance"
DATA_GRANULARITY = "1h"
TRAINED_MODEL_DIR = f"./trained_models"
TENSORBOARD_LOG_DIR = f"./tensorboard_log"
RESULTS_DIR = f"./results"

## time_fmt = '%Y-%m-%d'
START_DATE = "2020-01-01"
END_DATE = "2021-04-21"

TECHNICAL_INDICATORS_LIST = ["macd","boll_ub","boll_lb","rsi_30", "cci_30", "dx_30","close_30_sma","close_60_sma"]

SINGLE_TICKER = ['BTC']

TWO_TICKER = ['BTC','ETH']

MULTIPLE_TICKER_8 = ["BTC","ETH","BNB","ADA","XRP","DOGE","LINK","LTC"]


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
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "batch_size": 64,
    "ent_coef": "auto_0.1",
}

########################################################
############## Stock Ticker Setup starts ##############
SINGLE_TICKER = ["BTC"]