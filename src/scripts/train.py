from src.config import config
from src.model.runner import train_model
from src.preprocessing import data
import logging
from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    start_date = "2021-01-01"
    end_date = "2021-04-01"

    logging.info("Loading dataset")

    df = data.load_processed_df()
    train = data.data_split(df, start_date, end_date)

    features = data.build_features(df)

    env_params = {
        "initial_amount": 10000,
        "features": features,
        "max_amount_per_trade": 1000,
        "main_tickers": config.MULTIPLE_TICKER_8,
        "all_tickers": config.MULTIPLE_TICKER_8,
        "reward_type": "percentage",
        "comission_value": 0.01
    }

    model_name = "ppo"
    model_params = config.PPO_PARAMS
    total_timesteps_model = 100000

    logging.info("Training model")

    train_model(train, env_params, model_name, model_params, total_timesteps_model)
