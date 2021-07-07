import logging
logging.basicConfig(level=logging.INFO)

from src.config import config
from src.model.runner import train_model, test_model
from src.preprocessing import data

if __name__ == '__main__':
    start_date = "2020-01-31"
    end_date = "2021-05-15"
    start_date_test = "2021-06-01"
    end_date_test = "2021-07-01"

    logging.info("Loading dataset")
    df = data.load_processed_df()

    train = data.data_split(df, start_date, end_date)
    test = data.data_split(df, start_date_test, end_date_test)

    features = data.build_features(df)

    env_params = {
        "initial_amount": 1000000,
        "features": features,
        "max_assets_amount_per_trade": 100,
        "main_tickers": config.MULTIPLE_TICKER_8,
        "all_tickers": config.MULTIPLE_TICKER_8,
        "reward_type": "percentage",
        "comission_value": 0.01
    }

    model_name = "ppo"
    model_params = config.PPO_PARAMS
    total_timesteps_model = 100000

    logging.info("Training model")
    model = train_model(train, env_params, model_name, model_params, total_timesteps_model)
    logging.info("Testing model")
    results = test_model(test, env_params, model)

    logging.info(f"Results: {results}")
