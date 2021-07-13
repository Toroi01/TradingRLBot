import logging
logging.basicConfig(level=logging.INFO)
from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

from src.config import config
from src.preprocessing import data
from src.evaluate.time_series_validation import TimeSeriesValidation


if __name__ == '__main__':
    start_date_all = "2020-01-01"
    end_date_all = "2021-05-01"

    logging.info("Loading dataset")

    df = data.load_processed_df()
    df = data.data_split(df, start_date_all, end_date_all)

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
    model_params = config.BEST_PPO_PARAMS
    total_timesteps_model = 100000

    logging.info("Doing time series validation")

    tsv = TimeSeriesValidation(num_splits=3, total_timesteps_model=total_timesteps_model, with_graphs=True)
    results = tsv.run(df, env_params, model_name, model_params)

    logging.info(f"Results: {results}")
