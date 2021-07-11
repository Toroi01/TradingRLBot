import logging

logging.basicConfig(level=logging.INFO)

from src.config import config
from src.hyperparameter_tuning.tune import TuneBuilder

if __name__ == '__main__':
    start_date = config.HT_START_TRAIN_DATE
    end_date = config.HT_END_TRAIN_DATE

    env_params = {
        "initial_amount": 10000,
        "max_amount_per_trade": 1000,
        "main_tickers": config.MULTIPLE_TICKER_8,
        "all_tickers": config.MULTIPLE_TICKER_8,
        "reward_type": "percentage",
        "discrete_actionspace": False,
        "comission_value": 0.01
    }

    model_name = "ppo"

    tsv_params = {
        "num_splits": 1,
        "total_timesteps_model": 100,
        "with_graphs": False
    }

    n_trials = 1

    tuner = TuneBuilder.load(model_name, n_trials, env_params, tsv_params, start_date, end_date)
    results = tuner.run_study()

    logging.info(f"Results: {results}")
