import logging

logging.basicConfig(level=logging.INFO)

from src.config import config
from src.hyperparameter_tuning.tune import TuneBuilder

if __name__ == '__main__':
    start_date = "2020-01-01"
    end_date = "2021-05-01"

    env_params = {
        "initial_amount": 10000,
        "max_amount_per_trade": 1000,
        "main_tickers": config.MULTIPLE_TICKER_8,
        "all_tickers": config.MULTIPLE_TICKER_8,
        "reward_type": "percentage",
        "discrete_actionspace": True,
        "comission_value": 0.01
    }

    model_name = "dqn"

    tsv_params = {
        "num_splits": 1,
        "total_timesteps_model": 1,
        "with_graphs": False
    }

    n_trials = 1

    tuner = TuneBuilder.load(model_name, n_trials, env_params, tsv_params, start_date, end_date)
    results = tuner.run_study()

    logging.info(f"Results: {results}")
