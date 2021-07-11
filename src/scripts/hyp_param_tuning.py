import logging
logging.basicConfig(level=logging.INFO)
from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

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
        "discrete_actionspace": False,
        "comission_value": 0.01
    }

    model_name = "ppo"

    tsv_params = {
        "num_splits": 3,
        "total_timesteps_model": 100000,
        "with_graphs": False
    }

    n_trials = 10

    tuner = TuneBuilder.load(model_name, n_trials, env_params, tsv_params, start_date, end_date)
    results = tuner.run_study()

    logging.info(f"Results: {results}")
