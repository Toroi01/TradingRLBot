import logging
logging.basicConfig(level=logging.INFO)
from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

from src.config import config
from src.model.runner import train_model, test_model
from src.preprocessing import data
from src.environment.env_custom import CustomTradingEnv
from src.model.models import DRLAgent

if __name__ == '__main__':
    start_date_test = config.START_TEST_DATE
    end_date_test = config.END_TEST_DATE

    logging.info("Loading dataset")
    df = data.load_processed_df()

    test = data.data_split(df, start_date_test, end_date_test)

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

    env_test = CustomTradingEnv(df=test, **env_params)

    model_name = "ddpg"
    model_params = config.BEST_DDPG_PARAMS

    logging.info("Loading the best model")
    agent = DRLAgent(env=env_test)
    model = agent.get_model(model_name=model_name, model_kwargs=model_params)
    model = model.load(config.TRAINED_MODEL_DIR+"/best_model.pkl")

    logging.info("Testing the best model")
    results = test_model(test, env_params, model, with_graphs=True)


    logging.info(f"Results: {results}")
