from src.config import config
from src.model.runner import train_model
from src.preprocessing import data
import logging
from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    start_date = config.START_TRAIN_DATE
    end_date = config.END_TRAIN_DATE

    logging.info("Loading dataset")

    df = data.load_processed_df()
    train = data.data_split(df, start_date, end_date)

    features = data.build_features(df)


    discrete_actionspace = True if config.BEST_MODEL_NAME == "dqn" else False
    env_params = {
        "initial_amount": 10000,
        "features": features,
        "max_amount_per_trade": 1000,
        "main_tickers": config.MULTIPLE_TICKER_8,
        "all_tickers": config.MULTIPLE_TICKER_8,
        "reward_type": "percentage",
        "comission_value": 0.01,
        "discrete_actionspace": discrete_actionspace,
    }

    model_name =  config.BEST_MODEL_NAME
    model_params = config.BEST_MODEL_PARAMS

    total_timesteps_model = 1

    logging.info("Training model")

    model = train_model(train, env_params, model_name, model_params, total_timesteps_model, log_tensorboard=config.TRAINED_MODEL_DIR, tb_name="best_model")
    model.save(config.TRAINED_MODEL_DIR+f"/best_model_{config.BEST_MODEL_NAME}.pkl")
