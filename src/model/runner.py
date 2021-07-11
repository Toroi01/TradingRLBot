from src.environment.env_custom import CustomTradingEnv
from src.model.models import DRLAgent
from src.evaluate.backtest import BackTest
from src.preprocessing.data import build_features


def train_model(data, env_params, model_name, model_params, total_timesteps_model, log_tensorboard=None, tb_name=None):
    env_params["features"] = build_features(data)
    env_train_gym = CustomTradingEnv(df=data, **env_params)
    #env_train, _ = env_train_gym.get_sb_env()
    print(f"Train from [{data['date'].iloc[0]}] to [{data['date'].iloc[-1]}]")

    agent = DRLAgent(env=env_train_gym)
    model = agent.get_model(model_name=model_name, model_kwargs=model_params, tensorboard_log=log_tensorboard)
    return agent.train_model(model=model,
                             tb_log_name=tb_name,
                             total_timesteps=total_timesteps_model)


def test_model(data, env_params, model, with_graphs=False):
    print(f"Test from [{data['date'].iloc[0]}] to [{data['date'].iloc[-1]}]")
    env_params["features"] = build_features(data)
    env_test_gym = CustomTradingEnv(df=data, **env_params)
    _, _, allocation_values = DRLAgent.DRL_prediction(model=model, environment=env_test_gym)
    bat = BackTest(model, env_test_gym)
    results = bat.evaluate(allocation_values, data)
    if with_graphs:
        bat.plot_return_against_hold(allocation_values)
    return results
