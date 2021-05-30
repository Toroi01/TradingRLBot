import numpy as np
import pandas as pd
from pyfolio import create_full_tear_sheet

from model.models import DRLAgent


class BackTest:

    def __init__(self, model, test_gym):
        self.model = model
        self.test_gym = test_gym
        self.summary = []

    def run(self, hourly_returns, hourly_allocation):
        returns, positions, transactions = self.prepare_full_tearsheet(hourly_returns, hourly_allocation)
        return create_full_tear_sheet(returns, positions, transactions=None, market_data=self.test_gym.df)

    def prepare_full_tearsheet(self, df_daily_return, df_allocation_per_tick):
        returns = self.calculate_returns(df_daily_return)
        positions = self.calculate_positions(df_allocation_per_tick)
        return returns, positions, None

    def calculate_returns(self, df):
        # Here we pass from hourly to daily date in order to use pyfolio correctly
        strategy_returns = df.copy()
        strategy_returns["datetime"] = pd.to_datetime(strategy_returns["date"])
        strategy_returns.drop("date", axis=1, inplace=True)
        strategy_returns["date"] = pd.to_datetime(strategy_returns["datetime"].dt.date)
        strategy_returns.set_index("date", drop=False, inplace=True)

        def cumulative_return_intraday(returns):
            values = returns.values
            values = np.array(values) + 1
            return np.product(values) - 1

        ts = strategy_returns["daily_return"].groupby(strategy_returns.index).agg(
            cumulative_return_intraday).squeeze()

        return ts

    def calculate_positions(self, df_allocation_per_tick):
        df_allocation = df_allocation_per_tick.copy().reset_index()
        df_allocation["date"] = pd.to_datetime(df_allocation.date).dt.date # Transforming to date
        df_allocation["date"] = pd.to_datetime(df_allocation["date"])
        df_allocation = df_allocation.groupby("date").first()
        return df_allocation


