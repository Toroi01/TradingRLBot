import numpy as np
import pandas as pd
from pyfolio import create_full_tear_sheet
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from model.models import DRLAgent


class BackTest:

    def __init__(self, model, test_gym):
        self.model = model
        self.test_gym = test_gym
        self.summary = []

    def create_summary(self, allocation_amounts, allocation_values, transactions):
        returns = self._calculate_returns(allocation_values)
        positions = self._calculate_positions(allocation_amounts)
        return create_full_tear_sheet(returns, positions, transactions=None, market_data=self.test_gym.df)

    def _calculate_returns(self, df):
        # Here we pass from hourly to daily date in order to use pyfolio correctly
        strategy_returns = df.copy()
        strategy_returns["datetime"] = pd.to_datetime(strategy_returns["date"])
        strategy_returns.drop("date", axis=1, inplace=True)
        strategy_returns["date"] = pd.to_datetime(strategy_returns["datetime"].dt.date)
        strategy_returns.set_index("date", drop=False, inplace=True)
        strategy_returns["hourly_return"] = strategy_returns.sum(axis=1).pct_change()

        def cumulative_return_intraday(returns):
            values = returns.values
            values = np.array(values) + 1
            return np.product(values) - 1

        ts = strategy_returns["hourly_return"].groupby(strategy_returns.index).agg(
            cumulative_return_intraday).squeeze()

        return ts

    def _calculate_positions(self, df_allocation_per_tick):
        df_allocation = df_allocation_per_tick.copy().reset_index()
        df_allocation["date"] = pd.to_datetime(df_allocation.date).dt.date # Transforming to date
        df_allocation["date"] = pd.to_datetime(df_allocation["date"])
        df_allocation = df_allocation.groupby("date").first()
        return df_allocation

    def plot_return_against_hold(self, allocation_values):
        fig = make_subplots(rows=1)
        allocation_values["hourly_return"] = allocation_values.sum(axis=1).pct_change()

        # Holding different cryptos
        fig.add_trace(
            go.Scatter(x=allocation_values.date, y=BackTest.to_cumulative(allocation_values.hourly_return), name="Strategy"),
            row=1, col=1
        )
        market_data = self.test_gym.df.pivot_table(index='date', columns=["tic"], values=["close"]).reset_index()

        for tic in self.test_gym.all_tickers:
            crypto_evolution = BackTest.to_cumulative(market_data.close[tic].pct_change())
            fig.add_trace(
                go.Scatter(x=market_data.date, y=crypto_evolution, name=tic),
                row=1, col=1
            )
        fig.show()

    @staticmethod
    def to_cumulative(series):
        return (series + 1).cumprod()
