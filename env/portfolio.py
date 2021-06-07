import logging

import numpy as np
import pandas as pd

"""
Class used to store the amount of assets that we have from each ticker
"""


class Portfolio:

    def __init__(self, cash, ticker_list):
        # This is the fixed initial cash
        self.initial_cash = cash
        # This value will be updated in each iteration
        self.cash = cash
        self.ticker_list = ticker_list
        self._amounts = {ticker: 0.0 for ticker in ticker_list}

        self.historic_transactions = pd.DataFrame()

    def get_amount(self, ticker):
        if ticker not in self._amounts:
            self._amounts[ticker] = 0
        return self._amounts[ticker]

    def add(self, ticker, amount):
        if ticker not in self._amounts:
            self._amounts[ticker] = 0
        self._amounts[ticker] += amount

    def remove(self, ticker, amount):
        self._amounts[ticker] -= amount

    def reset(self):
        self.cash = self.initial_cash
        self._amounts = {ticker: 0.0 for ticker in self.ticker_list}
        self.historic_transactions = pd.DataFrame()

    def items(self):
        return self._amounts.items()

    def values(self):
        return list(self._amounts.values())

    def to_df(self):
        all_amounts = {**{"cash": self.cash}, **self._amounts}
        # This is needed in order to transform it into a df
        all_amounts = {k: [v] for k, v in all_amounts.items()}
        return pd.DataFrame(all_amounts)

    def buy(self, ticker, amount, price, timestamp, comission_value):
        """
        Check if there's enough cash to buy an asset and perform the operation.
        :param ticker: Asset name
        :param amount: Amount of asset
        :param price: Price of the asset
        :param comission_value: Type of comission to apply
        :return:
        """
        amount_in_cash = amount * price
        amount_in_cash += Portfolio.calculate_comission(amount_in_cash, comission_value)

        if self.cash < amount_in_cash:
            logging.debug("Insufficient cash to perform entire buy.")
            if self.cash > 0:
                amount = (self.cash - Portfolio.calculate_comission(self.cash, comission_value)) // price
                amount_in_cash = amount * price
                amount_in_cash += Portfolio.calculate_comission(amount_in_cash, comission_value)

        self.add(ticker, amount)
        self.cash -= amount_in_cash

        # Log transaction
        self.log_transaction(timestamp, 'buy', ticker, amount, price)

    def sell(self, ticker, amount, price, timestamp, comission_value):
        """
        Check if there's enough amount of an asset to sell it and perform the operation.
        :param ticker: Asset name
        :param amount: Amount of asset
        :param price: Price of the asset
        :param comission_value: Type of comission to apply
        :return:
        """
        if self.get_amount(ticker) < amount:
            logging.debug(f"Insufficient amount in [{ticker}] to perform sell. Selling everything.")
            amount = self.get_amount(ticker)
            if amount == 0:
                return

        self.remove(ticker, amount)

        amount_in_cash = amount * price
        amount_in_cash -= Portfolio.calculate_comission(amount_in_cash, comission_value)
        self.cash += amount_in_cash

        # Log transaction
        self.log_transaction(timestamp, 'sell', ticker, amount, price)

    def get_total_portfolio_value(self, hourly_data):
        """
        Using the last close prices, compute the value of the portfolio considering the cash and the value
        of the assets
        :param hourly_data: Data from the last hour
        :return: Portfolio value in dollars
        """
        return np.sum(list(self.get_assets_value(hourly_data).values()))

    def get_assets_value(self, hourly_data):
        """
        Given the hourly data, return the value that the portfolio has for each asset
        """
        price_per_asset = Portfolio.get_price_per_asset(hourly_data)
        asset_value = {"cash": self.cash}
        for asset, amount in self._amounts.items():
            asset_value[asset] = amount * price_per_asset[asset]["close"]
        return asset_value

    def log_transaction(self, timestamp, operation, ticker, amount, price):
        new_entry = pd.DataFrame(
            {'date': [timestamp], 'operation': [operation], 'tic': [ticker], 'amount': [amount], 'price': [price]})
        self.historic_transactions = self.historic_transactions.append(new_entry)

    @staticmethod
    def get_price_per_asset(hourly_data):
        ticker_and_price = hourly_data[['tic', 'close']]
        ticker_and_price.set_index('tic', inplace=True)
        return ticker_and_price.to_dict('index')

    @staticmethod
    def calculate_comission(amount_in_cash, comission_value):
        if comission_value is None:
            return 0
        else:
            return amount_in_cash * comission_value
