import pandas as pd

from lib.action import Action
from lib.portfolio import Portfolio


class BackTest:

    def __init__(self, cash=1000, ticker_column_name="tic", comission_type=None):
        """
        Each row should be a different timestamp and ticker.
        :param initial_capital: Budget in dollars at the beginning of the window
        :param ticker_column_name: Name of the column where the ticker is specified
        """
        self.ticker_column_name = ticker_column_name
        self.comission_type = comission_type

        # Object to keep track of the portfolio allocation
        self.portfolio = Portfolio(cash=cash)

        # Output results
        self.historic_portfolio = pd.DataFrame()
        self.historic_capital = pd.DataFrame()

    def run(self, model, dataset):
        """
        :param model: Model implementing AbstractRLModel
        :param dataset: Dataset with all the necessary features and preprocessing for the given model.
        """
        data = self.adapt_data(dataset)
        for i, data_row in data.iterrows():
            actions = model.decide_actions(data_row, self.portfolio)
            if len(actions) > 0:
                # Update capital distribution
                self.update_portfolio(actions, data_row)

            # Log capital value based on the market
            self.log_capital(data_row)
            # Log portfolio status
            self.log_portfolio(data_row)

    def adapt_data(self, dataset: pd.DataFrame):
        """
        Transform dataframe with one row per ticker and timestamp to one dataframe with tickers as columns
        :return: Dataframe with timestamp and tickers as columns
        """
        return dataset.pivot_table(index="datetime", columns=self.ticker_column_name).reset_index()

    def update_portfolio(self, actions, data_row):
        for action in actions:
            if action.name == Action.SELL:
                self.portfolio.sell(action.ticker, action.amount, data_row.close[action.ticker], self.comission_type)

            elif action.name == Action.BUY:
                self.portfolio.buy(action.ticker, action.amount, data_row.close[action.ticker], self.comission_type)

    def log_capital(self, data_row):
        """
        Add a row to the historic capital with the current status of the portfolio
        :param data_row: Row of the input dataset
        """
        total_monetary_value = 0
        for ticker, amount in self.portfolio.items():
            total_monetary_value += amount * data_row.close[ticker]

        total_monetary_value += self.portfolio.cash
        capital_row = pd.DataFrame({"total_capital": [total_monetary_value]})
        capital_row["datetime"] = data_row.datetime.iloc[0]

        self.historic_capital = self.historic_capital.append(capital_row)

    def log_portfolio(self, data_row):
        """
        Add a row to the historic portfolio with the current allocation
        :param data_row: Row of the input dataset
        """
        portfolio_sanitized = {k: [v] for k, v in self.portfolio.items()}
        portfolio_row = pd.DataFrame(portfolio_sanitized)
        portfolio_row["datetime"] = data_row.datetime.iloc[0]

        self.historic_portfolio = self.historic_portfolio.append(portfolio_row)

    # TODO:Add summary methods
