class State:
    """
    Stores the status of the step:
    - Cash
    - Allocation in each ticker
    - Hourly close prices
    - Technical indicators for all assets
    """

    def __init__(self, features, ticker_list):
        self.features = features
        self.ticker_list = ticker_list
        self.portfolio = None
        self.hourly_data = None

    def values(self):
        """
        Cash, close values of all tickers, allocation of main tickers and technical indicators of all tickers
        :return: Array of values
        """
        values = [self.portfolio.cash] + self.hourly_data.close.values.tolist() + self.portfolio.values()
        for name in self.features:
            values += self.hourly_data[name].values.tolist()
        return values

    def update(self, portfolio, hourly_data):
        self.portfolio = portfolio
        self.hourly_data = hourly_data

    def reset(self):
        self.portfolio = None
        self.hourly_data = None

    def get_size(self, main_tickers, all_tickers):
        """
        Size of the status array
        :param main_tickers: Tickers used to buy and sell
        :param all_tickers: All tickers used to buy or just for information
        :return: Size of the state variable
        """

        cash_size = 1
        all_tickers_close_prices_size = len(all_tickers)
        main_tickers_allocation = len(main_tickers)
        all_tickers_indicators = len(self.features) * len(all_tickers)
        return cash_size + all_tickers_close_prices_size + main_tickers_allocation + all_tickers_indicators
