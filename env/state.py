class State:
    """
    Stores the status of the step:
    - Cash
    - Allocation in each ticker
    - Hourly close prices
    - Technical indicators for all assets
    """

    def __init__(self, technical_indicator_list):
        self.technical_indicator_list = technical_indicator_list
        self.portfolio = None
        self.hourly_data = None

    def values(self):
        """
        Cash, close values, allocation and technical indicators
        :return: Array of values
        """
        values = [self.portfolio.cash] + self.hourly_data.close.values.tolist() + self.portfolio.values()
        for name in self.technical_indicator_list:
            values += self.hourly_data[name].values.tolist()
        return values


    def update(self, portfolio, hourly_data):
        self.portfolio = portfolio
        self.hourly_data = hourly_data

    def reset(self):
        self.portfolio = None
        self.hourly_data = None

    def get_size(self, num_tickers):
        return 1 + num_tickers * 2 + len(self.technical_indicator_list) * num_tickers
