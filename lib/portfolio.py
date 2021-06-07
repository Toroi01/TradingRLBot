class Portfolio:

    def __init__(self, cash):
        self.cash = cash
        self._amounts = {}

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
        self._amounts = {}

    def items(self):
        return self._amounts.items()

    def buy(self, ticker, amount, price, comission_type):
        """
        Check if there's enough cash to buy an asset and perform the operation.
        :param ticker: Asset name
        :param amount: Amount of asset
        :param price: Price of the asset
        :param comission_type: Type of comission to apply
        :return:
        """
        amount_in_cash = amount * price
        amount_in_cash += Portfolio.calculate_comission(amount_in_cash, comission_type)

        if self.cash < amount_in_cash:
            raise RuntimeError("Insufficient cash to perform buy")

        self.add(ticker, amount)
        self.cash -= amount_in_cash

    def sell(self, ticker, amount, price, comission_type):
        """
        Check if there's enough amount of an asset to sell it and perform the operation.
        :param ticker: Asset name
        :param amount: Amount of asset
        :param price: Price of the asset
        :param comission_type: Type of comission to apply
        :return:
        """
        if self.get_amount(ticker) < amount:
            raise RuntimeError(f"Insufficient amount in [{ticker}] to perform sell")

        self.remove(ticker, amount)

        amount_in_cash = amount * price
        amount_in_cash -= Portfolio.calculate_comission(amount_in_cash, comission_type)
        self.cash += amount_in_cash

    @staticmethod
    def calculate_comission(amount, comission_type):
        if comission_type is None:
            return 0
