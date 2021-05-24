class Action:

    BUY = "buy"
    SELL = "sell"

    def __init__(self, action_name, ticker, amount):
        assert action_name in (Action.BUY, Action.SELL)
        self.name = action_name
        self.ticker = ticker
        self.amount = amount
