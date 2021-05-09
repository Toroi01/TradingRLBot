# TradingRLBot

## Download and use a dataset
Right now we have two different datasets: Crypto and Stocks (Yahoo).

Both datasets have the same columns: ('open', 'close', 'high', 'low', 'volume', 'ticker', 'datetime').
Keep in mind that the Crypto one has hour granularity whereas the Stocks one has daily.

In order to use a dataset, you just need to import the specific dataset class
and call:

Crypto:
```
dataset = CryptoDataset('2021-05-01', '2021-05-04', ['BTC', 'ETH'])
df = dataset.load()
```

Stocks:
```
dataset = YahooDataset('2021-05-01', '2021-05-04', ['BTC', 'ETH'])
df = dataset.load()
```

If it's the first time that you're calling it or if you're using a different ticker
you will need to call ```download_data``` first.

```
dataset = CryptoDataset('2021-05-01', '2021-05-04', ['BTC', 'ETH'])
dataset.download_data()
df = dataset.load()
```

Stocks:
```
dataset = YahooDataset('2021-05-01', '2021-05-04', ['BTC', 'ETH'])
dataset.download_data()
df = dataset.load()
```