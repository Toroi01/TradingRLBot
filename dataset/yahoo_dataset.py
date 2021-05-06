import logging

import pandas as pd
import yfinance as yf

"""
Downloads finance data of specific tickers with daily granularity
Inspired by FinRL
https://github.com/AI4Finance-LLC/FinRL/blob/master/finrl/marketdata/yahoodownloader.py
"""


class YahooDataset:

    def __init__(self, start_date: str, end_date: str, ticker_list: list, output_path: str = "../resources"):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.output_path = \
            f"{output_path}/stocks/yahoo-dataset-{self.start_date}-{self.end_date}-n{len(self.ticker_list)}.csv"

    def _fetch_raw_data(self) -> pd.DataFrame:
        """
        Fetches data from Yahoo API
        :return pd.DataFrame: 7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        data_df = pd.DataFrame()
        for ticker in self.ticker_list:
            temp_df = yf.download(ticker, start=self.start_date, end=self.end_date)
            temp_df["ticker"] = ticker
            data_df = data_df.append(temp_df)

        return data_df.reset_index()

    def download_data(self) -> pd.DataFrame:
        raw_data = self._fetch_raw_data()
        # convert the column names to standardized names
        raw_data.columns = [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "adjcp",
            "volume",
            "ticker",
        ]
        # use adjusted close price instead of close price
        raw_data["close"] = raw_data["adjcp"]
        # drop the adjusted close price column
        raw_data.drop("adjcp", axis=1, inplace=True)
        # create day of the week column (monday = 0)
        # convert date to standard string format, easy to filter
        logging.info("Shape of DataFrame: ", raw_data.shape)

        # storing dataset in output path
        try:
            raw_data.to_csv(self.output_path, index = False)
        except FileNotFoundError:
            logging.error(f"Make sure that the folder [{self.output_path}] is created")

    def load(self):
        try:
            return pd.read_csv(self.output_path)
        except FileNotFoundError as e:
            logging.error(f"The output path provided does not exist. If it's the first time you call"
                          f" this method, make sure to call download_data first.")
            raise e


