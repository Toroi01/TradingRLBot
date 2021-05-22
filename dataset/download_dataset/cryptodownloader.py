import logging
import os
import pandas as pd

"""
Downloads crypto data from Kaggle with hourly granularity
"""

class CryptoDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list, output_path: str = './dataset/crypto_dataset'):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.output_path = f"{output_path}"

    def download_data(self):
        try:
            import kaggle
        except OSError as e:
            logging.error("Your credentials are not configured. You should follow the steps on this website."
                          "https://github.com/Kaggle/kaggle-api/issues/15")
            logging.error(e)

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('tencars/392-crypto-currency-pairs-at-minute-resolution',
                                          path=self.output_path,
                                          unzip=True)

        # Removing the unnecessary files
        for file in os.listdir(self.output_path):
            # If the file is not a comparison within crypto and usd, we discard it
            if not file.endswith("usd.csv"):
                os.remove(self.output_path+'/'+file)

    def load(self):
        df = pd.DataFrame()
        for ticker in self.ticker_list:
            try:
                temp_df = pd.read_csv(f"{self.output_path}/{ticker.lower()}usd.csv")
                temp_df["tic"] = ticker.lower()
                df = df.append(temp_df)

            except FileNotFoundError as e:
                logging.error(f"Ticker [{ticker}] does not exist or is not downloaded. If it's the first time you call"
                              f" this method, make sure to call download_data first.")
                raise e

        df = df.reset_index()
        
        # parsing to datetime and filtering by dates
        df['datetime'] = pd.to_datetime(df['time'] / 1000, unit='s')
        df = df[(df['datetime'] > self.start_date) & (df['datetime'] < self.end_date)]
        df.drop('time', axis=1, inplace=True)
        df.drop('index', axis=1, inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna()
        df = df.reset_index(drop=True)
        df = df.sort_values(by=['datetime','tic']).reset_index(drop=True)
        return df
