import logging
import os
import pandas as pd
import requests, zipfile, io
import datetime
from datetime import timedelta, date
from dateutil.relativedelta import relativedelta
import numpy as np

"""
Downloads crypto data from https://www.binance.com/ 
Doc: https://github.com/binance/binance-public-data/
"""

class CryptoDownloader_binance:
    '''
    A downloader class to download and load crypto dataset
    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        ticker_list : list
            a list of stock tickers (modified from config.py)
        output_path: str (modified from config.py)
            path to specify where to download the dataset
        granularity: str (modified from config.py)
            the time scale or level of detail in the data
            (all intervals are supported: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1mo)
    
    Methods
    -------
        month_year_iter
            a helper method to iterate through months
        download_data
            download the data from binance and store it at output_path
        load   
            loads the downloadad data 
    '''
    def __init__(self, start_date: str, end_date: str, ticker_list: list, output_path: str = './dataset/crypto_dataset_binance', granularity: str = "1h"):
        self.start_date = datetime.date.fromisoformat(start_date) 
        self.end_date = datetime.date.fromisoformat(end_date)
        self.ticker_list = [e.lower() for e in ticker_list]
        self.output_path = f"{output_path}"
        self.granularity = granularity
       
    def month_year_iter(self, start_month, start_year, end_month, end_year):
        ym_start= 12*start_year + start_month - 1
        ym_end= 12*end_year + end_month - 1
        for ym in range( ym_start, ym_end ):
            y, m = divmod( ym, 12 )
            yield y, m+1
            
    def download_data(self):     
        for year, month in self.month_year_iter(self.start_date.month, self.start_date.year, self.end_date.month, self.end_date.year):
            start_date = f"{year:04d}-{month:02d}"
            for tic in self.ticker_list:
                tic_pair = tic.upper()+"USDT"
                if not os.path.exists(f"{self.output_path}/{tic_pair}-{self.granularity}-{start_date}.csv"):
                    url = f"https://data.binance.vision/data/spot/monthly/klines/{tic_pair}/{self.granularity}/{tic_pair}-{self.granularity}-{start_date}.zip"
                    resp = requests.get(url)
                    if resp.ok:
                        z = zipfile.ZipFile(io.BytesIO(resp.content))
                        z.extractall(self.output_path)
                        print(f"Downloaded {url}")
                    else:
                        print (f'Error url! {url}')
                        print (f'Error resp! {resp}')       

    def load(self):
        if not os.path.exists(f"{self.output_path}/full_dataset.csv"):
            df = pd.DataFrame()
            for year, month in self.month_year_iter(self.start_date.month, self.start_date.year, self.end_date.month, self.end_date.year):
                start_date = f"{year:04d}-{month:02d}"
                if month == 12:
                    end_date = f"{year+1}-01-01"
                else:
                    end_date = f"{year}-{month+1:02d}-01"
                
                dates = pd.date_range(start=start_date,end=end_date, freq=self.granularity, closed='left')
                    
                for tic in self.ticker_list:
                    tic_pair = tic.upper()+"USDT"
                    file_name = f"{tic_pair}-{self.granularity}-{start_date}.csv"
                    temp_df_1 = pd.DataFrame(columns=["open","high","low","close","volume"])
                    temp_df_1["tic"] = tic
                    temp_df_1["date"] = dates                           

                    try:
                        temp_df = pd.read_csv(f"{self.output_path}/{file_name}", header=None, usecols=[0,1,2,3,4,5],names=["timestamp","open","high","low","close","volume"])
                        temp_df["tic"] = tic.lower()
                        temp_df["date"] = pd.to_datetime(temp_df["timestamp"] / 1000, unit='s')
                        temp_df.drop("timestamp", axis=1, inplace=True)
                        
                        temp_df = temp_df.set_index('date')
                        temp_df_1 = temp_df_1.set_index('date')
                        temp_df_1.update(temp_df)
                        temp_df_1["tic"] = tic
                        
                    except FileNotFoundError as e:
                        logging.error(f"File [{file_name}] does not exist or is not downloaded")
                    
                    temp_df_1 = temp_df_1.reset_index()
                    df = df.append(temp_df_1, ignore_index=True)
                    df.fillna(method='ffill', inplace=True)

            df.to_csv(f"{self.output_path}/full_dataset.csv",index=False)
            return df
        else:
            return pd.read_csv(f"{self.output_path}/full_dataset.csv")


    def add_usd(self, df):
        usd = df.groupby('date').sum().reset_index()
        usd[['open', 'high', 'low', 'close']] = 1
        usd['tic'] = 'cash'
        return df.append(usd)