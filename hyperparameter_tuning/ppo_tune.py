
from config import config
from trade.time_series_validation import TimeSeriesValidation


data_downloader = CryptoDownloader_binance(config.START_DATE, config.END_DATE, config.MULTIPLE_TICKER_8, config.DATA_SAVE_DIR, config.DATA_GRANULARITY)
if download_data:    
    data_downloader.download_data()
df = data_downloader.load()



#df

#env_params

#model_name

#model_params
#self, df, env_params, model_name, model_params
tsv = TimeSeriesValidation()
tsv.run()