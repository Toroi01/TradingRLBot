
import pandas as pd 
from stockstats import StockDataFrame
import ta


# List of indicators 
indicators_list = [ 
    ('bbm', ta.volatility.bollinger_mavg, ['close']),
    ('atr', ta.volatility.average_true_range, ['high', 'low', 'close']),
    ('bbw', ta.volatility.bollinger_wband, ["close"]),
    ('bbp', ta.volatility.bollinger_pband, ['close']),
    ('bbhi',  ta.volatility.bollinger_hband, ['close']),
    ('bbli',  ta.volatility.bollinger_lband, ['close']),
    ('kcp', ta.volatility.keltner_channel_hband, ['high', 'low', 'close']) ,
    ('kchi', ta.volatility.keltner_channel_hband_indicator ,['high', 'low','close']),
    ('kcli',ta.volatility.keltner_channel_hband, ['high', 'low', 'close']),
    ('macd', ta.trend.macd, ['close']),
    ('macd_diff', ta.trend.macd_diff, ['close']),
    ('adx', ta.trend.adx, ['high','low','close']),
    ('mass_index', ta.trend.mass_index, ['high', 'low']),
    ('dpo', ta.trend.dpo, ['close']),
    ('kst', ta.trend.kst, ['close']),
    ('aroon_up', ta.trend.aroon_up, ['close']),
    ('aroon_down', ta.trend.aroon_down, ['close']),
    ('ppo', ta.momentum.ppo, ['close'])
]

# Adding the indicators using the package ta. The default indicators are the ones selected after have analysed the correlation.
def add_indicators(df, indicators = ['atr', 'bbm', 'bbw', 'bbp', 'bbhi', 'bbli','kcp', 'kchi','kcli', 'macd', 'macd_diff', 'mass_index', 'dpo', 'kst','aroon_up', 'aroon_down','ppo']):
    """This function add the indicators using the package ta"""
    df_with_indicators = pd.DataFrame()
    for ticker in df.ticker.unique():
        df_temp = df[df["ticker"] == ticker].copy()
        indicators_selected = [indicator for indicator in indicators_list if indicator[0] in indicators]
        for name, f, arg_names in indicators_selected:
            wrapper = lambda func, args: func(*args)
            args = [df[arg_name] for arg_name in arg_names]
            df_temp[name] = wrapper(f, args)
        df_temp.fillna(method='bfill', inplace=True)
        df_with_indicators = df_with_indicators.append(df_temp)
    return df_with_indicators


# Option using stockstats and considerin the most standard indicators used for trading
def add_standard_indicators(df, indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']):
    """
    This function return the dataframe with the finantial indicators specify. See https://pypi.org/project/stockstats/ for the documentation. 
    """
    df_with_indicators = pd.DataFrame()
    for ticker in df.ticker.unique():
        df_temp = df[df["ticker"] == ticker].copy()
        df_stocks = StockDataFrame(df_temp.copy())
        # We could change some default values for example 
        # df_stocks.MACD_EMA_LONG = 12 
        for i in indicators:
            df_temp[i] = df_stocks[i]
        df_with_indicators = df_with_indicators.append(df_temp)
    df_with_indicators.fillna(method='bfill', inplace=True)
    return df_with_indicators 

    