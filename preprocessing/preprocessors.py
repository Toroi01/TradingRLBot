import numpy as np
import pandas as pd
import ta
from stockstats import StockDataFrame as Sdf

from config import config

indicators_list = [
    ('psar', ta.trend.psar_up_indicator, ['high', 'low', 'close']),
    ('ui', ta.volatility.ulcer_index, ['close']),
    ('atr', ta.volatility.average_true_range, ['high', 'low', 'close']),
    ('bbw', ta.volatility.bollinger_wband, ["close"]),
    ('bbp', ta.volatility.bollinger_pband, ['close']),
    ('bbhi', ta.volatility.bollinger_hband, ['close']),
    ('bbli', ta.volatility.bollinger_lband, ['close']),
    ('kcp', ta.volatility.keltner_channel_hband, ['high', 'low', 'close']),
    ('kchi', ta.volatility.keltner_channel_hband_indicator, ['high', 'low', 'close']),
    ('kcli', ta.volatility.keltner_channel_hband, ['high', 'low', 'close']),
    ('macd', ta.trend.macd, ['close']),
    ('macd_diff', ta.trend.macd_diff, ['close']),
    ('mass_index', ta.trend.mass_index, ['high', 'low']),
    ('dpo', ta.trend.dpo, ['close']),
    ('kst', ta.trend.kst, ['close']),
    ('aroon_up', ta.trend.aroon_up, ['close']),
    ('aroon_down', ta.trend.aroon_down, ['close']),
    ('ppo', ta.momentum.ppo, ['close'])
]


class FeatureEngineer:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            user user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """

    def __init__(
            self,
            use_technical_indicator=True,
            tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
            use_turbulence=False,
            use_covariance=True,
            user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature
        self.use_covariance = use_covariance

    def preprocess_data(self, df):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """

        if self.use_technical_indicator == True:
            # add technical indicators using stockstats
            df = self.add_indicators(df)
            print("Successfully added technical indicators")

        # add turbulence index for multiple stock
        if self.use_turbulence == True:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature == True:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df = df.fillna(method="bfill").fillna(method="ffill")
        df = self.limit_numbers(df)

        if self.use_covariance == True:
            df = self.add_covariance(df)
        return df

    def add_indicators(self, df):
        """This function add the indicators using the package ta"""
        df_with_indicators = pd.DataFrame()
        df = df.sort_values(by=['tic', 'date'])
        indicators = self.tech_indicator_list
        for ticker in df.tic.unique():
            df_temp = df[df["tic"] == ticker].copy()
            indicators_selected = [indicator for indicator in indicators_list if indicator[0] in indicators]
            for name, f, arg_names in indicators_selected:
                wrapper = lambda func, args: func(*args)
                args = [df[arg_name] for arg_name in arg_names]
                df_temp[name] = wrapper(f, args)
            df_temp.fillna(method='bfill', inplace=True)
            df_with_indicators = df_with_indicators.append(df_temp)
        return df_with_indicators

    # Eliminate this when we see that the new definition works
    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=['tic', 'date'])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator['tic'] = unique_ticker[i]
                    temp_indicator['date'] = df[df.tic == unique_ticker[i]]['date'].to_list()
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(indicator_df[['tic', 'date', indicator]], on=['tic', 'date'], how='left')
        df = df.sort_values(by=['date', 'tic'])
        return df

    def add_user_defined_feature(self, data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """

        df = data.copy()
        # stock = Sdf.retype(df.copy())
        unique_ticker = data.tic.unique()

        for column in self.tech_indicator_list + ["open", "close", "high", "low"]:
            for ticker in unique_ticker:
                df.loc[df.tic == ticker, f"{column}_diff"] = df.loc[df.tic == ticker, column].pct_change()
        return df

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
                ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min():].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(filtered_hist_price, axis=0)
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    def limit_numbers(self, df):
        """
        Avoid having extra big and extra small numbers
        :param df:
        :return:
        """

        def to_zero(row):
            if abs(row) < 1e-10:
                return 0
            elif abs(row) > 1e15:
                return 1e15
            else:
                return row

        for column in df.columns:

            if column not in ["date", "tic"]:
                df[column] = df[column].apply(to_zero)
        return df

    def add_covariance(self, df, lookback=4320):
        """This function return the dataframe with the followings modifications:
        - An extra columns with the covariance matrix calculated consider the previous amout of hours as specify in the lookback
        - The dataframe is order for hour and crypto and the index represent the timestamp
        - We have eliminat the first n observations where n=lookback
        """
        df = df.sort_values(['date', 'tic'], ignore_index=True)
        df.index = df.date.factorize()[0]
        cov_list = []
        # default look back is six months
        for i in range(lookback, len(df.index.unique())):
            data_lookback = df.loc[i - lookback:i, :]
            price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
            return_lookback = price_lookback.pct_change().dropna()
            covs = return_lookback.cov().values
            cov_list.append(covs)
        # We add the covariance metrices and we eliminate the first 6 month of training since we can not use them
        df_cov = pd.DataFrame({'date': df.date.unique()[lookback:], 'cov_list': cov_list})
        df = df.merge(df_cov, on='date')
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        # Covariance columns
        cov_columns = []
        for i in range(df.tic.nunique()):
            for j in range(df.tic.nunique()):
                cov_columns.append(f"cov_{i}_{j}")
        df[cov_columns] = df.cov_list.apply(lambda x: x.flatten()).apply(pd.Series)
        selected_cov_columns = []
        for i in range(df.tic.nunique()):
            for j in range(i, df.tic.nunique()):
                selected_cov_columns.append(f"cov_{i}_{j}")

        cov_to_remove = list(set(cov_columns) - set(selected_cov_columns))
        df.drop(cov_to_remove, axis=1, inplace=True)
        df.drop("cov_list", axis=1, inplace=True)
        return df
