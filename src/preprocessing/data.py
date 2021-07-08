from __future__ import division, absolute_import, print_function
import os
import pandas as pd

from src.config import config
from src.dataset.cryptodownloader import CryptoDownloader
from src.preprocessing.preprocessors import FeatureEngineer


def load_dataset(file_name):
    """
    Load csv dataset from path
    :return: (df) pandas dataframe
    """
    data = pd.read_csv(file_name)
    return data


def load_processed_df():
    df_path = os.path.join(config.DATA_SAVE_DIR,config.PREPROCESSED_DF_NAME)
    if not os.path.isfile(df_path):
        os.makedirs(config.DATA_SAVE_DIR,exist_ok=True)

        data_downloader = CryptoDownloader(config.START_DATE, config.END_DATE, config.MULTIPLE_TICKER_8,
                                                config.DATA_SAVE_DIR, config.DATA_GRANULARITY)
        data_downloader.download_data()
        df = data_downloader.load()

        fe = FeatureEngineer(
            use_technical_indicator=True,
            use_turbulence=False,
            user_defined_feature=True,
            use_covariance=True
        )
        df = fe.preprocess_data(df)
        df.to_pickle(df_path)
    else:
        df = pd.read_pickle(df_path)
    return df


def build_features(data):
    features = config.TECHNICAL_INDICATORS_SHORTPERIOD + config.TECHNICAL_INDICATORS_LONGPERIOD + ["open", "close",
                                                                                                   'high', 'low']
    features += [f"{feature}_diff" for feature in features]
    features += [feature for feature in data.columns if feature.startswith("cov_")]

    return features


def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.date >= start) & (df.date < end)]
    return format_for_env(data)


def format_for_env(df):
    df = df.sort_values(["date", "tic"], ignore_index=True)
    df.index = df.date.factorize()[0]
    return df

