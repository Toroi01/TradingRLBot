
from config import config
import pandas as pd
from trade.time_series_validation import TimeSeriesValidation


#df load the preprocessed dataset
print(config.ROOT_DIR)
print(config.DATA_SAVE_DIR)
#pd.read_pickle(config.DATA_SAVE_DIR+"/preprocess_df.pkl")
#env_params

#model_name

#model_params
#self, df, env_params, model_name, model_params
# tsv = TimeSeriesValidation()
# tsv.run()