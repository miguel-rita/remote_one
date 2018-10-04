import os
import pandas as pd
from pandas.io.json import json_normalize, json

def load_df(csv_path, nrows=None, to_csv=False, csv_name=None):
    JSON_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']

    print('Loading .csv file ...')

    df = pd.read_csv(
        csv_path,
        converters={
            column : json.loads for column in JSON_cols
        },
        dtype={'fullVisitorId' : str},
        nrows=nrows,
    )

    for col in JSON_cols:
        col_as_df = json_normalize(df[col])
        col_as_df.columns = [f"{col}.{subcol}" for subcol in col_as_df.columns]
        df = df.drop(col, axis=1).merge(col_as_df, right_index=True, left_index=True)

    print(f'Done loading and normalizing csv file')

    if to_csv:

        print('Saving flattened csv to disk ...')

        if csv_name is None:
            raise ValueError('Must provide .csv filename if saving flattened csv to disk')
        df.to_csv(os.getcwd() + '/data/' + csv_name, index_label=False)

        print('... Done saving.')

load_df('data/train.csv', to_csv=True, csv_name='train-flat.csv')
load_df('data/test.csv', to_csv=True, csv_name='test-flat.csv')
