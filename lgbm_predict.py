import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import feat_engineering

print('> LGBM Predict : Loading validation data . . .')

# Load validation data
df = pd.read_pickle(os.getcwd() + '/data/df_one.pkl')

# Feat engineering
df = feat_engineering.create_features(df)
df, cat_col_nums = feat_engineering.encoder(df)
val_df = df.loc['test'].copy()
x_val = val_df.iloc[:,1:-1].values # Drop dummy labels

print('> LGBM Predict : Done loading data. Loading LGBM booster ensemble . . .')

# Load booster ensemble
boost_folder_name = 'bst_model39_5_avg_cv_1.60090'
mean_cv_str = boost_folder_name[-7:]
boost_dir = '/lgbm_models/' + boost_folder_name + '/'
bsts = []
dir_path = os.getcwd() + boost_dir
for bst_name in os.listdir(dir_path):
    bsts.append(lgb.Booster(model_file=dir_path + bst_name))

print('> LGBM Predict : Done loading LGBM booster ensemble. Predicting . . .')

# Predict for bst ensemble
y_preds = np.zeros((x_val.shape[0], len(bsts)))
for i, bst in enumerate(bsts):
    print('> LGBM Predict : Predicting for booster number ',i,'...')
    y_pred = np.squeeze(bst.predict(x_val))
    y_pred = np.clip(y_pred, 0, np.inf)
    y_preds[:, i] = y_pred
y_pred_ensemble = np.mean(y_preds, axis=1)
y_pred_ensemble[y_pred_ensemble<-1] = 0.0
y_pred_ensemble = np.expm1(y_pred_ensemble)

# Build dataframe with predictions per session
session_df = val_df.iloc[:,[0,-1]].copy()
session_df.loc[:, 'totals.transactionRevenue'] = y_pred_ensemble

# Aggregate per user
sub_series = session_df.groupby('fullVisitorId')['totals.transactionRevenue'].sum()
# Apply log1p to aggregated predictions
sub_series = np.log1p(sub_series)
sub_series = sub_series.rename('PredictedLogRevenue')

print('> LGBM Predict : Done predicting. Saving submission . . .')

# Save submission file
sub_series.to_csv(
    os.getcwd() + '/submissions/' + f'lgbm_{i+1:d}cv_{mean_cv_str}.csv',
    index=True,
    header=True,
    index_label='fullVisitorId',
)