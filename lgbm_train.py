import os, re, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import feat_engineering
import ensembling_utils
from sklearn.model_selection import KFold

# Loading, preprocess, feat engineering

# Load dataframe
df = pd.read_pickle(os.getcwd() + '/data/df_one.pkl')
print('> LGBM Train : Loaded preprocessed df')

print('> LGBM Train : Beginning feat engineering . . .')

# Feat engineering
df = feat_engineering.create_features(df)
df, cat_col_nums = feat_engineering.encoder(df)
print('> LGBM Train : Finished feat engineering')





# Wrangling

# Separate train and validation dfs
train_test_df = df.loc['train'].copy()
val_df = df.loc['test'].copy()

# To numpy
x_train_test = train_test_df.iloc[:,1:-1].values
y_train_test = train_test_df.iloc[:,-1].values
x_val = val_df.iloc[:,1:-1].values # Drop dummy labels

# Set log1p of labels
y_train_test = np.log1p(y_train_test)

# Get training CV folds
num_folds = 5
user_folds = ensembling_utils.get_folds(train_test_df, num_folds)





# CV main loop

# Loop collector vars
cv_scores = []
bsts = []
importances = pd.DataFrame()

# Pipeline variables
oof_y_train_test_pred = np.zeros(y_train_test.shape[0])
y_val_preds_sess = np.zeros((x_val.shape[0], num_folds))

for i, (train_index, test_index) in enumerate(user_folds):

    # Get current fold
    x_train, y_train = x_train_test[train_index], y_train_test[train_index]
    x_test, y_test = x_train_test[test_index], y_train_test[test_index]

    # Create lgb booster
    bst = lgb.LGBMModel(
        objective='regression',
        num_leaves = 73,
        learning_rate = 0.072,
        n_estimators = 10000,
        min_child_samples= 155,
        subsample=0.99,
        reg_lambda=0.0,
    )

    bst.fit(
        X=x_train,
        y=y_train,
        eval_set=[(x_test, y_test)],
        eval_metric='rmse',
        early_stopping_rounds=20,
        categorical_feature=cat_col_nums,
        verbose=False,
    )

    # Calculate metrics below

    # Aux visualization
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_test_df.columns[1:-1]
    imp_df['gain'] = bst.booster_.feature_importance(importance_type='gain')
    imp_df['fold'] = i + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)

    # Predict
    y_pred = bst.predict(x_test, num_iteration=bst.best_iteration_)
    y_pred = np.squeeze(np.clip(y_pred, 0, np.inf))

    # Store OF prediction
    oof_y_train_test_pred[test_index] = y_pred

    # Store SUB prediction
    _sub_pred = bst.predict(x_val, num_iteration=bst.best_iteration_)
    y_val_preds_sess[:, i] = np.squeeze(np.clip(_sub_pred, 0, np.inf))

    # Session level loss
    val_loss = np.sqrt(np.mean((y_pred - y_test)**2))

    print('Session-level RMSE : ',val_loss)
    cv_scores.append(val_loss)
    bsts.append(bst)





# Save model committee (3 steps)

# Setup folder name
saved_names = os.listdir(os.getcwd() + '/lgbm_models/')
model_num = 0
if saved_names:
    for name in saved_names:
        num = int(re.search(r'model\d+', name)[0][5:]) + 1
        model_num = num if num>model_num else model_num
folder_name = f'bst_model{model_num:d}_{i+1:d}_avg_cv_{np.mean(cv_scores):.5f}/'

# Create folder to save model
dir_path = os.getcwd() + '/lgbm_models/' + folder_name
os.mkdir(dir_path)

# Save committee
for i, (bst, cv_score) in enumerate(zip(bsts, cv_scores)):
    bst.booster_.save_model(os.getcwd() + '/lgbm_models/' + folder_name + f'bst{i:d}_cv_fold_{cv_score:.5f}.txt')





print('> LGBM Train : Starting user level processing . . .')

# Setup user level

# Put sess-level predictions in original dfs, needed later for aggregation. Leave true label in last pos
train_test_df.insert(
    column='oof_exp_sess_preds',
    loc=len(train_test_df.columns)-1,
    value=np.expm1(oof_y_train_test_pred)
)
val_df.insert(
    column='sess_cv_avg_exp_pred',
    loc=len(val_df.columns)-1,
    value=np.mean(np.expm1(y_val_preds_sess), axis=1)
)

# Aggregate train-test features at user level
train_test_feats_df = train_test_df.iloc[:,:-2].groupby('fullVisitorId').mean() # Keep last 2 label columns out (oof preds and label)
val_feats_df = val_df.iloc[:,:-2].groupby('fullVisitorId').mean()

# Expand session predictions and their stats at user level
pred_col_names = [
    'oof_exp_sess_preds',
    'sess_cv_avg_exp_pred'
]
sess_dfs = [df.loc[:,['fullVisitorId', pred_col_names[i]]] for i,df in enumerate([train_test_df, val_df])]
expanded_dfs = ensembling_utils.expand_predictions(sess_dfs)
train_test_preds_expan_df = expanded_dfs[0]
val_preds_expan_df = expanded_dfs[1]

# Reindex before concat - make sure feats and expanded preds are alligned
train_test_preds_expan_df = train_test_preds_expan_df.reindex(index=train_test_feats_df.index)
val_preds_expan_df = val_preds_expan_df.reindex(index=val_feats_df.index)

# Final concat
train_test_user_df = pd.concat([train_test_feats_df, train_test_preds_expan_df], axis=1)
val_user_df = pd.concat([val_feats_df, val_preds_expan_df], axis=1)

del expanded_dfs, train_test_feats_df, train_test_preds_expan_df, val_feats_df, val_preds_expan_df
gc.collect()

# Fill expanded NAs with zeros
train_test_user_df.fillna(0, inplace=True)
val_user_df.fillna(0, inplace=True)

# Create visitor level target for train_test df
train_test_user_df['user_log_target'] = np.log1p(
    train_test_df.groupby('fullVisitorId')['totals.transactionRevenue'].sum()
)





# To numpy
x_train_test_user = train_test_user_df.iloc[:,:-1].values #No longer 1:-1 since ids were gone in groupby above
y_train_test_user = train_test_user_df.iloc[:,-1].values
x_val_user = val_user_df.values # No dummy labels to drop this time

# Set log1p of labels
y_train_test_user = np.log1p(y_train_test_user)

# Get training CV folds
num_folds = 5
user_folds = KFold(n_splits=num_folds)





print('> LGBM Train : Finished user level processing. Starting user training . . .')

# CV main loop

# Loop collector vars
cv_scores = []
user_bsts = []
y_val_preds_user = np.zeros((x_val_user.shape[0], num_folds))
oof_y_train_test_pred_user = np.zeros(y_train_test_user.shape[0])

for i, (train_index, test_index) in enumerate(user_folds.split(np.arange(y_train_test_user.size))):

    # Get current fold
    x_train, y_train = x_train_test_user[train_index], y_train_test_user[train_index]
    x_test, y_test = x_train_test_user[test_index], y_train_test_user[test_index]

    # Create lgb booster
    bst = lgb.LGBMModel(
        objective='regression',
        num_leaves = 73,
        learning_rate = 0.072,
        n_estimators = 10000,
        min_child_samples= 155,
        subsample=0.99,
        reg_lambda=0.0,
    )

    bst.fit(
        X=x_train,
        y=y_train,
        eval_set=[(x_test, y_test)],
        eval_metric='rmse',
        early_stopping_rounds=20,
        categorical_feature=cat_col_nums,
        verbose=False,
    )

    # Calculate metrics below

    # Predict
    y_pred = bst.predict(x_test, num_iteration=bst.best_iteration_)
    y_pred = np.squeeze(np.clip(y_pred, 0, np.inf))

    # Store OF prediction
    oof_y_train_test_pred_user[test_index] = y_pred

    # Store SUB prediction
    _sub_pred = bst.predict(x_val_user, num_iteration=bst.best_iteration_)
    y_val_preds_user[:, i] = np.squeeze(np.clip(_sub_pred, 0, np.inf))

    # User level loss
    val_loss = np.sqrt(np.mean((y_pred - y_test)**2))

    print('User-level RMSE : ',val_loss)
    cv_scores.append(val_loss)
    user_bsts.append(bst)





# Submission

sub_user = pd.DataFrame(data=np.mean(y_val_preds_user, axis=1), index=val_user_df.index)
mean_cv_str = str(np.mean(cv_scores))

# Save submission file
sub_user.to_csv(
    os.getcwd() + '/submissions/' + f'lgbm_user_{i+1:d}cv_{mean_cv_str}.csv',
    index=True,
    header=True,
    index_label='fullVisitorId',
)