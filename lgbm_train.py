import os, re
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import feat_engineering
import ensembling_utils

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

    print('User-level RMSE : ',val_loss)
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





# Visualization of importances

if False:
    # TODO - Examine zero log gain feats

    importances['gain_log'] = np.log1p(importances['gain'])
    mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
    importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])
    plt.figure(figsize=(8, 12))
    sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False))
