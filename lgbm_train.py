import os, re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import feat_engineering

RND_SEED = 67213

# Load dataframe
df = pd.read_pickle(os.getcwd() + '/data/df_one.pkl')
print('> LGBM Train : Loaded preprocessed df')

print('> LGBM Train : Beginning feat engineering . . .')
# Feat engineering
df = feat_engineering.create_features(df)
df, cat_col_nums = feat_engineering.encoder(df)
print('> LGBM Train : Finished feat engineering')

# Separate train and validation dfs
train_test_df = df.loc['train'].copy()
val_df = df.loc['test'].copy()

# To numpy
x_train_test = train_test_df.iloc[:,1:-1].values
y_train_test = train_test_df.iloc[:,-1].values

user_ids = train_test_df.iloc[:,0].astype(float).values
x_val = val_df.iloc[:,1:-1].values # Drop dummy labels

# Set log1p of labels
y_train_test = np.log1p(y_train_test)

# Setup CV
groups = y_train_test > 0
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    #random_state=RND_SEED,
)

cv_scores = []
bsts = []

# CV main loop
for i, (train_index, test_index) in enumerate(skf.split(x_train_test, groups)):

    # Get current fold
    x_train = x_train_test[train_index]
    y_train = y_train_test[train_index]
    x_test = x_train_test[test_index]
    y_test = y_train_test[test_index]
    eval_group = user_ids[test_index]

    # Create lgb booster
    bst = lgb.LGBMModel(
        objective='regression',
        num_leaves = 62,
        learning_rate = 0.080,
        n_estimators = 10000,
        min_child_samples= 155,
        subsample=0.98,
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

    # Calculate competition metric below

    # Predict
    y_pred = bst.predict(x_test)
    y_pred = np.clip(y_pred, 0, np.inf)
    y_pred = np.expm1(np.squeeze(y_pred))

    # Build dataframe with predictions and truth per session
    session_df = train_test_df.iloc[test_index, [0, -1]].copy()
    session_df['y_pred'] = y_pred

    # Aggregate pred and truth per user
    y_true_user = session_df.groupby('fullVisitorId')['totals.transactionRevenue'].sum().values
    y_pred_user = session_df.groupby('fullVisitorId')['y_pred'].sum().values

    # Apply log1p to aggregated predictions
    y_pred_user = np.log1p(y_pred_user)
    y_true_user = np.log1p(y_true_user)

    # Comp. metric
    val_loss = np.sqrt(np.mean(np.power(y_pred_user - y_true_user, 2)))

    print('Competition metric : ',val_loss)
    cv_scores.append(val_loss)
    bsts.append(bst)

# Save model committee

# Setup folder name
saved_names = os.listdir(os.getcwd() + '/lgbm_models/')
model_num = 0
if saved_names:
    for name in saved_names:
        num = int(re.search(r'model\d+', name)[0][5:]) + 1
        model_num = num if num>model_num else model_num
folder_name = f'bst_model{model_num:d}_{i+1:d}_avg_cv_{np.mean(cv_scores):.5f}/'

# Create folder to save model and scaler
dir_path = os.getcwd() + '/lgbm_models/' + folder_name
os.mkdir(dir_path)

for i, (bst, cv_score) in enumerate(zip(bsts, cv_scores)):
    bst.booster_.save_model(os.getcwd() + '/lgbm_models/' + folder_name + f'bst{i:d}_cv_fold_{cv_score:.5f}.txt')