import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import hyperopt.hp as hp
from hyperopt import tpe, fmin, STATUS_OK
import utils

# Load dataframe
df_pp = pd.read_pickle(os.getcwd() + '/data/full_df_preproc.pkl')
print('> Loaded full preprocessed dataframe')

# Feat engineering + processing
df = utils.process_df(df_pp)

# Separate train and validation dfs
train_test_df = df.loc['train'].copy()
val_df = df.loc['test'].copy()

print('> Separated train+test / validation sets')

# To numpy
x_train_test = train_test_df.iloc[:,1:-1].values
y_train_test = train_test_df.iloc[:,-1].values
user_ids = train_test_df.iloc[:,0].astype(float).values
x_val = val_df.iloc[:,1:-1].values # Drop dummy labels

# Set log1p of labels
y_train_test = np.log1p(y_train_test)

# Cat cols
ccols = [0, 3, 4, 5, 7, 9, 12, 13, 15, 17, 18, 20, 22, 23, 24, 26, 30]

# Setup CV
kf = KFold(
    n_splits=5,
    shuffle=True,
)

cv_scores = []
bsts = []

# Hyperopt search space
space = {
    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
    'lr': hp.loguniform('lr', np.log(0.009), np.log(0.2)),
    'min_child_samples': hp.quniform('min_child_samples', 20, 600, 5),
    'subsample' : hp.uniform('subsample', 0.9, 1.0),
    'reg_lambda' : hp.uniform('reg_lambda', 0.0, 0.5),
}

def objective(hpp):
    '''Returns compet. cv validation score - avg ensemble of folds'''

    # CV main loop
    for i, (train_index, test_index) in enumerate(kf.split(x_train_test)):

        # Get current fold
        x_train = x_train_test[train_index]
        y_train = y_train_test[train_index]
        x_test = x_train_test[test_index]
        y_test = y_train_test[test_index]

        # Create lgb booster
        bst = lgb.LGBMModel(
            objective='regression',
            num_leaves = int(hpp['num_leaves']),
            learning_rate = hpp['lr'],
            n_estimators = 10000,
            min_child_samples= int(hpp['min_child_samples']),
            subsample=hpp['subsample'],
            reg_lambda=hpp['reg_lambda'],
        )

        bst.fit(
            X=x_train,
            y=y_train,
            eval_set=[(x_test, y_test)],
            eval_metric='rmse',
            early_stopping_rounds=15,
            categorical_feature=ccols,
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

        return {
            'loss' : val_loss,
            'params' : hpp,
            'status' : STATUS_OK,
        }

# Optimize
best = fmin(
    fn = objective,
    space = space,
    algo = tpe.suggest,
    max_evals = 50,
)

best