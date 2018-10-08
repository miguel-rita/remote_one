from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import tqdm





def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding of each visitors fold"""

    #TODO - Rethink groupkfold
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = KFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids





def expand_predictions(id_pred_dfs):
    '''
    For each df supplied 'id_pred_dfs' expand its predictions into new dfs when passing from session to user level.
    This new df will also contain per-user stats eg. sum of all preds (in exp form)

    :param id_pred_df: List of 2 col dataframes, with first column = user id, second column = prediction (in exp1m form)
    :return: List of new user-level data frames, with index being unique user id and values predictions
        for that user. Note that since we have different num. of predictions per user some entries will
        assume the value np.nan (user didn't have that many sessions)
        Note that for all new df len(df.columns) = number of stats + max number of session in all supplied dfs
    '''

    final_dfs = []

    # Get list of preds per session for all dfs, as well as global max num of sessions per user

    print('> Ensembling Tools : Expanding predictions . . .')

    pred_list_by_id_all_dfs = []
    for id_pred_df in id_pred_dfs:
        pred_list_by_id_all_dfs.append(id_pred_df.groupby(id_pred_df.columns[0])[id_pred_df.columns[-1]].apply(list))

    print('> Ensembling Tools : Done expanding predictions.')

    max_num_preds_per_sess = np.max([[np.max([len(l) for l in pred_list_by_id.values])] for pred_list_by_id in pred_list_by_id_all_dfs])

    for id_pred_df, pred_list_by_id in zip(id_pred_dfs, pred_list_by_id_all_dfs):

        # Create empty placeholder for final df pred values
        final_values = np.zeros((pred_list_by_id.shape[0], max_num_preds_per_sess))
        final_values.fill(np.nan)

        # Assign predictions
        for i,l in enumerate(pred_list_by_id.values):
            final_values[i, :len(l)] = l

        # Compute stats
        stats_names = ['log_mean', 'log_median', 'log_sum']
        log_mean = np.log1p(np.nanmean(final_values, axis=1))
        log_median = np.log1p(np.nanmedian(final_values, axis=1))
        log_sum = np.log1p(np.nansum(final_values, axis=1))

        # Concat stats to expanded preds
        final_values = np.vstack([final_values.T, log_mean, log_median, log_sum]).T

        # Create final df
        final_df = pd.DataFrame(
            data=final_values,
            index=pred_list_by_id.index,
            columns=['pred_'+str(i+1) for i in range(max_num_preds_per_sess)] + stats_names,
        )

        final_dfs.append(final_df)

    return final_dfs

# df = pd.DataFrame(
#     {
#         'id' :   [3,4,5,5,5],
#         'pred' : [1,2,3,8,1],
#     }
# )
#
#
# df2 = pd.DataFrame(
#     {
#         'id' :   [2,3,3,4,1,1,0,0],
#         'pred' : [1,2,3,8,1,9,2,5],
#     }
# )
#
# print(expand_predictions([df,df2]))