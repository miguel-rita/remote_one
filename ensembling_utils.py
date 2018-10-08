from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import pickle, tqdm





def get_folds(fullVisitorIdCol=None, n_splits=5, save_to_pkl=False):
    '''
    Returns two fold lists, at session and user level

    Both lists are alligned ie. eg. the sessions in fold 1 are the ones corresponding to users in fold 1

    :param fullVisitorIdCol: A pd.Series of fullVisitorIds, sorted
    :param n_splits: Num of splits for CV
    :param save_to_pkl: If True will save both fold lists to two pickle files : sess_folds.pkl, user_folds.pkl
    :return: The two aforementioned lists of folds - sess_folds, user_folds
    '''

    # Get session values
    sessions = fullVisitorIdCol.values

    # Define groups array ie. one user per group
    groups = get_np_groups(sessions)

    # GroupKFold on sessions, each group one user
    sess_folds_it = GroupKFold(n_splits=n_splits)
    user_folds = []
    sess_folds = []

    for train, test in sess_folds_it.split(X=sessions, groups=groups):

        # By definition, since each group is a user
        sess_folds.append([train, test])

        user_folds.append(
            [
                np.unique(groups[train]).astype(int),
                np.unique(groups[test]).astype(int),
            ]
        )

    if save_to_pkl:
        pickle.dump(sess_folds, open('sess_folds.pkl', 'wb'))
        pickle.dump(user_folds, open('user_folds.pkl', 'wb'))

    return sess_folds, user_folds





def get_np_groups(arr):
    '''
    Get an array of equal len to arr containing group numbers in ascending order

    Assumes arr to be sorted

    :param arr: Input array
    :return: Output arr of same lenght containing groups
    '''

    groups = np.zeros(arr.size)
    prev_sess = np.nan
    group_num = -1
    for i, sess in enumerate(arr):
        if sess != prev_sess:
            group_num += 1
            prev_sess = sess
        groups[i] = group_num

    return groups



def expand_np_array(arr, groups, max_cols=30):
    '''
    Expand 1D arr values into 2D arr with max_cols. Fills with

    :param arr: Sorted array to expand
    :param groups: Groups to perform expansion by
    :param max_cols: Max num of sessions to expand
    :return: Expanded arr
    '''

    group_nums, group_starts = np.unique(groups, return_index=True)
    num_gs = group_starts.size

    expan_arr = np.zeros((num_gs, max_cols))
    expan_arr.fill(np.nan)

    for i in range(num_gs):
        gs = group_starts[i]
        ge = group_starts[i+1] if i < num_gs - 1 else arr.size

        num_expan = ge - gs
        if num_expan > max_cols:
            ge = gs + max_cols

        expan_arr[i, :ge-gs] = arr[gs:ge]


    return expan_arr





def expand_predictions(sorted_preds_df, max_num_sessions=30, reindex=None):
    '''
    Expanded sorted_preds into a 2D numpy array, with max_num_sessions cols even if no user has max_num_sessions
    NaN insert where there was no sess

    :param sorted_preds: pd dataframe with 2 cols : 1st is the sorted fullVisitorId and 2nd the preds
    :param max_num_sessions: Expand up to max_num_sessions
    :param reindex: Final index to maintain coherence for future concats
    :return: Expanded preds in df form, with num_lines = num unique users and num_cols = max_num_sessions
    '''

    print('> Ensembling Tools : Expanding predictions . . .')


    pred_values = sorted_preds_df.iloc[:,-1].values
    ids_values = sorted_preds_df['fullVisitorId']

    expanded_arr = expand_np_array(
        arr=pred_values,
        groups=get_np_groups(ids_values),
        max_cols=max_num_sessions,
    )
    num_missing_cols = max_num_sessions - expanded_arr.shape[1] - 2
    if num_missing_cols > 0: # Means that no user reached max_num_sessions, must complete the array
        extra_cols = np.zeros(expanded_arr.shape[0], num_missing_cols)
        extra_cols.fill(np.nan)
        expanded_arr = np.hstack([expanded_arr, extra_cols])

    # Compute stats
    stats_names = ['log_mean', 'log_median', 'log_sum', 'sum_log']
    log_mean = np.log1p(np.nanmean(expanded_arr, axis=1))
    log_median = np.log1p(np.nanmedian(expanded_arr, axis=1))
    log_sum = np.log1p(np.nansum(expanded_arr, axis=1))
    sum_log = np.nansum(np.log1p(expanded_arr), axis=1)


    # Concat stats to expanded preds
    expanded_arr = np.vstack([expanded_arr.T, log_mean, log_median, log_sum, sum_log]).T

    # Create final df
    final_df = pd.DataFrame(
        data=expanded_arr,
        index=reindex,
        columns=['pred_'+str(i+1) for i in range(max_num_sessions)] + stats_names,
    )

    print('> Ensembling Tools : Done expanding predictions.')

    return final_df

# ar = expand_np_array(np.array([50,0,50,1,1,1,1,2,2,3,3,3,5,10,10]), np.array([0,0,0,1,1,1,1,2,2,3,3,3,4,5,5]), max_cols=3)
# print('BOF')

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

# fid = pd.Series([str(x) for x in np.sort([0,0,0,1,1,2,3,3,3,4,5,6])])
#
# sess, usr = get_folds(fullVisitorIdCol=fid, n_splits=3)
# print('Check!')