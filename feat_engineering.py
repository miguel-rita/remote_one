import utils

def create_features(df_orig):
    '''
    Module to generate new features

    :param df (pd df): Original df
    :return: New dataframe with the added features
    '''

    df = df_orig.copy()

    # Hits/pv ratio
    s = df['totals.hits'] / df['totals.pageviews']
    df.insert(
        df.shape[1] - 1, # Before last pos
        column='ratio_hits_pv',
        value=s.astype(float),
    )

    return df

def encoder(df_orig, params=None):
    '''
    Module to encode categorical feats

    :param df (pd df): Original df
    :param params (dict, default=See above): Controls encoding scheme used : TODO
    :return: (new dataframe with categorical feats encoded according to 'params', cat feat col indexes)
    '''

    df = df_orig.copy()

    # Default encoding settings
    if not params:
        high_card_limit = 50000
        high_card_feats = []
        cat_encode_feats = []
    # Custom encoding settings
    else:
        high_card_limit = params['high_card_limit']
        high_card_feats = params['high_card_feats']
        cat_encode_feats = params['cat_encode_feats']

    cat_feat_indexes = []

    # Encode high cardinality feats to numeric
    for i,c in enumerate(df.columns):

        if str(df[c].dtype) == 'category':
            df[c] = df[c].cat.codes

            # TODO : Refactor for legibility

            if high_card_feats: # If high card feats were specified by hand
                if c in high_card_feats:
                    df[c] = df[c].astype(float) # Encode as numeric
                else:
                    cat_feat_indexes.append(i - 1) # Or keep as categorical, direct encoding

            else: # Else if a threshold for high card was specified
                if df[c].nunique() > high_card_limit:
                    df[c] = df[c].astype(float)
                else:
                    cat_feat_indexes.append(i - 1)

    return df, cat_feat_indexes