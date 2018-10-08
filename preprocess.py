import pandas as pd

num_rows = 10000

print(f'> Preprocessing : Loading raw csv files . . .')
train_df = pd.read_csv('data/train-flat.csv', dtype={'fullVisitorId' : str}, nrows=num_rows, low_memory=False)
test_df = pd.read_csv('data/test-flat.csv', dtype={'fullVisitorId' : str}, nrows=num_rows, low_memory=False)
print(f'> Preprocessing : Done loading raw csv files')

df = pd.concat([train_df, test_df], keys=['train', 'test'], sort=False)
cols_drop = []

for col in df.columns:

    nu = df[col].nunique()
    na = df[col].isna().sum()

    # Unique drops
    if nu <= 1 and na == 0:
        cols_drop.append(col)
        print(f'> Preprocessing : Dropped col named {col} due to being unique')

# Create date features
vst_datetime = pd.to_datetime(df['visitStartTime'], unit='s')
df['fhour'] = vst_datetime.dt.hour + vst_datetime.dt.minute / 60
df['weekday'] = vst_datetime.dt.weekday

print(f'> Preprocessing : Created datetime features')

# Manual exception drops - reason in comments
cols_drop.extend([
    'date',  # Same information as visitStartTime yet less accurate
    'sessionId',  # Almost unique, extreme cardinality
    'visitId',  # Almost unique, extreme cardinality
    'trafficSource.campaignCode', # Almost unique, extreme cardinality
])

# Drop cols marked as such
df = df.drop(cols_drop, axis=1)
for c in cols_drop:
    print(f'> Preprocessing : Dropped col named {c}')

print(f'> Preprocessing : Started filling NAs . . .')
# Fill NAs
other_cat_code = 'other_cat'
na_dict = {
    'trafficSource.isTrueDirect': other_cat_code,
    'trafficSource.adwordsClickInfo.isVideoAd': other_cat_code,
    'totals.bounces': 0,
    'totals.newVisits': 0,
    'totals.transactionRevenue': 0,
    'trafficSource.adContent': other_cat_code,
    'trafficSource.adwordsClickInfo.adNetworkType': other_cat_code,
    'trafficSource.adwordsClickInfo.gclId': other_cat_code,
    'trafficSource.adwordsClickInfo.slot': other_cat_code,
    'trafficSource.keyword': other_cat_code,
    'trafficSource.referralPath': other_cat_code,
}
df = df.fillna(na_dict)

# Special fill NAs - mean without leakage
mean_feats = ['trafficSource.adwordsClickInfo.page', 'totals.pageviews']
for s in ['train', 'test']:
    for feat in mean_feats:
        mean = df.loc['train', feat].mean()  # Train mean - no leakage
        df.loc[s, feat] = df.loc[s, feat].fillna(
            mean).values  # .values critical since index out of fillna is not multiindex

# Sanity check - no NA left
assert df.isna().sum().sum() == 0

print(f'> Preprocessing : Done filling NAs')
print(f'> Preprocessing : Making last adjustments . . .')

# Manual corrections - individual explanations below
# If session has only one pageview it must be a bounce
df.loc[(df['totals.bounces'] == 0.0) & (df['totals.pageviews'] == 1.0), 'totals.bounces'] = 1

# Set feature types - categorical feats as category, numeric and target as float64, unique_id as object
unique_id = [
    'fullVisitorId',
]

num_feats = [
    'visitNumber',
    'visitStartTime',
    'totals.hits',
    'totals.pageviews',
    'trafficSource.adwordsClickInfo.page',
    'fhour',
]

cat_feats = [
    'channelGrouping',
    'device.browser',
    'device.deviceCategory',
    'device.isMobile',
    'device.operatingSystem',
    'geoNetwork.city',
    'geoNetwork.continent',
    'geoNetwork.country',
    'geoNetwork.metro',
    'geoNetwork.networkDomain',
    'geoNetwork.region',
    'geoNetwork.subContinent',
    'totals.bounces',
    'totals.newVisits',
    'trafficSource.adContent',
    'trafficSource.adwordsClickInfo.adNetworkType',
    'trafficSource.adwordsClickInfo.gclId',
    'trafficSource.adwordsClickInfo.isVideoAd',
    'trafficSource.adwordsClickInfo.slot',
    'trafficSource.campaign',
    'trafficSource.isTrueDirect',
    'trafficSource.keyword',
    'trafficSource.medium',
    'trafficSource.referralPath',
    'trafficSource.source',
    'weekday',
]

label = [
    'totals.transactionRevenue',
]

df[cat_feats] = df[cat_feats].astype('category')
df[num_feats + label] = df[num_feats + label].astype(float)

# Move label to end, unique id to beggining
cols = list(df.columns)
cols.append(cols.pop(cols.index('totals.transactionRevenue')))
cols.insert(0, cols.pop(cols.index('fullVisitorId')))
df = df[cols]

df.to_pickle('data/df_one_r.pkl')

print(f'> Preprocessing : DONE')