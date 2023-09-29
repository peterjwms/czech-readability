import sys
from pathlib import Path
from pprint import pprint

import pandas as pd
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier

with open(Path('datasets/balanced_readability.csv'), encoding='UTF8') as ds_file:
    dataset = pd.read_csv(ds_file, keep_default_na=False, na_values=['_'], sep=',', index_col='name')  # avoid 'NA' category being interpreted as missing data  # noqa

array = dataset.values  # two-dimensional numpy array
names = dataset.index

feats = array[:, 1:-2]  # `:` copies all rows, `:-1` slices all but last two columns
feat_columns = list(dataset.columns[1:-2])
print(feat_columns)
groups = array[:, -2] # `:` copies all rows, `-2` selects only second to last column
labels = array[:, -1].astype('int32')  # `:` copies all rows, `-1` selects only last column
seed = 42  # used to make the same 'p

split = model_selection.train_test_split(feats, labels, groups, 
                                         test_size=0.2,
                                         random_state=seed)
feats_train, feats_validation, labels_train, labels_validation, groups_train, groups_validation = split


xgb_estimator = XGBClassifier()
xgb_selector = RFE(xgb_estimator, n_features_to_select=1, step=1)
xgb_selector = xgb_selector.fit(feats_train, labels_train)
# print(xgb_selector.support_)
# print(xgb_selector.ranking_)
xgb_results = list(zip(*sorted(zip(xgb_selector.ranking_, feat_columns))))[1]
# xgb_sorted = list(zip(*xgb_results))[1]
pprint(xgb_results)

lda_estimator = LinearDiscriminantAnalysis()
lda_selector = RFE(lda_estimator, n_features_to_select=1, step=1)
lda_selector = lda_selector.fit(feats_train, labels_train)
# print(lda_selector.support_)
# print(lda_selector.ranking_)
lda_results = list(zip(*sorted(zip(lda_selector.ranking_, feat_columns))))[1]
# pprint(sorted(lda_results))

xgbrf_estimator = XGBRFClassifier()
xgbrf_selector = RFE(xgbrf_estimator, n_features_to_select=1, step=1)
xgbrf_selector = xgbrf_selector.fit(feats_train, labels_train)
# this line sorts the columns according to their rank
xgbrf_results = list(zip(*sorted(zip(xgbrf_selector.ranking_, feat_columns))))[1]

rf_estimator = RandomForestClassifier()
rf_selector = RFE(rf_estimator, n_features_to_select=1, step=1)
rf_selector = rf_selector.fit(feats_train, labels_train)
rf_results = list(zip(*sorted(zip(rf_selector.ranking_, feat_columns))))[1]

all_results = list(zip(feat_columns, lda_selector.ranking_, xgb_selector.ranking_, rf_selector.ranking_, xgbrf_selector.ranking_))
results = pd.DataFrame(all_results, columns=['feat', 'LDA', 'XGB', 'RF', 'XGBRF'])
results.to_csv(Path("results/final_paper_revisions_rfe_results.csv"), index=False)

sorted_results = list(zip(xgb_results, lda_results, xgbrf_results, rf_results))
sorted_df = pd.DataFrame(sorted_results, columns=['XGB', 'LDA', 'XGBRF', 'RF'])
sorted_df.to_csv(Path("results/final_paper_revisions_rfe_sorted_results.csv"), index=False)




# output the list of features in order from rfe for each model
# loop through the cv process - have the full dataframe available at the start of each loop
# then select the next column to add in based on the rankings and train the model again
# pull the next column from the df and put those columns specifically to a numpy array
# import the function to do cv from grouped_fold_models.py and use that, also adjust that to pass in the model we want each time
#  

# feat_columns needs to be sorted first and pulled in from this script
# for i in range(1, len(feat_columns) + 1):
#     curr_feats = dataset[feat_columns[:i]].to_numpy()


#     lda = LinearDiscriminantAnalysis()
#     lda.fit()
    # just track the f1 for this - list of lists of f1s where we have the f1 score for each fold for each feature model count something
    # line charts and box plots for each model with all of the stuff