import sys
from pathlib import Path
from pprint import pprint
from matplotlib import pyplot as plt

from numpy import nanmean
import pandas as pd
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier

from grouped_fold_models import create_folds, cross_validation, train_validation_split, train_model

def get_average_f1(scores: list):
    all_f1s = []
    for score in scores: # unpack 'score' as iterating over it
        for f1 in score[2]:
            print(f1)
            all_f1s.append(f1) # this should be getting just the f1
    
    avg = nanmean(all_f1s)
    print(avg)
    return avg

def get_average_weighted_f1(scores: list):
    all_f1s = []
    for score in scores:
        all_f1s.append(score[2])

    avg = nanmean(all_f1s)
    return avg


with open(Path('results/final_paper_revisions_rfe_results.csv'), encoding='UTF8') as ds_file:
    sorted_feats = pd.read_csv(ds_file, keep_default_na=False, na_values=['_'], sep=',')

xgb_columns = sorted_feats['XGB']
lda_columns = sorted_feats['LDA']
xgbrf_columns = sorted_feats['XGBRF']
rf_columns = sorted_feats['RF']
pprint(xgb_columns)

with open(Path('datasets/balanced_readability.csv'), encoding='UTF8') as ds_file:
    dataset = pd.read_csv(ds_file, keep_default_na=False, na_values=['_'], sep=',', index_col='name')



# output the list of features in order from rfe for each model - DONE
# loop through the cv process - have the full dataframe available at the start of each loop
# then select the next column to add in based on the rankings and train the model again
# pull the next column from the df and put those columns specifically to a numpy array
# import the function to do cv from grouped_fold_models.py and use that, also adjust that to pass in the model we want each time
#  

# TODO: we want the test-train split done here, then all the training data is what's used for the cv
# function that does the group stuff and pulls out the first fold as test split

test, training, group_labels = create_folds(dataset)

lda_f1 = []
rf_f1 = []
xgb_f1 = []
xgbrf_f1 = []

seed = 42

# feat_columns needs to be sorted first and pulled in from this script - DONE
for i in range(1, len(xgb_columns) + 1):
    groups = dataset.to_numpy()[:, -2] # `:` copies all rows, `-2` selects only second to last column
    labels = dataset.to_numpy()[:, -1].astype('int32')
    xgb_feats = dataset.iloc[:, xgb_columns[:i]]  # these were giving a key error, but hopefully work with .iloc
    lda_feats = dataset.iloc[:, lda_columns[:i]]  # had .to_numpy() at end of each, don't remember why, seems better w/o
    rf_feats = dataset.iloc[:, rf_columns[:i]]
    xgbrf_feats = dataset.iloc[:, xgbrf_columns[:i]]

    
    fold_true_labels = []
    fold_groups = []
    lda_scores = []
    lda_predictions = []
    lda_matrices = []
    rf_scores = []
    rf_predictions = []
    rf_matrices = []
    xgb_scores = []
    xgb_predictions = []
    xgb_matrices = []
    xgbrf_scores = []
    xgbrf_predictions = []
    xgbrf_matrices = []

    # TODO: make another function in the other file to do the train-validation split - DONE

    # split the function so that the outer loop is here, and the splitting of feats happens in another function
    for j, fold in enumerate(training):
        # this needs to be different because we're pulling a different set of features for each model each time based on the rfe importance
        # feats_train, labels_train, feats_val, labels_val = train_validation_split(training, i, fold_groups, fold_true_labels, groups)

        lda = LinearDiscriminantAnalysis()
        lda_feats_train, labels_train, feats_val, labels_val = train_validation_split(training, j, fold_groups, fold_true_labels, groups, lda_feats, labels)
        train_model(lda, lda_predictions, lda_scores, lda_matrices, lda_feats_train, labels_train, feats_val, labels_val)

        rf = RandomForestClassifier(random_state=seed)
        rf_feats_train, labels_train, feats_val, labels_val = train_validation_split(training, j, fold_groups, fold_true_labels, groups, rf_feats, labels)
        train_model(rf, rf_predictions, rf_scores, rf_matrices, rf_feats_train, labels_train, feats_val, labels_val)

        xgb = XGBClassifier(random_state=seed)
        xgb_feats_train, labels_train, feats_val, labels_val = train_validation_split(training, j, fold_groups, fold_true_labels, groups, xgb_feats, labels)
        train_model(xgb, xgb_predictions, xgb_scores, xgb_matrices, xgb_feats_train, labels_train, feats_val, labels_val)

        xgbrf = XGBRFClassifier(random_state=seed)
        xgbrf_feats_train, labels_train, feats_val, labels_val = train_validation_split(training, j, fold_groups, fold_true_labels, groups, xgbrf_feats, labels)
        train_model(xgbrf, xgbrf_predictions, xgbrf_scores, xgbrf_matrices, xgbrf_feats_train, labels_train, feats_val, labels_val)

    
    # pprint(fold_groups)
    # pprint(lda_scores)
    # pprint(rf_scores)
    # pprint(xgb_scores)
    # pprint(xgbrf_scores)
    # get_average_f1(lda_scores)
    # exit()

    lda_f1.append(get_average_weighted_f1(lda_scores))
    rf_f1.append(get_average_weighted_f1(rf_scores))
    xgb_f1.append(get_average_weighted_f1(xgb_scores))
    xgbrf_f1.append(get_average_weighted_f1(xgbrf_scores))


    # results = pd.DataFrame()
    # results['Groups'] = list(fold_groups)
    # results['LDA'] = lda_scores
    # results['RF'] = rf_scores
    # results['XGB'] = xgb_scores
    # results['XGBRF'] = xgbrf_scores
    # results.to_csv(Path(f'feat_imp_results/feat_imps_results{i}.csv'), encoding='UTF8', sep=',', index=False)
    print(i)

    # TODO: need to get macro-averages for F1 scores for each model and store each one in a list for each loop
    # then output those lists so we can see the curve and change over time
averages = pd.DataFrame()
# averages['Groups'] = list(fold_groups)
averages['LDA'] = lda_f1
averages['RF'] = rf_f1
averages['XGB'] = xgb_f1
averages['XGBRF'] = xgbrf_f1
averages.to_csv(Path(f'feat_imp_results/final_paper_revisions_feat_imps_averages.csv'), encoding='UTF8', sep=',', index=True)
    
# TODO: confusion matrix and classification reports here in the same way
#       focusing mostly on just the best model 


    # train-test split, then call the other function for the folds and cross-validation
    # need to adjust function to take name of model, training data, labels, folds?
        # cross_validation(training, groups, xgb_feats, labels)

    # this should be using just the training data, so after we've first split the dataframe
    
    # just track the f1 for this - list of lists of f1s where we have the f1 score for each fold for each feature model count something
    # line charts and box plots for each model with all of the stuff

manual_predictions = pd.DataFrame()
manual_predictions['Groups'] = list(fold_groups)
manual_predictions['True'] = fold_true_labels
manual_predictions['LDA'] = lda_predictions
manual_predictions['RF'] = rf_predictions
manual_predictions['XGB'] = xgb_predictions
manual_predictions['XGBRF'] = xgbrf_predictions
# manual_predictions.to_csv(Path('results/feat_imp_predictions.csv'), encoding='UTF8', sep=',', index=False)
exit()

flattened_true = [x for f in manual_predictions['True'] for x in f]

lda_matrix = sum(lda_matrices)
print('LDA')
print(lda_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=lda_matrix,
                                display_labels=[0, 1, 2])
disp.plot()
# plt.show()
plt.title("LDA Confusion Matrix")
plt.savefig(Path('matrices/feats_lda_matrix.png'))
print(classification_report(flattened_true, [x for f in manual_predictions['LDA'] for x in f], labels=[0, 1, 2]))


rf_matrix = sum(rf_matrices)
print('RF')
print(rf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=rf_matrix,
                                display_labels=[0, 1, 2])
disp.plot()
# plt.show()
plt.title("RF Confusion Matrix")
plt.savefig(Path('matrices/feats_rf_matrix.png'))
print(classification_report(flattened_true, [x for f in manual_predictions['RF'] for x in f], labels=[0, 1, 2]))


xgb_matrix = sum(xgb_matrices)
print('XGB')
print(xgb_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=xgb_matrix,
                                display_labels=[0, 1, 2])
disp.plot()
# plt.show()
plt.title("XGB Confusion Matrix")
plt.savefig(Path('matrices/feats_xgb_matrix.png'))
print(classification_report(flattened_true, [x for f in manual_predictions['XGB'] for x in f], labels=[0, 1, 2]))


xgbrf_matrix = sum(xgbrf_matrices)
print('XGBRF')
print(xgbrf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=xgbrf_matrix,
                                display_labels=[0, 1, 2])
disp.plot()
# plt.show()
plt.title("XGBRF Confusion Matrix")
plt.savefig(Path('matrices/feats_xgbrf_matrix.png'))
print(classification_report(flattened_true, [x for f in manual_predictions['XGBRF'] for x in f], labels=[0, 1, 2]))
