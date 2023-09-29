from itertools import zip_longest
import re
import sys
from collections import Counter
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRFClassifier


# declare this as a function at the top, everything else in main
def cross_validation(training, groups, feats, labels):
    for i, fold in enumerate(training):
        # now for each tuple, pull out all the data in the test set that matches those groups as the test set, then set the rest of the data as training
        # print(i)
        # val_split = training[i]
        # fold_groups.append(val_split)
        # # print(val_split)
        # # print(val_split)
        # train_split = [training[j] for j, _ in enumerate(training) if j != i] # just make this a set of the values?

        # # print(train_split)

        # # need to pull out the data for each group in the set - need the feats, labels for each one
        # group_set = set()
        # for tup in train_split:
        #     group_set.update(tup)

        # group_bools = [row in group_set for row in groups]

        # feats_train = feats[group_bools]
        # labels_train = labels[group_bools]

        # val_bools = [row in val_split for row in groups]
        # feats_val = feats[val_bools]
        # labels_val = labels[val_bools]
        # fold_true_labels.append(labels_val)

        feats_train, labels_train, feats_val, labels_val = train_validation_split(training, i, fold_groups, fold_true_labels, groups, feats, labels)
        
        # TODO: use the seed for each model
        lda = LinearDiscriminantAnalysis()
        train_model(lda, lda_predictions, lda_scores, lda_matrices, feats_train, labels_train, feats_val, labels_val)

        rf = RandomForestClassifier(random_state=seed)
        train_model(rf, rf_predictions, rf_scores, rf_matrices, feats_train, labels_train, feats_val, labels_val)

        xgb = XGBClassifier(random_state=seed)
        train_model(xgb, xgb_predictions, xgb_scores, xgb_matrices, feats_train, labels_train, feats_val, labels_val)

        xgbrf = XGBRFClassifier(random_state=seed)
        train_model(xgbrf, xgbrf_predictions, xgbrf_scores, xgbrf_matrices, feats_train, labels_train, feats_val, labels_val)


def train_validation_split(training, i, fold_groups: list, fold_true_labels: list, groups, feats, labels):
    val_split = training[i]
    if val_split not in fold_groups:
        fold_groups.append(val_split)
    # print(val_split)
    train_split = [training[j] for j, _ in enumerate(training) if j != i] # just make this a set of the values?

    # print(train_split)

    # need to pull out the data for each group in the set - need the feats, labels for each one
    group_set = set()
    for tup in train_split:
        group_set.update(tup)

    group_bools = [row in group_set for row in groups]

    feats_train = feats[group_bools]
    labels_train = labels[group_bools]

    val_bools = [row in val_split for row in groups]
    feats_val = feats[val_bools]
    labels_val = labels[val_bools]
    fold_true_labels.append(labels_val)

    return feats_train, labels_train, feats_val, labels_val
    

def train_model(model, predictions: list, scores: list, matrices: list, feats_train, labels_train, feats_val, labels_val):
    """
    Train a model

    Parameters:

    """
    model.fit(feats_train, labels_train)
    prediction = model.predict(feats_val)
    score = precision_recall_fscore_support(labels_val, prediction, zero_division=np.nan, average="weighted")
    accuracy = accuracy_score(labels_val, prediction)
    # print(accuracy)
    score = list(score)
    score.append(accuracy)
    print(score)  # TODO: finish adding accuracy here and in the grouped_results df - need to get it in the df for each model at the last index
    predictions.append(prediction)
    scores.append(score)
    matrices.append(confusion_matrix(labels_val, prediction, labels=[0,1,2]))


def create_folds(dataset: pd.DataFrame):
    groupA = {int(row['group']) for i, row in dataset.iterrows() if row['label'] == 0}
    groupB = {int(row['group']) for i, row in dataset.iterrows() if row['label'] == 1}
    groupC = {int(row['group']) for i, row in dataset.iterrows() if row['label'] == 2}
    
    folds = zip_longest(groupA, groupB, groupC, fillvalue=None)
    fold_list = list(folds)

    groupA_labels = {group: 0 for group in groupA}
    groupB_labels = {group: 1 for group in groupB}
    groupC_labels = {group: 2 for group in groupC}
    group_labels = groupA_labels | groupB_labels | groupC_labels

    test = fold_list[0]
    training = fold_list[1:len(fold_list)]
    return test, training, group_labels
    

def get_title(name: str) -> str:
    parts = name.split('_')
    id_len = len(parts[0])
    if id_len == 4:
        title = parts[2] + '_' + parts[3]
        return title
    elif id_len == 5:
        return 'CTDC'


#### TEMPLATE #####
if __name__ == "__main__":


    with open(Path('datasets/balanced_readability.csv'), encoding='UTF8') as ds_file:
        dataset = pd.read_csv(ds_file, keep_default_na=False, na_values=['_'], sep=',', index_col='name')  # avoid 'NA' category being interpreted as missing data  # noqa
    print(type(dataset), file=sys.stderr)
        
    description = (dataset.describe())
    description.to_csv(Path("datasets/ds_desc.csv"))
    dataset.hist(figsize=(20, 20), sharey = True) # TODO: look at taking out outliers
    fig = plt.gcf()
    fig.savefig(Path('datasets/hists.png'))
    # exit()
    # remove any row where the group is between 31 and 40
    ds_no_c = dataset[dataset['group'] < 31]
    print(ds_no_c.head(20))
    print(ds_no_c.shape)

    # rearrange the group column so it's at the end, just before 'label'
    # group_col = dataset.pop('group')
    # dataset.insert(-2, group_col.name, group_col)


    # Summarize the data
    print('"Shape" of dataset:', dataset.shape,
        f'({dataset.shape[0]} instances of {dataset.shape[1]} attributes)',
        end='\n\n', file=sys.stderr)
    print('"head" of data:\n', dataset.head(20), end='\n\n', file=sys.stderr)
    print('Description of data:\n:', dataset.describe(), end='\n\n',
        file=sys.stderr)
    print('Class distribution:\n', dataset.groupby('label').size(), end='\n\n',
        file=sys.stderr)

    # Visualize the data
    # print('Drawing boxplot...', file=sys.stderr)
    # grid_size = 0
    # while grid_size ** 2 < len(dataset):
    #     grid_size += 1
    # dataset.plot(kind='box', subplots=True, layout=(grid_size, grid_size),
    #              sharex=False, sharey=False)
    # fig = plt.gcf()  # get current figure
    # fig.savefig('boxplots.png')

    # # histograms
    # print('Drawing histograms...', file=sys.stderr)
    # dataset.hist()
    # fig = plt.gcf()
    # fig.savefig('histograms.png')

    # # scatter plot matrix
    # print('Drawing scatterplot matrix...', file=sys.stderr)
    # scatter_matrix(dataset)
    # fig = plt.gcf()
    # fig.savefig('scatter_matrix.png')

    print('Splitting training/development set and validation set...',
        file=sys.stderr)
    # Split-out validation dataset
    # TODO: need to change the dataset here back to the full set
    array = dataset.to_numpy()  # two-dimensional numpy array
    names = dataset.index
    # print(names)
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing  # noqa: E501

    # TODO: need to access the correct columns and then run the model
    # names = array[:, 0]
    # TODO: parse through to get just the book title from the file name
    feats = array[:, 1:-2]  # `:` copies all rows, `:-1` slices all but last two columns
    groups = array[:, -2] # `:` copies all rows, `-2` selects only second to last column
    # TODO: bootstrapping goes here (removing various features)
    labels = array[:, -1].astype('int32')  # `:` copies all rows, `-1` selects only last column
    print('\tfull original data ([:5]) and their respective labels:',
        file=sys.stderr)
    print(feats[:5], labels[:5], sep='\n\n', end='\n\n\n', file=sys.stderr)
    seed = 42  # used to make the same 'pseudo-random' choices in each run

    names_groups = dict()
    # print(names)
    for i, group in enumerate(groups):
        # need to 
        print(get_title(names[i]))
        names_groups[group] = get_title(names[i])
    
    print(names_groups)

    # split into train/test splits here w/ each book being individually held as the test split against the train split of everything else 
    # maybe have a chunk of A/B/C for each test split - when out of B books, just do A/C split
    # randomly select a chunk of data for the C portion of the split 
    # use sklearn.GroupKFold or sklearn.StratifiedGroupKFold or sklearn.LeaveOneGroupOut

    # split = model_selection.train_test_split(feats, labels, groups, 
    #                                          test_size=0.2,
    #                                          random_state=seed)
    # feats_train, feats_validation, labels_train, labels_validation, groups_train, groups_validation = split
    # print('\ttraining data:\n', feats_train[:5],
    #       '\ttraining labels:\n', labels_train[:5],
    #       '\tvalidation data:\n', feats_validation[:5],
    #       '\tvalidation labels:\n', labels_validation[:5], sep='\n\n')


    # hold out one set of A,B,C groups as true test set
    # use zip_longest to get tuples of group numbers, then do folds w/ those tuples holding out each of those tuples as the validation set 

    # need to first split the dataset into the groups - just put in the group numbers first? and then within the other loop access the feats, labels, groups and hold out all the data part of that group
    # print("groups")
    # groupA = {int(row['group']) for i, row in dataset.iterrows() if row['label'] == 0}
    # groupB = {int(row['group']) for i, row in dataset.iterrows() if row['label'] == 1}
    # groupC = {int(row['group']) for i, row in dataset.iterrows() if row['label'] == 2}
    # # print(groupA)
    # # print(groupB)
    # # print(groupC)

    # folds = zip_longest(groupA, groupB, groupC, fillvalue=None)
    # fold_list = list(folds)

    # groupA_labels = {group:0 for group in groupA}
    # groupB_labels = {group:1 for group in groupB}
    # groupC_labels = {group:2 for group in groupC}
    # group_labels = groupA_labels | groupB_labels | groupC_labels
    # # print(group_labels)
    # # print(group_labels.values())

    # test = fold_list[0]
    # # print(test) # validation set that isn't touched by the models, only used for prediction
    # training = fold_list[1:len(fold_list)]
    # print(training) # make this a set?
    # train_set = set()
    # for tup in training:
    #     for x in tup:
    #         train_set.add(x)

    # print(train_set)

    test, training, group_labels = create_folds(dataset)

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

    cross_validation(training, groups, feats, labels)

    manual_results = pd.DataFrame()
    manual_results['Groups'] = list(fold_groups)
    manual_results['LDA'] = lda_scores
    # print(lda_scores)
    manual_results['RF'] = rf_scores
    manual_results['XGB'] = xgb_scores
    manual_results['XGBRF'] = xgbrf_scores
    manual_results.to_csv(Path('results/final_paper_revisions_results.csv'), encoding='UTF8', sep=',', index=False)
    print(manual_results)
    grouped_results = []

    # for i, _ in enumerate(manual_results['Groups']):
        # order: group, support(not in rn), LDA(prec, rec, f1, acc), RF(all four), XGB(all four), XGBRF(all four) 
        # for adding accuracy, would just need to add that as a new column 

        # print(type(manual_results['Groups']))


        # use this version when the scoring is unweighted
        # split_scores = list(zip_longest(manual_results['Groups'][i], manual_results['LDA'][i][3], 
        #                             manual_results['LDA'][i][0], manual_results['LDA'][i][1], manual_results['LDA'][i][2], 
        #                             manual_results['RF'][i][0], manual_results['RF'][i][1], manual_results['RF'][i][2], 
        #                             manual_results['XGB'][i][0], manual_results['XGB'][i][1], manual_results['XGB'][i][2], 
        #                             manual_results['XGBRF'][i][0], manual_results['XGBRF'][i][1], manual_results['XGBRF'][i][2]))
        # grouped_results.append(split_scores[0])
        # grouped_results.append(split_scores[1])
        # grouped_results.append(split_scores[2])

        # use this version when scoring is weighted
    # split_scores = list(zip_longest(manual_results['Groups'], 
    #                             manual_results['LDA'][0], manual_results['LDA'][1], manual_results['LDA'][2], manual_results['LDA'][4], 
    #                             manual_results['RF'][0], manual_results['RF'][1], manual_results['RF'][2], manual_results['RF'][4], 
    #                             manual_results['XGB'][0], manual_results['XGB'][1], manual_results['XGB'][2], manual_results['XGB'][4], 
    #                             manual_results['XGBRF'][0], manual_results['XGBRF'][1], manual_results['XGBRF'][2], manual_results['XGBRF'][4]))
    # print(split_scores)
    # grouped_results = split_scores
        # grouped_results.append(split_scores[0])
        # grouped_results.append(split_scores[1])
        # grouped_results.append(split_scores[2])

    lda_cols = pd.DataFrame(manual_results['LDA'].to_list(), columns=['LDA prec', 'LDA rec', 'LDA f1', 'LDA Support', 'LDA acc'])
    rf_cols = pd.DataFrame(manual_results['RF'].to_list(), columns=['RF prec', 'RF rec', 'RF f1', 'RF Support', 'RF acc'])
    xgb_cols = pd.DataFrame(manual_results['XGB'].to_list(), columns=['XGB prec', 'XGB rec', 'XGB f1', 'XGB Support', 'XGB acc',])
    xgbrf_cols = pd.DataFrame(manual_results['XGBRF'].to_list(), columns=['XGBRF prec', 'XGBRF rec', 'XGBRF f1', 'XGBRF Support', 'XGBRF acc'])
    
    grouped_results = manual_results['Groups']
    df_by_group = pd.DataFrame(grouped_results, columns=['Group1', 'Group2', 'Group3'])
    df_by_group = pd.concat([df_by_group, lda_cols], axis=1)
    df_by_group = pd.concat([df_by_group, rf_cols], axis=1)
    df_by_group = pd.concat([df_by_group, xgb_cols], axis=1)
    df_by_group = pd.concat([df_by_group, xgbrf_cols], axis=1)
    print(df_by_group)


    # df_by_group = pd.DataFrame(grouped_results, columns=['Group', 'LDA prec', 'LDA rec', 'LDA f1', 'LDA acc', 'RF prec', 'RF rec', 'RF f1', 'RF acc', 
                                                        #  'XGB prec', 'XGB rec', 'XGB f1', 'XGB acc', 'XGBRF prec', 'XGBRF rec', 'XGBRF f1', 'XGBRF acc'])
    # df_by_group.insert(1, 'Label', group_labels)
    df_by_group.insert(1, 'Label', df_by_group['Group1'].apply(lambda group: group_labels.get(group)))
    df_by_group.insert(2, 'Name', df_by_group['Group1'].apply(lambda group: names_groups.get(group)))
    df_by_group.to_csv(Path('results/final_paper_revisions_grouped_results.csv'), encoding='UTF8', sep=',', index=False)


    # TODO: adjust the way the data looks as it comes out - add names - DONE
    # TODO: binary classification ignoring all the c-text to see if those are far off - DONE
    # TODO: LDA and XGB for feature evaluation - DONE
    # TODO: more features - classic readability measure especially - DONE
    # TODO: transform all the features using the sklearn transformers or other functions to give them to the neural model
    # activation functions
    manual_predictions = pd.DataFrame()
    manual_predictions['Groups'] = list(fold_groups)
    manual_predictions['True'] = fold_true_labels
    manual_predictions['LDA'] = lda_predictions
    manual_predictions['RF'] = rf_predictions
    manual_predictions['XGB'] = xgb_predictions
    manual_predictions['XGBRF'] = xgbrf_predictions
    manual_predictions.to_csv(Path('results/final_paper_revisions_predictions.csv'), encoding='UTF8', sep=',', index=False)
    exit()

    flattened_true = [x for f in manual_predictions['True'] for x in f]

    font = {'family' : 'normal',
    'weight' : 'bold',
    'size'   : 16}
    plt.rc('font', **font)

    cm_labels=['A', 'B', 'C']

    # LDA confusion matrix
    lda_matrix = sum(lda_matrices)
    print('LDA')
    print(lda_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=lda_matrix, display_labels=cm_labels)
    disp.plot(cmap='Greys')
    # plt.show()
    # plt.title("LDA Confusion Matrix")
    plt.savefig(Path('matrices/rev_lda_matrix.png'))
    # cr = dict()
    cr = (classification_report(flattened_true, [x for f in manual_predictions['LDA'] for x in f], labels=cm_labels))
    print(cr)
    # cr_df = pd.DataFrame(eval(cr)).transpose()
    # # cr_df.transpose()
    # cr_df.to_csv(Path("class_reports/lda_cr.csv"))



    rf_matrix = sum(rf_matrices)
    print('RF')
    print(rf_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=rf_matrix,
                                   display_labels=cm_labels)
    disp.plot(cmap="Greys")
    # plt.show()
    # plt.title("RF Confusion Matrix")
    plt.savefig(Path('matrices/rev_rf_matrix.png'))
    cr = classification_report(flattened_true, [x for f in manual_predictions['RF'] for x in f], labels=cm_labels)
    print(cr)
    # cr_df = pd.DataFrame(cr).transpose()
    # cr_df.to_csv(Path("class_reports/rf_cr.csv"))



    xgb_matrix = sum(xgb_matrices)
    print('XGB')
    print(xgb_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=xgb_matrix,
                                   display_labels=cm_labels)
    disp.plot(cmap="Greys")
    # plt.show()
    # plt.title("XGB Confusion Matrix")
    plt.savefig(Path('matrices/rev_xgb_matrix.png'))
    cr = (classification_report(flattened_true, [x for f in manual_predictions['XGB'] for x in f], labels=cm_labels))
    print(cr)
    # cr_df = pd.DataFrame(cr).transpose()
    # cr_df.to_csv(Path("class_reports/xgb_cr.csv"))


    xgbrf_matrix = sum(xgbrf_matrices)
    print('XGBRF')
    print(xgbrf_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=xgbrf_matrix,
                                   display_labels=cm_labels)
    disp.plot(cmap="Greys")
    # plt.show()
    # plt.title("XGBRF Confusion Matrix")
    plt.savefig(Path('matrices/rev_xgbrf_matrix.png'))
    cr = (classification_report(flattened_true, [x for f in manual_predictions['XGBRF'] for x in f], labels=cm_labels))
    print(cr)
    # cr_df = pd.DataFrame(cr).transpose()
    # cr_df.to_csv(Path("class_reports/xgbrf_cr.csv"))


    


    # TODO: train models on all of the data (except test) and then test on test data
    exit()

    print('Initializing models...', file=sys.stderr)
    models = [('LR', LogisticRegression(solver='lbfgs', multi_class='auto')),
            ('LDA', LinearDiscriminantAnalysis()),
            ('KNN', KNeighborsClassifier()),
            ('CART', DecisionTreeClassifier()),
            ('NB', GaussianNB()),
            ('SVM', SVC(gamma='scale')),
            ('RF', RandomForestClassifier()),
            ('XGB', XGBClassifier()),
            ('XGBRF', XGBRFClassifier())]
    print('Training and testing each model using 10-fold cross-validation...',
        file=sys.stderr)
    # evaluate each model in turn
    results = []  # track results for boxplot
    for name, model in models:
        # https://chrisjmccormick.files.wordpress.com/2013/07/10_fold_cv.png

        # TODO: make the num_splits here dynamic based on the number of groups
    #     gkf = model_selection.GroupKFold(n_splits=31)
        sgkf = model_selection.StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    #     for train, test in sgkf.split(feats_train, labels_train, groups=groups_train):
    #         pass # TODO:
        cv_results = model_selection.cross_val_score(model, feats_train,
                                                    labels_train, groups=groups_train, cv=sgkf,
                                                    scoring='f1_macro') # also want to get precision_macro, recall_macro, (f1_macro, accuracy)
        results.append(cv_results)
        msg = f'{name}: {cv_results.mean()} ({cv_results.std()})'
        print(msg, file=sys.stderr)

    print('\n\nDrawing algorithm comparison boxplots...', file=sys.stderr)
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels([name for name, model in models])
    fig = plt.gcf()
    fig.savefig('compare_algorithms.png')


    rf = RandomForestClassifier()
    rf.fit(feats_train, labels_train)
    print("feats: ", len(rf.feature_importances_))
    print("dataset: ", len(dataset))
    pprint(sorted(zip(dataset, rf.feature_importances_), key=lambda x: x[1], reverse=True))
    # make predictions on the validation - get the model ourselves, not just the score
    # loop that trains, makes predictions, scores

    xgb = XGBClassifier()
    xgb.fit(feats_train, labels_train)
    print("XGB: ")
    pprint(sorted(zip(dataset, xgb.feature_importances_), key=lambda x: x[1], reverse=True))

    xgbrf = XGBRFClassifier()
    xgbrf.fit(feats_train, labels_train)
    print("XGBRF: ")
    pprint(sorted(zip(dataset, xgbrf.feature_importances_), key=lambda x: x[1], reverse=True))
