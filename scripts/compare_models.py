import re
import sys
from collections import Counter
from pathlib import Path
from pprint import pprint

import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRFClassifier


#### TEMPLATE #####

with open(Path('datasets/balanced_feats_rand_groups.csv'), encoding='UTF8') as ds_file:
    dataset = pd.read_csv(ds_file, keep_default_na=False, na_values=['_'], sep=',', index_col='name')  # avoid 'NA' category being interpreted as missing data  # noqa
print(type(dataset), file=sys.stderr)
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
array = dataset.values  # two-dimensional numpy array
# https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing  # noqa: E501
feats = array[:, :-2]  # `:` copies all rows, `1:` slices all but first column
# TODO: bootstrapping goes here (removing various features)
labels = array[:, -1]  # `:` copies all rows, `` selects only first column
print('\tfull original data ([:5]) and their respective labels:',
      file=sys.stderr)
print(feats[:5], labels[:5], sep='\n\n', end='\n\n\n', file=sys.stderr)
seed = 42  # used to make the same 'pseudo-random' choices in each run

# TODO: split into train/test splits here w/ each book being individually held as the test split against the train split of everything else 
# maybe have a chunk of A/B/C for each test split - when out of B books, just do A/C split
# randomly select a chunk of data for the C portion of the split 
# use sklearn.GroupKFold or sklearn.StratifiedGroupKFold or sklearn.LeaveOneGroupOut





split = model_selection.train_test_split(feats, labels,
                                         test_size=0.2,
                                         random_state=seed)
feats_train, feats_validation, labels_train, labels_validation = split
# print('\ttraining data:\n', feats_train[:5],
#       '\ttraining labels:\n', labels_train[:5],
#       '\tvalidation data:\n', feats_validation[:5],
#       '\tvalidation labels:\n', labels_validation[:5], sep='\n\n')


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
    kfold = model_selection.KFold(n_splits=10, shuffle=True,
                                  random_state=seed)
    cv_results = model_selection.cross_val_score(model, feats_train,
                                                 labels_train, cv=kfold,
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


xgb = XGBClassifier()
xgb.fit(feats_train, labels_train)
print("XGB: ")
pprint(sorted(zip(dataset, xgb.feature_importances_), key=lambda x: x[1], reverse=True))

xgbrf = XGBRFClassifier()
xgbrf.fit(feats_train, labels_train)
print("XGBRF: ")
pprint(sorted(zip(dataset, xgbrf.feature_importances_), key=lambda x: x[1], reverse=True))


# TODO: next steps 
# set up the validation to be going book by book
# slice up the C text by genre/label 