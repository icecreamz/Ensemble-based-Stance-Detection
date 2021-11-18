from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from datetime import datetime
import pandas as pd
import ast
import os

SEED = 2
N_FOLDS = 5

def nested_cv(dir_reports, clf_name, clf, p_grid, feature_indexes, dir_vectors, file_vectors):
    # Load the dataset
    X_data, y_data = load_svmlight_file(os.path.join(dir_vectors, file_vectors))

    X_data = X_data[:X_data.shape[0] - 1, feature_indexes]
    y_data = y_data[:len(y_data) - 1]

    inner_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=clf, param_grid=p_grid, cv=inner_cv, verbose=10, scoring="f1_macro")
    clf.fit(X_data, y_data)

    # Nested CV with parameter optimization
    nested_score = cross_val_score(clf, X=X_data, y=y_data, cv=outer_cv, verbose=10, scoring="f1_macro")

    with open(os.path.join(dir_reports, 'gridsearchcv.txt'), 'a', encoding='utf-8') as foutput:
        foutput.write(clf_name + '\t')
        foutput.write(' '.join('{0}={1}'.format(key, val) for key, val in clf.best_params_.items()) + '\n')

def select_features(dir_feature_indexes, dir_vectors, file_params, target):
    # Feature types: N-grams, Dependency, Linguistic, Target name, Stance-indicative,
    # Stylistic, Sentiment, Word2vec
    FEATURE_TYPES = ['N-grams', 'Dependency', 'Linguistic', 'Target name', 'Stance-indicative',
                     'Stylistic', 'Sentiment', 'Word2vec']

    if os.path.isfile(file_params):
        df = pd.read_csv(file_params, sep='\t')
        require_feature_types = ast.literal_eval(df.loc[df['Target'] == target].iloc[0]['Best feature types'])
    else:
        require_feature_types = FEATURE_TYPES

    feature_indexes = []

    # Load feature indexes from file
    if 'N-grams' not in require_feature_types:
        with open(os.path.join(dir_feature_indexes, target + '_feature_indexes.txt'), 'r') as f:
            feature_indexes.extend([int(i) for i in f.read().split()])

    # Add require feature indexes
    df = pd.read_csv(os.path.join(dir_vectors, 'counter_features.tsv'), sep='\t')
    for feature_type in require_feature_types:
        if feature_type != 'N-grams':
            start_index = df.loc[df['Target'] == target].iloc[0][FEATURE_TYPES[FEATURE_TYPES.index(feature_type) - 1]]
        else:
            start_index = 0
        end_index = df.loc[df['Target'] == target].iloc[0][feature_type]
        feature_indexes.extend([i for i in range(start_index, end_index)])

    return feature_indexes

def run(path):
    dir_vectors = os.path.join(path, 'vectors_tsv', 'outer')
    dir_feature_indexes = os.path.join(path, 'feature_indexes')

    if len(path) == 0:
        file_params = os.path.join('..', 'DECCV_params.tsv')
    else:
        file_params = 'DECCV_params.tsv'

    targets = set()
    for file in os.listdir(dir_vectors):
        if file.endswith('.txt'):
            targets.add(file.split('_')[0])
    targets = sorted(list(targets))

    for target in targets:
        print('{0} Search optimal parameters of classifiers for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), target))

        dir_reports = os.path.join(path, 'reports', 'reports_' + target)
        file_vectors = target + '_train_vectors.txt'

        if not os.path.isdir(dir_reports):
            os.mkdir(dir_reports)

        feature_indexes = select_features(dir_feature_indexes, dir_vectors, file_params, target)

        if (os.path.isfile(os.path.join(dir_reports, 'gridsearchcv.txt'))):
            os.remove(os.path.join(dir_reports, 'gridsearchcv.txt'))
        with open(os.path.join(dir_reports, 'gridsearchcv.txt'), 'a', encoding='utf-8') as foutput:
            foutput.write('Classifier\tParameters\n')

        print('{0}  Search hyperparameters for SVM'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

        # Set up possible values of parameters to optimize over
        p_grid = {"kernel":('linear',), "C": [0.1], "gamma": [0.0001]}

        # We will use a Support Vector Classifier
        clf = SVC()

        nested_cv(dir_reports, 'SVM', clf, p_grid, feature_indexes, dir_vectors, file_vectors)

        print('{0}  Search hyperparameters for kNN'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

        # Set up possible values of parameters to optimize over
        p_grid = {"n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

        # We will use a k-Nearest Neighbors
        clf = KNeighborsClassifier()

        nested_cv(dir_reports, 'kNN', clf, p_grid, feature_indexes, dir_vectors, file_vectors)

        print('{0}  Search hyperparameters for AB'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

        # Set up possible values of parameters to optimize over
        p_grid = {"n_estimators": [50, 100, 150]}

        # We will use a AdaBoost
        clf = AdaBoostClassifier()

        nested_cv(dir_reports, 'AB', clf, p_grid, feature_indexes, dir_vectors, file_vectors)

        print('{0}  Search hyperparameters for DT'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

        # Set up possible values of parameters to optimize over
        p_grid = {"max_depth": [1, 5, 10, 15, 20]}

        # We will use a Decision Tree
        clf = tree.DecisionTreeClassifier()

        nested_cv(dir_reports, 'DT', clf, p_grid, feature_indexes, dir_vectors, file_vectors)

        print('{0}  Search hyperparameters for RF'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

        # Set up possible values of parameters to optimize over
        p_grid = {"n_estimators": [50, 100, 150]}

        # We will use a Random Forest
        clf = RandomForestClassifier()

        nested_cv(dir_reports, 'RF', clf, p_grid, feature_indexes, dir_vectors, file_vectors)


if __name__ == '__main__':
    run('')
