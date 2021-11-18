from sklearn.datasets import load_svmlight_file
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import tree
from datetime import datetime
import pandas as pd
import ast
import os

SEED = 2
N_FOLDS = 5

def cross_val(dir_reports, dir_vectors, clf, feature_indexes, target):
    for i in range(N_FOLDS):
        X_train, y_train = load_svmlight_file(os.path.join(dir_vectors, 'inner', target + '_train' + str(i + 1) + '_vectors.txt'))

        X_train = X_train[:X_train.shape[0] - 1, feature_indexes]
        y_train = y_train[:len(y_train) - 1]

        clf.fit(X_train, y_train)

        X_test, y_test = load_svmlight_file(os.path.join(dir_vectors, 'inner', target + '_test' + str(i + 1) + '_vectors.txt'))

        X_test = X_test[:X_test.shape[0] - 1, feature_indexes]
        y_test = y_test[:len(y_test) - 1]

        y_pred = clf.predict(X_test)

        with open(os.path.join(dir_reports, 'predict_inner.txt'), 'a', encoding='utf-8') as f:
            f.write(' '.join(str(int(x)) for x in y_pred) + '\n')

        with open(os.path.join(dir_reports, 'real_inner.txt'), 'a', encoding='utf-8') as f:
            f.write(' '.join(str(int(x)) for x in y_test) + '\n')

def predict_test_data(dir_reports, dir_vectors, clf, feature_indexes, target):
    X_train, y_train = load_svmlight_file(os.path.join(dir_vectors, 'outer', target + '_train_vectors.txt'))

    X_train = X_train[:X_train.shape[0] - 1, feature_indexes]
    y_train = y_train[:len(y_train) - 1]

    clf.fit(X_train, y_train)

    X_test, y_test = load_svmlight_file(os.path.join(dir_vectors, 'outer', target + '_test_vectors.txt'))

    X_test = X_test[:X_test.shape[0] - 1, feature_indexes]
    y_test = y_test[:len(y_test) - 1]

    y_pred = clf.predict(X_test)

    with open(os.path.join(dir_reports, 'predict_test.txt'), 'a', encoding='utf-8') as f:
        f.write(' '.join(str(int(x)) for x in y_pred) + '\n')

    with open(os.path.join(dir_reports, 'real_test.txt'), 'a', encoding='utf-8') as f:
        f.write(' '.join(str(int(x)) for x in y_test) + '\n')

def predict(isinner, dir_vectors, dir_reports, feature_indexes, target):
    if isinner:
        if (os.path.isfile(os.path.join(dir_reports + 'predict_inner.txt'))):
            os.remove(os.path.join(dir_reports + 'predict_inner.txt'))
        if (os.path.isfile(os.path.join(dir_reports + 'real_inner.txt'))):
            os.remove(os.path.join(dir_reports + 'real_inner.txt'))
    else:
        if (os.path.isfile(os.path.join(dir_reports + 'predict_test.txt'))):
            os.remove(os.path.join(dir_reports + 'predict_test.txt'))
        if (os.path.isfile(os.path.join(dir_reports + 'real_test.txt'))):
            os.remove(os.path.join(dir_reports + 'real_test.txt'))

    df = pd.read_csv(os.path.join(dir_reports, 'gridsearchcv.txt'), sep='\t')
    params_svm = df.loc[df['Classifier'] == 'SVM'].iloc[0]['Parameters']
    params_knn = df.loc[df['Classifier'] == 'kNN'].iloc[0]['Parameters']
    params_ab = df.loc[df['Classifier'] == 'AB'].iloc[0]['Parameters']
    params_dt = df.loc[df['Classifier'] == 'DT'].iloc[0]['Parameters']
    params_rf = df.loc[df['Classifier'] == 'RF'].iloc[0]['Parameters']

    print('{0}  Getting predictions for SVM'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    params = params_svm.split()
    params_dict = {params[0].split('=')[0]: params[0].split('=')[1], params[1].split('=')[0]: params[1].split('=')[1],
                   params[2].split('=')[0]: params[2].split('=')[1]}
    clf = svm.SVC(kernel=params_dict['kernel'], C=float(params_dict['C']), gamma=float(params_dict['gamma']))
    if isinner:
        cross_val(dir_reports, dir_vectors, clf, feature_indexes, target)
    else:
        predict_test_data(dir_reports, dir_vectors, clf, feature_indexes, target)

    print('{0}  Getting predictions for kNN'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    clf = KNeighborsClassifier(n_neighbors=int(params_knn.split('=')[1]))
    if isinner:
        cross_val(dir_reports, dir_vectors, clf, feature_indexes, target)
    else:
        predict_test_data(dir_reports, dir_vectors, clf, feature_indexes, target)

    print('{0}  Getting predictions for AdaBoost'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    clf = AdaBoostClassifier(n_estimators=int(params_ab.split('=')[1]))
    if isinner:
        cross_val(dir_reports, dir_vectors, clf, feature_indexes, target)
    else:
        predict_test_data(dir_reports, dir_vectors, clf, feature_indexes, target)

    print('{0}  Getting predictions for DT'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    clf = tree.DecisionTreeClassifier(max_depth=int(params_dt.split('=')[1]))
    if isinner:
        cross_val(dir_reports, dir_vectors, clf, feature_indexes, target)
    else:
        predict_test_data(dir_reports, dir_vectors, clf, feature_indexes, target)

    print('{0}  Getting predictions for RF'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    clf = RandomForestClassifier(n_estimators=int(params_rf.split('=')[1]))
    if isinner:
        cross_val(dir_reports, dir_vectors, clf, feature_indexes, target)
    else:
        predict_test_data(dir_reports, dir_vectors, clf, feature_indexes, target)

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
    df = pd.read_csv(os.path.join(dir_vectors, 'outer', 'counter_features.tsv'), sep='\t')
    for feature_type in require_feature_types:
        if feature_type != 'N-grams':
            start_index = df.loc[df['Target'] == target].iloc[0][FEATURE_TYPES[FEATURE_TYPES.index(feature_type) - 1]]
        else:
            start_index = 0
        end_index = df.loc[df['Target'] == target].iloc[0][feature_type]
        feature_indexes.extend([i for i in range(start_index, end_index)])

    return feature_indexes

def run(path):
    dir_vectors = os.path.join(path, 'vectors_tsv')
    dir_feature_indexes = os.path.join(path, 'feature_indexes')

    if len(path) == 0:
        file_params = os.path.join('..', 'DECCV_params.tsv')
    else:
        file_params = 'DECCV_params.tsv'

    targets = set()
    for file in os.listdir(os.path.join(dir_vectors, 'outer')):
        if file.endswith('.txt'):
            targets.add(file.split('_')[0])
    targets = sorted(list(targets))

    for target in targets:
        print('{0} Getting classifier predictions for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), target))

        dir_reports = os.path.join(path, 'reports', 'reports_' + target)

        feature_indexes = select_features(dir_feature_indexes, dir_vectors, file_params, target)

        predict(True, dir_vectors, dir_reports, feature_indexes, target)
        predict(False, dir_vectors, dir_reports, feature_indexes, target)

if __name__ == '__main__':
    run('')
