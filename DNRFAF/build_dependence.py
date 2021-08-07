from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from datetime import datetime
import os

N_FOLDS = 5
SEED = 2
STEP = 10

def run(path):
    dir_fs = os.path.join(path, 'feature_indexes')
    dir_vectors = os.path.join(path, 'vectors_tsv')
    dir_res_dep = os.path.join(path, 'result_dependences')

    targets = set()
    for file in os.listdir(os.path.join(dir_vectors, 'outer')):
        targets.add(file.split('_')[0])
    targets = sorted(list(targets))

    for target in targets:
        print('{0} Building dependence for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), target))

        if os.path.exists(os.path.join(dir_res_dep, 'dep_' + target + '.txt')):
            os.remove(os.path.join(dir_res_dep, 'dep_' + target + '.txt'))

        with open(os.path.join(dir_res_dep, 'dep_' + target + '.txt'), 'a', encoding='utf-8') as foutput:
            foutput.write("n_features\tbest_param\tfold_1_train\tfold_1_test\tbest_param\tfold_2_train\tfold_2_test\t" +
                          "best_param\tfold_3_train\tfold_3_test\tbest_param\tfold_4_train\tfold_4_test\t" +
                          "best_param\tfold_5_train\tfold_5_test\touter_test\n")

        X_data, y_data = load_svmlight_file(os.path.join(dir_vectors, 'inner', target + '_train1_vectors.txt'))
        num_features = X_data.shape[1]

        for n in range(STEP, num_features + 1, STEP):
            print('{0}  Number of features: {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), n))

            with open(os.path.join(dir_res_dep, 'dep_' + target + '.txt'), 'a', encoding='utf-8') as foutput:
                foutput.write(str(n))

            # Inner
            for i in range(N_FOLDS):
                # Load the dataset
                X_data, y_data = load_svmlight_file(
                    os.path.join(dir_vectors, 'inner', target + '_train' + str(i + 1) + '_vectors.txt'))

                X_train = X_data[:X_data.shape[0] - 1, :]
                y_train = y_data[:len(y_data) - 1]

                with open(os.path.join(dir_fs, 'inner', target + '_train' + str(i + 1) + '_feature_indexes.txt'), 'r',
                          encoding='utf-8') as finput:
                    idx = [int(x) for x in finput.read().split()]

                X_train_fs = X_train[:, idx[:n]]

                p_grid = {"kernel": ('linear',), "C": [0.1], "gamma": [0.0001]}

                # We will use a Support Vector Classifier
                clf = SVC()

                inner_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

                # Non_nested parameter search and scoring
                clf = GridSearchCV(estimator=clf, param_grid=p_grid, cv=inner_cv, verbose=0, scoring="f1_macro")
                clf.fit(X_train_fs, y_train)
                best_params = clf.best_params_

                with open(os.path.join(dir_res_dep, 'dep_' + target + '.txt'), 'a', encoding='utf-8') as foutput:
                    foutput.write('\t{0}\t{1}'.format(str(clf.best_params_), str(clf.best_score_)))

                X_test, y_test = load_svmlight_file(
                    os.path.join(dir_vectors, 'inner', target + '_test' + str(i + 1) + '_vectors.txt'))

                X_test = X_test[:X_test.shape[0] - 1, :]
                y_test = y_test[:len(y_test) - 1]

                X_test_fs = X_test[:, idx[:n]]

                clf = SVC(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma'])
                clf.fit(X_train_fs, y_train)
                y_pred = clf.predict(X_test_fs)

                score = f1_score(y_test, y_pred, average='macro')
                with open(os.path.join(dir_res_dep, 'dep_' + target + '.txt'), 'a', encoding='utf-8') as foutput:
                    foutput.write('\t' + str(score))

            # Outer
            # Load the dataset
            X_train, y_train = load_svmlight_file(os.path.join(dir_vectors, 'outer', target + '_train_vectors.txt'))

            with open(os.path.join(dir_fs, 'outer', target + '_train_feature_indexes.txt'), 'r',
                      encoding='utf-8') as finput:
                idx = [int(x) for x in finput.read().split()]

            X_train_fs = X_train[:, idx[:n]]

            p_grid = {"kernel": ('linear',), "C": [0.1, 1, 10], "gamma": [0.0001]}

            # We will use a Support Vector Classifier
            clf = SVC()

            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

            # Non_nested parameter search and scoring
            clf = GridSearchCV(estimator=clf, param_grid=p_grid, cv=inner_cv, verbose=0, scoring="f1_macro")
            clf.fit(X_train_fs, y_train)
            best_params = clf.best_params_

            X_test, y_test = load_svmlight_file(os.path.join(dir_vectors, 'outer', target + '_test_vectors.txt'))

            X_test = X_test[:X_test.shape[0] - 1, :]
            y_test = y_test[:len(y_test) - 1]

            X_test_fs = X_test[:, idx[:n]]

            clf = SVC(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma'])
            clf.fit(X_train_fs, y_train)
            y_pred = clf.predict(X_test_fs)

            score = f1_score(y_test, y_pred, average='macro')
            with open(os.path.join(dir_res_dep, 'dep_' + target + '.txt'), 'a', encoding='utf-8') as foutput:
                foutput.write('\t' + str(score) + '\n')


if __name__ == '__main__':
    run('')
