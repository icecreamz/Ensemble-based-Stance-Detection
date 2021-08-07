from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
import pandas as pd
import itertools
from sklearn.metrics import f1_score
from datetime import datetime
import os

N_SPLITS = 5
SEED = 2

def run(path):
    dir_inner_fs = os.path.join(path, 'feature_indexes', 'inner')
    dir_outer_fs = os.path.join(path, 'feature_indexes', 'outer')
    dir_result = os.path.join(path, 'result_ensemble_fs')
    dir_vectors_inner = os.path.join(path, 'vectors_tsv', 'inner')
    if len(path) == 0:
        file_num_features = os.path.join('..', 'DNRFAF', 'num_relevant_features', 'num_relevant_features.txt')
    else:
        file_num_features = os.path.join('DNRFAF', 'num_relevant_features', 'num_relevant_features.txt')
    dir_result_feature_indexes = os.path.join(dir_result, 'feature_indexes_uni')
    dir_relevant_feature_indexes = os.path.join(dir_result, 'relevant_feature_indexes')

    with open(file_num_features, 'r', encoding='utf-8') as f:
        df = pd.read_csv(file_num_features, sep='\t')
        targets = df['Target']
        N_OPT = df['Number of relevant features']

    best_comb_int = [0 for i in range(len(targets))]
    best_comb_uni = [0 for i in range(len(targets))]
    best_n_int = [0 for i in range(len(targets))]
    best_n_uni = [0 for i in range(len(targets))]
    best_score_int = [0 for i in range(len(targets))]
    best_score_uni = [0 for i in range(len(targets))]

    def eval_ens(i, idx, filename_train, filename_test):
        # Load the dataset
        X_data, y_data = load_svmlight_file(filename_train)

        X_train = X_data[:X_data.shape[0] - 1, :]
        y_train = y_data[:len(y_data) - 1]

        X_train_fs = X_train[:, idx]

        # Set up possible values of parameters to optimize over
        p_grid = {"kernel": ('linear',), "C": [1], "gamma": [0.0001]}

        # We will use a Support Vector Classifier
        clf = SVC()

        inner_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=clf, param_grid=p_grid, cv=inner_cv, verbose=0, scoring="f1_macro")
        clf.fit(X_train_fs, y_train)
        best_params = clf.best_params_

        X_test, y_test = load_svmlight_file(filename_test)

        X_test = X_test[:X_test.shape[0] - 1, :]
        y_test = y_test[:len(y_test) - 1]

        X_test_fs = X_test[:, idx]

        clf = SVC(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma'])
        clf.fit(X_train_fs, y_train)
        y_pred = clf.predict(X_test_fs)

        score = f1_score(y_test, y_pred, average='macro')

        return score

    for i in range(len(targets)):
        print('{0} Determining the relevant features set for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), targets[i]))

        idx = []
        idx_opt = []

        f_output_result = open(os.path.join(dir_result, targets[i] + '_ensemble.txt'), 'w')
        f_output_result.write('Comb\tN_int\tF1_val_int\tN_uni\tF1_val_uni\n')

        f_output_int = open(os.path.join(dir_result, 'feature_indexes_int', targets[i] + '_feature_indexes.txt'), 'w')
        f_output_int.write('Comb\tIndexes\n')

        f_output_uni = open(os.path.join(dir_result, 'feature_indexes_uni', targets[i] + '_feature_indexes.txt'), 'w')
        f_output_uni.write('Comb\tIndexes\n')

        with open(os.path.join(dir_outer_fs, targets[i] + '_train_feature_indexes.txt'), 'r') as f_input:
            idx_outer = [int(x) for x in f_input.read().split()]
        with open(os.path.join(dir_result, 'feature_indexes_outer', targets[i] + '_feature_indexes.txt'), 'w') as f_output:
            f_output.write(' '.join(str(x) for x in idx_outer[:N_OPT[i]]))

        for j in range(N_SPLITS):
            with open(os.path.join(dir_inner_fs, targets[i] + '_test' + str(j + 1) + '_feature_indexes.txt'), 'r') as f_input:
                idx.append([int(x) for x in f_input.read().split()])

        for j in range(N_SPLITS):
            idx_opt.append(set(idx[j][:N_OPT[i]]))

        for num in range(1, N_SPLITS + 1):
            for comb in itertools.combinations(''.join(str(x) for x in range(N_SPLITS)), num):
                print('{0}  Processing comb {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), comb))

                set_idx_int = idx_opt[int(comb[0])]
                set_idx_uni = idx_opt[int(comb[0])]
                for k in range(1, len(comb)):
                    set_idx_int = set_idx_int.intersection(idx_opt[int(comb[k])])
                    set_idx_uni = set_idx_uni.union(idx_opt[int(comb[k])])
                f_output_int.write(''.join(str(int(x) + 1) for x in comb) + '\t' + ' '.join(str(x) for x in set_idx_int) + '\n')
                f_output_uni.write(''.join(str(int(x) + 1) for x in comb) + '\t' + ' '.join(str(x) for x in set_idx_uni) + '\n')
                score_int_val = eval_ens(i, list(set_idx_int), os.path.join(dir_vectors_inner, targets[i] + '_train' + str(N_SPLITS + 1) + '_vectors.txt'),
                                         os.path.join(dir_vectors_inner, targets[i] + '_test' + str(N_SPLITS + 1) + '_vectors.txt'))
                score_uni_val = eval_ens(i, list(set_idx_uni), os.path.join(dir_vectors_inner, targets[i] + '_train' + str(N_SPLITS + 1) +'_vectors.txt'),
                                         os.path.join(dir_vectors_inner, targets[i] + '_test' + str(N_SPLITS + 1) + '_vectors.txt'))
                if score_int_val > best_score_int[i]:
                    best_comb_int[i] = comb
                    best_n_int[i] = len(set_idx_int)
                    best_score_int[i] = score_int_val
                if score_uni_val > best_score_uni[i]:
                    best_comb_uni[i] = comb
                    best_n_uni[i] = len(set_idx_uni)
                    best_score_uni[i] = score_uni_val
                f_output_result.write(''.join(str(int(x) + 1) for x in comb) + '\t' +
                                      str(len(set_idx_int)) + '\t' + str(score_int_val) + '\t' +
                                      str(len(set_idx_uni)) + '\t' + str(score_uni_val) + '\t' + '\n')

        for num in range(2, N_SPLITS):
            set_idx_int = []
            for comb in itertools.combinations(''.join(str(x) for x in range(N_SPLITS)), num):
                set_idx_int.append(idx_opt[int(comb[0])])
                for k in range(1, len(comb)):
                    set_idx_int[len(set_idx_int) - 1] = set_idx_int[len(set_idx_int) - 1].intersection(idx_opt[int(comb[k])])
            set_idx_uni = set_idx_int[0]
            for j in range(1, len(set_idx_int)):
                set_idx_uni = set_idx_uni.union(set_idx_int[j])
            score_uni_val = eval_ens(i, list(set_idx_uni), os.path.join(dir_vectors_inner, targets[i] + '_train' + str(N_SPLITS + 1) + '_vectors.txt'),
                                     os.path.join(dir_vectors_inner, targets[i] + '_test' + str(N_SPLITS + 1) + '_vectors.txt'))
            f_output_result.write('uni' + str(num) + '\t' +
                                  str(len(set_idx_uni)) + '\t' + str(score_uni_val) + '\n')

        f_output_result.close()
        f_output_int.close()
        f_output_uni.close()

    with open(os.path.join(dir_result, 'best_score.txt'), 'w') as f_output:
        f_output.write('Comb\tN_int\tF1_val_int\tN_uni\tF1_val_uni\n')
        for i in range(len(targets)):
            f_output.write(''.join(str(int(x) + 1) for x in best_comb_int[i]) + '\t' + str(best_n_int[i]) + '\t' + str(best_score_int[i]) + '\t' +
                           ''.join(str(int(x) + 1) for x in best_comb_uni[i]) + '\t' + str(best_n_uni[i]) + '\t' + str(best_score_uni[i]) + '\n')

            with open(os.path.join(dir_result_feature_indexes, targets[i] + '_feature_indexes.txt'), 'r', encoding='utf-8') as f_input:
                lines = f_input.read().split('\n')

            j = 0
            items = lines[j].split('\t')
            best_comb = ''.join(str(int(x) + 1) for x in best_comb_uni[i])
            while j < len(lines) and items[0] != best_comb:
                j += 1
                items = lines[j].split('\t')

            if j < len(lines):
                with open(os.path.join(dir_relevant_feature_indexes, targets[i] + '_feature_indexes.txt'), 'w', encoding='utf-8') as f:
                    f.write(items[1])


if __name__ == '__main__':
    run('')

