from sklearn.metrics import f1_score
from datetime import datetime
import numpy as np
import itertools
import os

N = 5
M = 5
# 0 - SVM, 1 - kNN, 2 - AB, 3 - DT, 4 - RF
clfs = [0, 1, 2, 3, 4]

def comb_inner(dir_reports):
    combs_best = []

    with open(os.path.join(dir_reports, 'predict_inner.txt'), 'r', encoding='utf-8') as f:
        lines = list(filter(None, f.read().split('\n')))

    with open(os.path.join(dir_reports, 'real_inner.txt'), 'r', encoding='utf-8') as f:
        real = list(filter(None, f.read().split('\n')))

    with open(os.path.join(dir_reports, 'report_comb_inner.csv'), 'w', encoding='utf-8') as f_report:
        f_report.write('Combination;Scores;MeanScore\n')

        scores = []
        mean_scores = []
        combs = []

        for num in range(1, len(clfs) + 1):
            for comb in itertools.combinations(''.join(str(x) for x in clfs), num):
                combs.append(comb)
                f1_scores = []
                for j in range(N):
                    labels = []
                    y_pred = []
                    for c in range(M):
                        labels.append(lines[c * N + j].split())
                    y_true = [int(x) for x in real[j].split()]
                    for k in range(len(labels[0])):
                        rates = [0, 0, 0]
                        for index_clf in range(num):
                            rates[int(labels[int(comb[index_clf])][k]) - 1] += 1
                        if rates[0] > rates[1] and rates[0] > rates[2]:
                            y_pred.append(1)
                        elif rates[1] > rates[0] and rates[1] > rates[2]:
                            y_pred.append(2)
                        elif rates[2] > rates[0] and rates[2] > rates[1]:
                            y_pred.append(3)
                        else:
                            y_pred.append(2)
                    f1_scores.append(f1_score(y_true, y_pred, average='macro'))
                scores.append(' '.join(str(x) for x in f1_scores))
                mean_scores.append(np.mean(f1_scores))

        mean_score_max = max(mean_scores)
        index_max = [index for index, value in enumerate(mean_scores) if value == mean_score_max]
        combs_best.append(','.join(str(x) for x in combs[index_max[0]]))
        for index in range(len(scores)):
            f_report.write(' '.join(str(x) for x in combs[index]) + ';')
            f_report.write(scores[index] + ';')
            f_report.write(str(mean_scores[index]))
            if index == index_max[0]:
                f_report.write(';*')
            f_report.write('\n')

    with open(os.path.join(dir_reports, 'best_comb_inner.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(x for x in combs_best))

def comb_outer(dir_reports):
    scores = []

    with open(os.path.join(dir_reports, 'best_comb_inner.txt'), 'r', encoding='utf-8') as f:
        comb = list(filter(None, f.read().split('\n')))[0].split(',')

    with open(os.path.join(dir_reports, 'predict_test.txt'), 'r', encoding='utf-8') as f:
        lines = list(filter(None, f.read().split('\n')))

    with open(os.path.join(dir_reports, 'real_test.txt'), 'r', encoding='utf-8') as f:
        real = list(filter(None, f.read().split('\n')))

    labels = []
    y_pred = []
    y_true = [int(x) for x in real[0].split()]
    for c in range(M):
        labels.append(lines[c].split())
    for k in range(len(labels[0])):
        rates = [0, 0, 0]
        for index_clf in range(len(comb)):
            rates[int(labels[int(comb[index_clf])][k]) - 1] += 1
        if rates[0] > rates[1] and rates[0] > rates[2]:
            y_pred.append(1)
        elif rates[1] > rates[0] and rates[1] > rates[2]:
            y_pred.append(2)
        elif rates[2] > rates[0] and rates[2] > rates[1]:
            y_pred.append(3)
        else:
            y_pred.append(2)
    scores.append(f1_score(y_true, y_pred, average='macro'))

    with open(os.path.join(dir_reports, 'predict_ens_test.txt'), 'w', encoding='utf-8') as f:
        f.write(' '.join(str(x) for x in y_pred))

def run(path):
    dir_vectors = os.path.join(path, 'vectors_tsv')

    targets = set()
    for file in os.listdir(os.path.join(dir_vectors, 'outer')):
        if file.endswith('.txt'):
            targets.add(file.split('_')[0])
    targets = sorted(list(targets))

    for target in targets:
        print('{0} Getting classifier predictions for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), target))

        dir_reports = os.path.join(path, 'reports', 'reports_' + target)

        comb_inner(dir_reports)
        comb_outer(dir_reports)

if __name__ == '__main__':
    run('')
