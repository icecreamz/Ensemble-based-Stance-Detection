import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import exp, loadtxt
from datetime import datetime
import numpy as np
import pandas as pd
import ast
import os

def run(path):
    if len(path) == 0:
        file_params = os.path.join('..', 'DNRFAF_params.tsv')
    else:
        file_params = 'DNRFAF_params.tsv'

    # initial data
    dir_data = os.path.join(path, 'points_for_graph')
    dir_results = os.path.join(path, 'num_relevant_features')
    dir_vectors = os.path.join(path, 'vectors_tsv')
    file_num_relevant_features = 'num_relevant_features.txt'
    step = 10
    ymin = 0  # minimum value on the y-axis when plotting
    ymax = 0.8  # the maximum value on the y-axis when plotting graphs
    fontsize = 30

    if os.path.isfile(file_params):
        df = pd.read_csv(file_params, sep='\t')
        targets = df['Target']
        eps = [float(x) for x in df['Epsilon']]
        bounds = []
        for x in df['Bounds']:
            if x == '(-np.inf, np.inf)':
                bounds.append((-np.inf, np.inf))
            else:
                bounds.append(ast.literal_eval(x))
    else:
        targets = set()
        for file in os.listdir(os.path.join(dir_vectors, 'outer')):
            targets.add(file.split('_')[0])
        targets = sorted(list(targets))
        eps = [1e-05 for _ in range(len(targets))]
        bounds = [(-np.inf, np.inf) for _ in range(len(targets))]

    relevant_feature_numbers = []

    for i in range(len(targets)):
        print('{0} Determining the number of relevant features for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), targets[i]))

        # loading data
        data_train = loadtxt(os.path.join(dir_data, targets[i] + '_points_train.txt'))
        x_train = data_train[:, 0]
        y_train = data_train[:, 1]

        # approximating function - Weibull distribution function
        def func(x, a, b, c, d):
            return a - b * exp(-pow(x * c, d))

        # derivative of the approximating function
        def diff_func(x, a, b, c, d):
            return b * d * exp(-pow(x * c, d)) * pow(c * x, d) / x

        # function of determining the number of relevant features
        # diff_y_best - derivative values
        def find_n(diff_y_best, eps):
            flag = False
            n = -1
            for i in range(len(diff_y_best)):
                if (diff_y_best[i] < eps) & (not flag):
                    flag = True
                    n = i
                if (diff_y_best[i] >= eps) & (flag):
                    flag = False
            return n

        # determining of the parameters of the approximating function
        popt, pcov = curve_fit(func, x_train, y_train, method='dogbox', bounds=bounds[i])
        a, b, c, d = popt

        print('a = {0}\nb = {1}\nc = {2}\nd = {3}'.format(*tuple(popt)))

        # finding the values of the approximating function and building graphs
        y_best = np.array(func(x_train, *popt))
        fig = plt.figure(figsize=(18, 8))
        plt.ylim([ymin, ymax])
        x_train_plt = [x * step for x in x_train]
        plt.plot(x_train_plt, y_train, 'bo', label='train')
        plt.plot(x_train_plt, y_best, color='red', linewidth=4, label='f(x)')
        plt.title(targets[i], fontsize=fontsize, fontstyle='italic')
        plt.xlabel('Number of features', fontsize=fontsize, fontstyle='italic')
        plt.ylabel('F1-score', fontsize=fontsize, fontstyle='italic')
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.legend(loc='lower right', fontsize=fontsize)

        n_opt = find_n(diff_func(x_train, a, b, c, d), eps[i])
        plt.axvline(x=(n_opt + 1) * step, color='black', linestyle='--')
        # plt.show()
        fig.savefig(os.path.join(dir_results, targets[i] + '.png'), dpi=fig.dpi)
        print("n_opt = ", (n_opt + 1) * step)

        relevant_feature_numbers.append((n_opt + 1) * step)

        with open(os.path.join(dir_results, 'results_' + targets[i] + '.txt'), 'w', encoding='utf-8') as f:
            f.write("n_opt = {0}\n".format((n_opt + 1) * step))
            f.write("eps = {0}\n".format(eps[i]))
            f.write("bounds = {0}\n".format(bounds[i]))
            f.write('a = {0}\nb = {1}\nc = {2}\nd = {3}\n'.format(*tuple(popt)))

    df = pd.DataFrame(columns=['Target', 'Number of relevant features'])
    for i in range(len(targets)):
        df = df.append({'Target': targets[i], 'Number of relevant features': relevant_feature_numbers[i]}, ignore_index=True)
    df.to_csv(os.path.join(dir_results, file_num_relevant_features), sep='\t', index=False)


if __name__ == '__main__':
    run('')
