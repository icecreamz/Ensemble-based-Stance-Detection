from datetime import datetime
import shutil
import os

def ReadGuessData(labels):
    guess = []
    for label in labels:
        if int(label) == 1:
            guess.append('AGAINST')
        elif int(label) == 2:
            guess.append('FAVOR')
        elif int(label) == 3:
            guess.append('NONE')

    return guess

def run(path, dir_input, dir_output):
    dir_vectors = os.path.join(path, 'vectors_tsv')

    targets = set()
    for file in os.listdir(os.path.join(dir_vectors, 'outer')):
        if file.endswith('.txt'):
            targets.add(file.split('_')[0])
    targets = sorted(list(targets))

    if len(dir_output) == 0:
        dir_results = os.path.join('..', 'results_classification')
    else:
        dir_results = dir_output

    if os.path.exists(dir_results):
        shutil.rmtree(dir_results)

    os.mkdir(dir_results)

    for target in targets:
        print('{0} Converting predictions to text file for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), target))

        if len(dir_input) == 0:
            input_gold_data = os.path.join('..', 'data', 'targets_src', target + '_test.tsv')
        else:
            input_gold_data = os.path.join(dir_input, target + '_test.tsv')

        dir_reports = os.path.join(path, 'reports', 'reports_' + target)

        with open(input_gold_data, 'r', encoding='utf-8') as f:
            lines_gold = list(filter(None, f.read().split('\n')))

        with open(os.path.join(dir_reports, 'predict_ens_test.txt'), 'r', encoding='utf-8') as f:
            labels = f.read().split(' ')
            guess_data = ReadGuessData(labels)

        with open(os.path.join(dir_results, target + '_guess_ens.txt'), 'w', encoding='utf-8') as f:
            f.write('ID\tTarget\tTweet\tStance')
            for i in range(1, len(lines_gold)):
                f.write('\n%d\t%s\t%s\t%s' % (i, target, lines_gold[i].split('\t')[0], guess_data[i-1]))

        with open(os.path.join(dir_results, target + '_gold_ens.txt'), 'w', encoding='utf-8') as f:
            f.write('ID\tTarget\tTweet\tStance')
            for i in range(1, len(lines_gold)):
                items = lines_gold[i].split('\t')
                f.write('\n%d\t%s\t%s\t%s' % (i, target, items[0], items[2]))


if __name__ == '__main__':
    run('', '', '')
