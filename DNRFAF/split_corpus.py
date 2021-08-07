from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from shutil import copyfile
import os

N_SPLITS = 5
SEED = 2

def print_to_file(texts, corpus, labels, last_line, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for text_index in texts:
            f.write(labels[text_index] + ' ' + corpus[text_index] + '\n')
        f.write(last_line)
        f.close()

def run(path):
    if len(path) == 0:
        dir_src = os.path.join('..', 'data', 'vectors', 'n-grams')
    else:
        dir_src = os.path.join('data', 'vectors', 'n-grams')
    dir_inner = os.path.join(path, 'vectors_tsv', 'inner')
    dir_outer = os.path.join(path, 'vectors_tsv', 'outer')

    for file in os.listdir(dir_src):
        if '_train' in file:
            print('{0} Splitting {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), file))

            corpus = []
            labels = []

            with open(os.path.join(dir_src, file), 'r', encoding='utf-8') as f:
                lines = list(filter(None, f.read().split('\n')))

            last_line = lines[-1]
            lines = lines[:len(lines) - 1]

            for line in lines:
                items = line.split()
                labels.append(items[0])
                corpus.append(' '.join(x for x in items[1:]))

            kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

            j = 1
            for train, test in kf.split(corpus, labels):
                print_to_file(train, corpus, labels, last_line, os.path.join(dir_inner, file.replace('_train', '_train' + str(j))))
                print_to_file(test, corpus, labels, last_line, os.path.join(dir_inner, file.replace('_train', '_test' + str(j))))
                j += 1

        if file.endswith('.txt'):
            copyfile(os.path.join(dir_src, file), os.path.join(dir_outer, file))

if __name__ == '__main__':
    run('')
