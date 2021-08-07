from sklearn.datasets import load_svmlight_file
from datetime import datetime
import os

def run(path):
    dir_tsv = os.path.join(path, 'vectors_tsv')
    dir_arff = os.path.join(path, 'vectors_arff')

    for dir_cv in ['inner', 'outer']:
        for file in os.listdir(os.path.join(dir_tsv, dir_cv)):
            if (dir_cv == 'inner' and '_test' in file) or (dir_cv == 'outer' and '_train' in file):
                print('{0} Converting tsv to arff for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), file))

                target = file.split('_')[0]

                # Load the dataset
                X_data, y_data = load_svmlight_file(os.path.join(dir_tsv, dir_cv, file))

                X_data = X_data[:X_data.shape[0] - 1, :]
                y_data = y_data[:len(y_data) - 1]

                with open(os.path.join(dir_arff, dir_cv, file.replace('.txt', '.arff')), 'w', encoding='utf-8') as f:
                    f.write('% 1. Title: {0}'.format(target))
                    f.write('\n@RELATION {0}\n'.format(file.split('_')[0].replace(' ', '')))

                    for i in range(X_data.shape[1]):
                        f.write('\n@ATTRIBUTE word{0} NUMERIC'.format(str(i)))

                    f.write('\n@ATTRIBUTE class {1,2,3}')
                    f.write('\n\n@DATA')

                    for i in range(X_data.shape[0]):
                        f.write('\n' + ','.join(str(float(X_data[i,j])) for j in range(X_data.shape[1])) + ',' + str(int(y_data[i])))

if __name__ == '__main__':
    run('')