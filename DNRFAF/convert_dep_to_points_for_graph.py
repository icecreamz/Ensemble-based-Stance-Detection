import os

N_FOLDS = 5

def run(path):
    dir_input = os.path.join(path, 'result_dependences')
    dir_output = os.path.join(path, 'points_for_graph')

    for file in os.listdir(dir_input):
        with open(os.path.join(dir_input, file), 'r', encoding='utf-8') as f:
            lines = list(filter(None, f.read().split('\n')))

        num_features_score_train = []
        num_features_score_test = []
        for line in lines[1:]:
            line_splitted = line.split('\t')
            num_features_score_train.append(0)

            for i in range(N_FOLDS):
                num_features_score_train[-1] += float(line_splitted[3 * (i + 1)])

            num_features_score_test.append(float(line_splitted[3 * N_FOLDS + 1]))
            num_features_score_train[-1] = num_features_score_train[-1] / N_FOLDS

        with open(os.path.join(dir_output, file.replace('dep_', '').replace('.txt', '_points_train.txt')), 'w', encoding='utf-8') as f:
            f.write('\n'.join(str(i) + '\t' + str(num_features_score_train[i]) for i in range(len(num_features_score_train))))

        with open(os.path.join(dir_output, file.replace('dep_', '').replace('.txt', '_points_test.txt')), 'w', encoding='utf-8') as f:
            f.write('\n'.join(str(i) + '\t' + str(num_features_score_test[i]) for i in range(len(num_features_score_test))))

if __name__ == '__main__':
    run('')
