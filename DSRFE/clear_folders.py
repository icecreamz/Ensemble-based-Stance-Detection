import os

def clear_dir(dir):
    for file in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, file)):
            os.remove(os.path.join(dir, file))

def run(path):
    clear_dir(os.path.join(path, 'feature_indexes', 'inner'))
    clear_dir(os.path.join(path, 'feature_indexes', 'outer'))
    clear_dir(os.path.join(path, 'result_ensemble_fs', 'feature_indexes_int'))
    clear_dir(os.path.join(path, 'result_ensemble_fs', 'feature_indexes_outer'))
    clear_dir(os.path.join(path, 'result_ensemble_fs', 'feature_indexes_uni'))
    clear_dir(os.path.join(path, 'result_ensemble_fs', 'relevant_feature_indexes'))
    clear_dir(os.path.join(path, 'result_ensemble_fs'))
    clear_dir(os.path.join(path, 'vectors_arff', 'inner'))
    clear_dir(os.path.join(path, 'vectors_arff', 'outer'))
    clear_dir(os.path.join(path, 'vectors_tsv', 'inner'))
    clear_dir(os.path.join(path, 'vectors_tsv', 'outer'))

if __name__ == '__main__':
    run('')