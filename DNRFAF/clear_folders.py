import os

def clear_dir(dir):
    for file in os.listdir(dir):
        os.remove(os.path.join(dir, file))

def run(path):
    clear_dir(os.path.join(path, 'feature_indexes', 'inner'))
    clear_dir(os.path.join(path, 'feature_indexes', 'outer'))
    clear_dir(os.path.join(path, 'num_relevant_features'))
    clear_dir(os.path.join(path, 'points_for_graph'))
    clear_dir(os.path.join(path, 'result_dependences'))
    clear_dir(os.path.join(path, 'vectors_arff', 'inner'))
    clear_dir(os.path.join(path, 'vectors_arff', 'outer'))
    clear_dir(os.path.join(path, 'vectors_tsv', 'inner'))
    clear_dir(os.path.join(path, 'vectors_tsv', 'outer'))

if __name__ == '__main__':
    run('')