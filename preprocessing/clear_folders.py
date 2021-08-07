import shutil
import os

def clear_dir(dir):
    for file in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, file)):
            os.remove(os.path.join(dir, file))
        else:
            shutil.rmtree(os.path.join(dir, file))

def run(path):
    if len(path) == 0:
        path = '..'
    else:
        path = ''

    clear_dir(os.path.join(path, 'data', 'target_dictionaries'))
    clear_dir(os.path.join(path, 'data', 'targets_preprocessed'))
    clear_dir(os.path.join(path, 'data', 'targets_preprocessed_for_stance_indicative_features'))
    clear_dir(os.path.join(path, 'data', 'vectors', 'n-grams'))
    clear_dir(os.path.join(path, 'data', 'vectors', 'dependency'))
    clear_dir(os.path.join(path, 'data', 'vectors', 'linguistic'))
    clear_dir(os.path.join(path, 'data', 'vectors', 'target name'))
    clear_dir(os.path.join(path, 'data', 'vectors', 'stance-indicative'))
    clear_dir(os.path.join(path, 'data', 'vectors', 'stylistic'))
    clear_dir(os.path.join(path, 'data', 'vectors', 'sentiment'))
    clear_dir(os.path.join(path, 'data', 'vectors', 'word2vec'))

if __name__ == '__main__':
    run('')