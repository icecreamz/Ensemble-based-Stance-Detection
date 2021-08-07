import shutil
import os

def clear_dir(dir):
    for file in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, file)):
            os.remove(os.path.join(dir, file))
        else:
            shutil.rmtree(os.path.join(dir, file))

def run(path):
    clear_dir(os.path.join(path, 'feature_indexes'))
    clear_dir(os.path.join(path, 'reports'))
    clear_dir(os.path.join(path, 'vectors_tsv', 'inner'))
    clear_dir(os.path.join(path, 'vectors_tsv', 'outer'))

if __name__ == '__main__':
    run('')