from datetime import datetime
import weka.core.jvm as jvm
import sys
import os

import preprocessing.clear_folders
import preprocessing.preprocessing
import preprocessing.feature_engeneering
import preprocessing.build_text_representation_model

import DNRFAF.clear_folders
import DNRFAF.split_corpus
import DNRFAF.convert_tsv_to_arff
import DNRFAF.select_features
import DNRFAF.build_dependence
import DNRFAF.convert_dep_to_points_for_graph
import DNRFAF.find_num_relevant_features

import DSRFE.clear_folders
import DSRFE.split_corpus
import DSRFE.convert_tsv_to_arff
import DSRFE.select_features
import DSRFE.ensemble_fs

import DECCV.clear_folders
import DECCV.split_corpus
import DECCV.find_optimal_params_clfs
import DECCV.get_predictions_clfs
import DECCV.find_optimal_comb
import DECCV.convert_predictions_to_text

def print_help():
    print('Usage: python esd.py [--input <input_data_path>] [--output <output_data_path>]\n')
    print('Arguments:')
    print('\t--input\t\t\t\tpath to input directory (default: data/targets_src)')
    print('\t--output\t\t\tpath to output directory (default: results_classification)')


if __name__ == '__main__':
    input_dir = os.path.join('data', 'targets_src')
    output_dir = 'results_classification'
    stop_run = False

    if len(sys.argv) > 1:
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == '--help':
                print_help()
                stop_run = True
                break
            elif sys.argv[i] == '--input' and i + 1 < len(sys.argv):
                input_dir = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]
                i += 2
            else:
                print('Error: invalid command line argument')
                print_help()
                stop_run = True
                break

    if stop_run:
        exit(0)

    jvm.start()

    print('{0} Stage 1: Preprocessing'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    preprocessing.clear_folders.run('ESD')
    preprocessing.preprocessing.run('ESD', input_dir)
    preprocessing.feature_engeneering.run('ESD', input_dir)
    preprocessing.build_text_representation_model.run('ESD', input_dir)

    print('\n{0} Stage 2: DNRFAF algorithm'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    DNRFAF.clear_folders.run('DNRFAF')
    DNRFAF.split_corpus.run('DNRFAF')
    DNRFAF.convert_tsv_to_arff.run('DNRFAF')
    DNRFAF.select_features.run('DNRFAF')
    DNRFAF.build_dependence.run('DNRFAF')
    DNRFAF.convert_dep_to_points_for_graph.run('DNRFAF')
    DNRFAF.find_num_relevant_features.run('DNRFAF')

    print('\n{0} Stage 3: DSRFE algorithm'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    DSRFE.clear_folders.run('DSRFE')
    DSRFE.split_corpus.run('DSRFE')
    DSRFE.convert_tsv_to_arff.run('DSRFE')
    DSRFE.select_features.run('DSRFE')
    DSRFE.ensemble_fs.run('DSRFE')

    print('\n{0} Stage 4: DECCV algorithm'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    DECCV.clear_folders.run('DECCV')
    DECCV.split_corpus.run('DECCV')
    DECCV.find_optimal_params_clfs.run('DECCV')
    DECCV.get_predictions_clfs.run('DECCV')
    DECCV.find_optimal_comb.run('DECCV')
    DECCV.convert_predictions_to_text.run('DECCV', input_dir, output_dir)

    jvm.stop()

    exit(0)
