from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from weka.core.converters import Loader
from datetime import datetime
import weka.core.jvm as jvm
import os

def run(path):
    dir_data = os.path.join(path, 'vectors_arff')
    dir_feature_indexes = os.path.join(path, 'feature_indexes')

    if len(path) == 0:
        jvm.start()

    loader = Loader(classname="weka.core.converters.ArffLoader")
    search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-T", "-1.7976931348623157E308", "-N", "-1"])
    evaluator = ASEvaluation(classname="weka.attributeSelection.InfoGainAttributeEval")
    attsel = AttributeSelection()
    attsel.search(search)
    attsel.evaluator(evaluator)

    for dir_cv in ['inner', 'outer']:
        for file in os.listdir(os.path.join(dir_data, dir_cv)):
            if (dir_cv == 'inner' and '_test' in file) or (dir_cv == 'outer' and '_train' in file):
                print('{0} Selection features for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), file))

                data = loader.load_file(os.path.join(dir_data, dir_cv, file))
                data.class_is_last()

                attsel.select_attributes(data)

                print('# number of attributes: {0}'.format(str(attsel.number_attributes_selected)))
                print('attributes: {0}'.format(str(attsel.selected_attributes)))

                with open(os.path.join(dir_feature_indexes, dir_cv, file.replace('_vectors.arff', '_feature_indexes.txt')), 'w', encoding='utf-8') as f:
                    f.write(' '.join(str(i) for i in attsel.selected_attributes[:len(attsel.selected_attributes) - 1]))

    if len(path) == 0:
        jvm.stop()


if __name__ == '__main__':
    run('')