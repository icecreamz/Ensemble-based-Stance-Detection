from datetime import datetime
from nltk import ngrams
from collections import Counter
import pandas as pd
import stanza
import shutil
import os

ngrams_word_num = 3
ngrams_character_num = 5
ngrams_pos_num = 3

#stanza.download('en')

# Create character ngrams, word ngrams and pos ngrams
def ngrams_features(target, dir_targets_preprocessed, dir_target_dict, dir_targets_for_stance_indicative_features):
    print('{0}  Processing ngrams features'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    filename = 'train'
    dict_ngrams = {}
    dict_ngrams_pos = {}
    df = pd.read_csv(os.path.join(dir_targets_preprocessed, target + '_' + filename + '.tsv'), sep='\t')
    df['Tweet'] = df['Tweet'].fillna('')
    for i in range(len(df['Tweet'])):
        message_tokens = df['Tweet'][i].split()

        for k in range(1, ngrams_pos_num + 1):
            if len(message_tokens) >= k:
                for gram in ngrams(message_tokens, k):
                    gram_joined = ' '.join(x for x in gram)
                    if gram_joined in dict_ngrams_pos.keys():
                        dict_ngrams_pos[gram_joined] += 1
                    else:
                        dict_ngrams_pos[gram_joined] = 1

        for j in range(len(message_tokens)):
            token_splitted = message_tokens[j].split('_')
            message_tokens[j] = '_'.join(token_splitted[k] for k in range(len(token_splitted) - 1))

        for k in range(1, ngrams_word_num + 1):
            if len(message_tokens) >= k:
                for gram in ngrams(message_tokens, k):
                    gram_joined = ' '.join(x for x in gram)
                    if gram_joined in dict_ngrams.keys():
                        dict_ngrams[gram_joined] += 1
                    else:
                        dict_ngrams[gram_joined] = 1

        for k in range(1, ngrams_character_num + 1):
            for token in message_tokens:
                for j in range(len(token) - k + 1):
                    if token[j:j + k] in dict_ngrams.keys():
                        dict_ngrams[token[j:j + k]] += 1
                    else:
                        dict_ngrams[token[j:j + k]] = 1

    dict_ngrams_pos_filtered = []
    for key in dict_ngrams_pos.keys():
        if dict_ngrams_pos[key] > 1:
            dict_ngrams_pos_filtered.append(key)

    dict_ngrams_filtered = []
    for key in dict_ngrams.keys():
        if dict_ngrams[key] > 1:
            dict_ngrams_filtered.append(key)

    dict_ngrams_pos_filtered = sorted(dict_ngrams_pos_filtered)
    dict_ngrams_filtered = sorted(dict_ngrams_filtered)

    with open(os.path.join(dir_target_dict, target + '_ngram_pos_dict.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(x for x in dict_ngrams_pos_filtered))

    with open(os.path.join(dir_target_dict, target + '_ngram_dict.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(x for x in dict_ngrams_filtered))

    dict_ngrams_stance = {}
    df = pd.read_csv(os.path.join(dir_targets_for_stance_indicative_features, target + '_' + filename + '.tsv'), sep='\t')
    df['Tweet'] = df['Tweet'].fillna('')
    for i in range(len(df['Tweet'])):
        message_tokens = df['Tweet'][i].split()

        for j in range(len(message_tokens)):
            token_splitted = message_tokens[j].split('_')
            message_tokens[j] = '_'.join(token_splitted[k] for k in range(len(token_splitted) - 1))

        for k in range(1, ngrams_word_num + 1):
            if len(message_tokens) >= k:
                for gram in ngrams(message_tokens, k):
                    gram_joined = ' '.join(x for x in gram)
                    if gram_joined in dict_ngrams_stance.keys():
                        dict_ngrams_stance[gram_joined] += 1
                    else:
                        dict_ngrams_stance[gram_joined] = 1

    dict_ngrams_stance_filtered = []
    for key in dict_ngrams_stance.keys():
        if dict_ngrams_stance[key] > 1:
            dict_ngrams_stance_filtered.append(key)

    dict_ngrams_stance_filtered = list(set(dict_ngrams_stance_filtered).difference(set(dict_ngrams_filtered)))
    dict_ngrams_stance_filtered = sorted(dict_ngrams_stance_filtered)
    with open(os.path.join(dir_target_dict, target + '_ngram_stance_indicative_features_dict.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(x for x in dict_ngrams_stance_filtered))

def dependency_features(target, nlp, dir_targets_src, dir_target_dict):
    print('{0}  Processing dependency features'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    filename = 'train'
    deps = {}
    df = pd.read_csv(os.path.join(dir_targets_src, target + '_' + filename + '.tsv'), sep='\t')
    df['Tweet'] = df['Tweet'].fillna('')
    for i in range(len(df['Tweet'])):
        doc = nlp(df['Tweet'][i])
        for sentence in doc.sentences:
            for dep in sentence.dependencies:
                if dep[1] not in ['root', 'punct']:
                    dep_joined = "{0} {1}".format(dep[0].lemma.lower(), dep[1])
                    if dep_joined in deps.keys():
                        deps[dep_joined] += 1
                    else:
                        deps[dep_joined] = 1

                    dep_joined = "{0} {1}".format(dep[0].lemma.lower(), dep[2].lemma.lower())
                    if dep_joined in deps.keys():
                        deps[dep_joined] += 1
                    else:
                        deps[dep_joined] = 1

                    dep_joined = "{0} {1} {2}".format(dep[0].lemma.lower(), dep[1], dep[2].lemma.lower())
                    if dep_joined in deps.keys():
                        deps[dep_joined] += 1
                    else:
                        deps[dep_joined] = 1

    deps_filtered = []
    for key in deps.keys():
        if deps[key] > 1:
            deps_filtered.append(key)

    deps_filtered = sorted(deps_filtered)

    with open(os.path.join(dir_target_dict, target + '_dependences_dict.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(x for x in deps_filtered))

def stance_indicative_features(target, dir_targets_preprocessed, dir_targets_for_stance_indicative_features):
    print('{0}  Processing stance indicative features'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    filename = 'train'
    df = pd.read_csv(os.path.join(dir_targets_preprocessed, target + '_' + filename + '.tsv'), sep='\t')
    df['Tweet'] = df['Tweet'].fillna('')

    df_favor = df[df['Stance'] == 'FAVOR']
    message_tokens_favor = []
    for text in df_favor['Tweet']:
        message_tokens = text.split()

        for j in range(len(message_tokens)):
            token_splitted = message_tokens[j].split('_')
            message_tokens[j] = '_'.join(token_splitted[k] for k in range(len(token_splitted) - 1))

        message_tokens_favor.extend(message_tokens)

    df_against = df[df['Stance'] == 'AGAINST']
    message_tokens_against = []
    for text in df_against['Tweet']:
        message_tokens = text.split()

        for j in range(len(message_tokens)):
            token_splitted = message_tokens[j].split('_')
            message_tokens[j] = '_'.join(token_splitted[k] for k in range(len(token_splitted) - 1))

        message_tokens_against.extend(message_tokens)

    message_tokens_all = []
    for text in df['Tweet']:
        message_tokens = text.split()

        for j in range(len(message_tokens)):
            token_splitted = message_tokens[j].split('_')
            message_tokens[j] = '_'.join(token_splitted[k] for k in range(len(token_splitted) - 1))

        message_tokens_all.extend(message_tokens)

    message_tokens_favor_count = Counter(message_tokens_favor)
    message_tokens_against_count = Counter(message_tokens_against)
    message_tokens_all_count = Counter(message_tokens_all)

    favor_list = []
    against_list = []

    for token in message_tokens_favor_count.keys():
        if message_tokens_all_count[token] > 1 and \
                message_tokens_favor_count[token] / message_tokens_all_count[token] > 0.7:
            favor_list.append(token)

    for token in message_tokens_against_count.keys():
        if message_tokens_all_count[token] > 1 and \
                message_tokens_against_count[token] / message_tokens_all_count[token] > 0.7:
            against_list.append(token)

    for i in range(len(df['Tweet'])):
        message_tokens = df['Tweet'][i].split()
        for token_favor_list in favor_list:
            for j in range(len(message_tokens)):
                if message_tokens[j].startswith(token_favor_list + '_'):
                    message_tokens[j] = message_tokens[j].replace(token_favor_list + '_', 'favor_')
        for token_against_list in against_list:
            for j in range(len(message_tokens)):
                if message_tokens[j].startswith(token_against_list + '_'):
                    message_tokens[j] = message_tokens[j].replace(token_against_list + '_', 'against_')
        df['Tweet'][i] = ' '.join(x for x in message_tokens)

    df.to_csv(os.path.join(dir_targets_for_stance_indicative_features, target + '_' + filename + '.tsv'), sep='\t', index=False)

    filename = 'test'
    df = pd.read_csv(os.path.join(dir_targets_preprocessed, target + '_' + filename + '.tsv'), sep='\t')
    df['Tweet'] = df['Tweet'].fillna('')

    for i in range(len(df['Tweet'])):
        message_tokens = df['Tweet'][i].split()
        for token_favor_list in favor_list:
            for j in range(len(message_tokens)):
                if message_tokens[j].startswith(token_favor_list + '_'):
                    message_tokens[j] = message_tokens[j].replace(token_favor_list + '_', 'favor_')
        for token_against_list in against_list:
            for j in range(len(message_tokens)):
                if message_tokens[j].startswith(token_against_list + '_'):
                    message_tokens[j] = message_tokens[j].replace(token_against_list + '_', 'against_')
        df['Tweet'][i] = ' '.join(x for x in message_tokens)

    df.to_csv(os.path.join(dir_targets_for_stance_indicative_features, target + '_' + filename + '.tsv'), sep='\t', index=False)

def run(path, dir_input):
    nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse')

    if len(path) == 0:
        path = '..'
    else:
        path = ''

    if len(dir_input) == 0:
        dir_targets_src = os.path.join(path, 'data', 'targets_src')
    else:
        dir_targets_src = dir_input

    dir_targets_preprocessed = os.path.join(path, 'data', 'targets_preprocessed')
    dir_targets_for_stance_indicative_features = os.path.join(path, 'data', 'targets_preprocessed_for_stance_indicative_features')

    targets = set()
    for file in os.listdir(dir_targets_src):
        if file.endswith('.tsv'):
            targets.add(file.split('_')[0])
    targets = sorted(list(targets))

    for target in targets:
        print('{0} Processing {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), target))

        dir_target_dict = os.path.join(path, 'data', 'target_dictionaries', target)

        if os.path.isdir(dir_target_dict):
            shutil.rmtree(os.path.join(dir_target_dict))

        os.mkdir(dir_target_dict)

        stance_indicative_features(target, dir_targets_preprocessed, dir_targets_for_stance_indicative_features)
        ngrams_features(target, dir_targets_preprocessed, dir_target_dict, dir_targets_for_stance_indicative_features)
        dependency_features(target, nlp, dir_targets_src, dir_target_dict)

if __name__ == '__main__':
    run('', '')
