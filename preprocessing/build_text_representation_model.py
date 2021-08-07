from datetime import datetime
from nltk import ngrams
from collections import Counter
from twokenize import tokenizeRawTweetText
import os
import gensim.downloader
import pandas as pd
import numpy as np
import stanza

os.environ['PYTHONINSPECT'] = 'True'

ngrams_word_num = 3
ngrams_character_num = 5
ngrams_character_num_target = 5
ngrams_pos_num = 3

# N-grams features
def ngrams_features(dir_targets_src, dir_target_dict, dir_vectors, dir_targets_preprocessed):
    print('{0} Creating n-gram features'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    targets = set()
    for file in os.listdir(dir_targets_src):
        if file.endswith('.tsv'):
            targets.add(file.split('_')[0])
    targets = sorted(list(targets))

    counter_features = {}
    for target in targets:
        counter_features[target] = 0

    for target in targets:
        print('{0}  Building text representation model for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), target))

        with open(os.path.join(dir_target_dict, target, target + '_ngram_dict.txt'), 'r') as f:
            dict_ngrams = f.read().split('\n')

        with open(os.path.join(dir_target_dict, target, target + '_ngram_pos_dict.txt'), 'r') as f:
            dict_ngrams_pos = f.read().split('\n')

        filename = ['train', 'test']
        for file in filename:
            with open(os.path.join(dir_vectors, 'n-grams', target + '_' + file + '_vectors.txt'), 'w') as f:
                df = pd.read_csv(os.path.join(dir_targets_preprocessed, target + '_' + file + '.tsv'), sep='\t')
                df['Tweet'] = df['Tweet'].fillna('')

                ngram_doc = {}
                for i in range(len(df['Tweet'])):
                    message_tokens = df['Tweet'][i].split()

                    for j in range(len(message_tokens)):
                        token_splitted = message_tokens[j].split('_')
                        message_tokens[j] = '_'.join(token_splitted[k] for k in range(len(token_splitted) - 1))

                    ngrams_tweet = []
                    for k in range(1, ngrams_word_num + 1):
                        if len(message_tokens) >= k:
                            for gram in ngrams(message_tokens, k):
                                gram_joined = ' '.join(x for x in gram)
                                if gram_joined in dict_ngrams:
                                    ngrams_tweet.append(gram_joined)

                    ngrams_tweet_count = Counter(ngrams_tweet)
                    for key in ngrams_tweet_count.keys():
                        if key in ngram_doc.keys():
                            ngram_doc[key] += 1
                        else:
                            ngram_doc[key] = 1

                for i in range(len(df['Tweet'])):
                    message_tokens = df['Tweet'][i].split()

                    ngrams_pos_tweet = []
                    for k in range(1, ngrams_pos_num + 1):
                        if len(message_tokens) >= k:
                            for gram in ngrams(message_tokens, k):
                                gram_joined = ' '.join(x for x in gram)
                                if gram_joined in dict_ngrams_pos:
                                    ngrams_pos_tweet.append(gram_joined)

                    for j in range(len(message_tokens)):
                        token_splitted = message_tokens[j].split('_')
                        message_tokens[j] = '_'.join(token_splitted[k] for k in range(len(token_splitted) - 1))

                    ngrams_tweet = []
                    for k in range(1, ngrams_word_num + 1):
                        if len(message_tokens) >= k:
                            for gram in ngrams(message_tokens, k):
                                gram_joined = ' '.join(x for x in gram)
                                if gram_joined in dict_ngrams:
                                    ngrams_tweet.append(gram_joined)

                    for k in range(1, ngrams_character_num + 1):
                        for token in message_tokens:
                            for j in range(len(token) - k + 1):
                                if token[j:j + k] in dict_ngrams:
                                    ngrams_tweet.append(token[j:j + k])

                    ngrams_pos_tweet_count = Counter(ngrams_pos_tweet)
                    ngrams_tweet_count = Counter(ngrams_tweet)

                    if df['Stance'][i] == 'AGAINST':
                        f.write('1')
                    elif df['Stance'][i] == 'FAVOR':
                        f.write('2')
                    else:
                        f.write('3')

                    feature_indexes = []
                    for key in sorted(ngrams_tweet_count):
                        if key in ngram_doc.keys():
                            feature_indexes.append(str(dict_ngrams.index(key)) + ':' + str(ngrams_tweet_count[key]/ngram_doc[key]))
                        else:
                            feature_indexes.append(str(dict_ngrams.index(key)) + ':' + str(ngrams_tweet_count[key]))
                    for key in sorted(ngrams_pos_tweet_count):
                        feature_indexes.append(str(len(dict_ngrams) + dict_ngrams_pos.index(key)) + ':' + str(ngrams_pos_tweet_count[key]))

                    if len(feature_indexes) > 0:
                        f.write(' ' + ' '.join(x for x in feature_indexes))
                    f.write('\n')

                f.write('3 0:1 ' + str(counter_features[target] + len(dict_ngrams) + len(dict_ngrams_pos) - 1) + ':1')

        counter_features[target] += len(dict_ngrams) + len(dict_ngrams_pos)

    dict_counter_features = {'Target': targets, 'N-grams': list(counter_features.values())}
    df_counter_features = pd.DataFrame(dict_counter_features, columns = dict_counter_features.keys())
    df_counter_features.to_csv(os.path.join(dir_vectors, 'n-grams', 'counter_features.tsv'), sep='\t', index=False)

def dependency_features(dir_targets_src, dir_target_dict, dir_vectors):
    nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse')

    print('{0} Creating dependency features'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    df_counter_features = pd.read_csv(os.path.join(dir_vectors, 'n-grams', 'counter_features.tsv'), sep='\t')

    counter_features = {}
    df_targets = df_counter_features['Target']
    df_ngrams = df_counter_features['N-grams']
    for i in range(len(df_targets)):
        counter_features[df_targets[i]] = df_ngrams[i]

    targets = set()
    for file in os.listdir(dir_targets_src):
        if file.endswith('.tsv'):
            targets.add(file.split('_')[0])
    targets = sorted(list(targets))

    for target in targets:
        print('{0}  Building text representation model for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), target))

        with open(os.path.join(dir_target_dict, target, target + '_dependences_dict.txt'), 'r') as f:
            dict_dependences = f.read().split('\n')

        filename = ['train', 'test']
        for file in filename:
            with open(os.path.join(dir_vectors, 'n-grams', target + '_' + file + '_vectors.txt'), 'r') as f:
                text_vectors = f.read().split('\n')
                text_vectors = text_vectors[:len(text_vectors) - 1]

            with open(os.path.join(dir_vectors, 'dependency', target + '_' + file + '_vectors.txt'), 'w') as f:
                df = pd.read_csv(os.path.join(dir_targets_src, target + '_' + file + '.tsv'), sep='\t')
                df['Tweet'] = df['Tweet'].fillna('')
                for i in range(len(df['Tweet'])):
                    doc = nlp(df['Tweet'][i])
                    deps_tweet = []
                    for sentence in doc.sentences:
                        for dep in sentence.dependencies:
                            if dep[1] not in ['root', 'punct']:
                                dep_joined = "{0} {1}".format(dep[0].lemma.lower(), dep[1])
                                if dep_joined in dict_dependences:
                                    deps_tweet.append(dep_joined)

                                dep_joined = "{0} {1}".format(dep[0].lemma.lower(), dep[2].lemma.lower())
                                if dep_joined in dict_dependences:
                                    deps_tweet.append(dep_joined)

                                dep_joined = "{0} {1} {2}".format(dep[0].lemma.lower(), dep[1], dep[2].lemma.lower())
                                if dep_joined in dict_dependences:
                                    deps_tweet.append(dep_joined)

                    deps_tweet_count = Counter(deps_tweet)

                    feature_indexes = []
                    for key in sorted(deps_tweet_count):
                        feature_indexes.append(str(counter_features[target] + dict_dependences.index(key)) +
                                               ':' + str(deps_tweet_count[key]))

                    f.write(text_vectors[i])
                    if len(feature_indexes) > 0:
                        f.write(' ' + ' '.join(x for x in feature_indexes))
                    f.write('\n')

                f.write('3 0:1 ' + str(counter_features[target] + len(dict_dependences) - 1) + ':1')

        counter_features[target] += len(dict_dependences)

    df_counter_features['Dependency'] = list(counter_features.values())
    df_counter_features.to_csv(os.path.join(dir_vectors, 'dependency', 'counter_features.tsv'), sep='\t', index=False)

def linguistic_features(dir_targets_src, dir_vectors, dir_targets_preprocessed, dir_dict):
    print('{0} Creating linguistic features'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    df_counter_features = pd.read_csv(os.path.join(dir_vectors, 'dependency', 'counter_features.tsv'), sep='\t')

    counter_features = {}
    df_targets = df_counter_features['Target']
    df_dependency = df_counter_features['Dependency']
    for i in range (len(df_targets)):
        counter_features[df_targets[i]] = df_dependency[i]

    clusters = {}
    with open(os.path.join(dir_dict, '50mpaths2'), 'r', encoding='utf-8') as f:
        lines = filter(None, f.read().split('\n'))
    for line in lines:
        line_splitted = line.split('\t')
        if line_splitted[0] in clusters.keys():
            clusters[line_splitted[0]].append(line_splitted[1])
        else:
            clusters[line_splitted[0]] = []

    targets = set()
    for file in os.listdir(dir_targets_src):
        if file.endswith('.tsv'):
            targets.add(file.split('_')[0])
    targets = sorted(list(targets))

    for target in targets:
        print('{0}  Building text representation model for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), target))

        with open(os.path.join(dir_dict, 'en_negations.txt'), 'r') as f:
            dict_negations = f.read().split('\n')

        filename = ['train', 'test']
        for file in filename:
            with open(os.path.join(dir_vectors, 'dependency', target + '_' + file + '_vectors.txt'), 'r') as f:
                text_vectors = f.read().split('\n')
                text_vectors = text_vectors[:len(text_vectors) - 1]

            with open(os.path.join(dir_vectors, 'linguistic', target + '_' + file + '_vectors.txt'), 'w') as f:
                df = pd.read_csv(os.path.join(dir_targets_preprocessed, target + '_' + file + '.tsv'), sep='\t')
                df['Tweet'] = df['Tweet'].fillna('')
                for i in range(len(df['Tweet'])):
                    message_tokens = df['Tweet'][i].split()

                    for j in range(len(message_tokens)):
                        token_splitted = message_tokens[j].split('_')
                        message_tokens[j] = '_'.join(token_splitted[k] for k in range(len(token_splitted) - 1))

                    tokens_count = Counter(message_tokens)
                    negations_count = 0
                    for token in tokens_count.keys():
                        if token in dict_negations:
                            negations_count += tokens_count[token]

                    f.write(text_vectors[i])

                    if negations_count > 0:
                        f.write(' ' + str(counter_features[target]) + ':' + str(negations_count))

                    for j in range(len(clusters)):
                        count_tokens = 0
                        for token in tokens_count.keys():
                            if token in clusters[list(clusters.keys())[j]]:
                                count_tokens += 1

                        if count_tokens > 0:
                            f.write(' ' + str(counter_features[target] + 1 + j) + ':' + str(count_tokens))
                    f.write('\n')

                f.write('3 0:1 ' + str(counter_features[target] + len(clusters)) + ':1')

        counter_features[target] += 1 + len(clusters)

    df_counter_features['Linguistic'] = list(counter_features.values())
    df_counter_features.to_csv(os.path.join(dir_vectors, 'linguistic', 'counter_features.tsv'), sep='\t', index=False)

def target_features(dir_targets_src, dir_vectors):
    print('{0} Creating target features'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    df_counter_features = pd.read_csv(os.path.join(dir_vectors, 'linguistic', 'counter_features.tsv'), sep='\t')

    counter_features = {}
    df_targets = df_counter_features['Target']
    df_linguistic = df_counter_features['Linguistic']
    for i in range(len(df_targets)):
        counter_features[df_targets[i]] = df_linguistic[i]

    targets = set()
    for file in os.listdir(dir_targets_src):
        if file.endswith('.tsv'):
            targets.add(file.split('_')[0])
    targets = sorted(list(targets))

    for target in targets:
        print('{0}  Building text representation model for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), target))

        filename = ['train', 'test']
        for file in filename:
            with open(os.path.join(dir_vectors, 'linguistic', target + '_' + file + '_vectors.txt'), 'r') as f:
                text_vectors = f.read().split('\n')
                text_vectors = text_vectors[:len(text_vectors) - 1]

            with open(os.path.join(dir_vectors, 'target name', target + '_' + file + '_vectors.txt'), 'w') as f:
                df = pd.read_csv(os.path.join(dir_targets_src, target + '_' + file + '.tsv'), sep='\t')
                df['Tweet'] = df['Tweet'].fillna('')
                for i in range(len(df['Tweet'])):
                    f.write(text_vectors[i])

                    target_lower = target.lower()

                    # Presence of the whole name of the target
                    if target_lower in df['Tweet'][i].lower():
                        f.write(' ' + str(counter_features[target]) + ':1')

                    # Presence of character N-grams of the name of the target
                    count = 0
                    for k in range(2, ngrams_character_num_target + 1):
                        for j in range(len(target_lower) - k + 1):
                            count += 1
                            if target_lower[j:j + k] in df['Tweet'][i].lower():
                                f.write(' ' + str(counter_features[target] + count) + ':1')

                    # Presence of word N-grams of the name of the target
                    target_tokens = target_lower.split()
                    for k in range(1, ngrams_word_num + 1):
                        if len(target_tokens) >= k:
                            for gram in ngrams(target_tokens, k):
                                count += 1
                                gram_joined = ' '.join(x for x in gram)
                                if gram_joined in df['Tweet'][i].lower():
                                    f.write(' ' + str(counter_features[target] + count) + ':1')

                    f.write('\n')

                f.write('3 0:1 ' + str(counter_features[target] + count) + ':1')

        counter_features[target] += count + 1

    df_counter_features['Target name'] = list(counter_features.values())
    df_counter_features.to_csv(os.path.join(dir_vectors, 'target name', 'counter_features.tsv'), sep='\t', index=False)

def stance_indicative_features(dir_targets_src, dir_target_dict, dir_vectors, dir_targets_for_stance_indicative_features):
    print('{0} Creating stance-indicative features'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    df_counter_features = pd.read_csv(os.path.join(dir_vectors, 'target name', 'counter_features.tsv'), sep='\t')

    counter_features = {}
    df_targets = df_counter_features['Target']
    df_target_name = df_counter_features['Target name']
    for i in range(len(df_targets)):
        counter_features[df_targets[i]] = df_target_name[i]

    targets = set()
    for file in os.listdir(dir_targets_src):
        if file.endswith('.tsv'):
            targets.add(file.split('_')[0])
    targets = sorted(list(targets))

    for target in targets:
        print('{0}  Building text representation model for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), target))

        with open(os.path.join(dir_target_dict, target, target + '_ngram_stance_indicative_features_dict.txt'), 'r') as f:
            dict_ngrams = f.read().split('\n')

        filename = ['train', 'test']
        for file in filename:
            with open(os.path.join(dir_vectors, 'target name', target + '_' + file + '_vectors.txt'), 'r') as f:
                text_vectors = f.read().split('\n')
                text_vectors = text_vectors[:len(text_vectors) - 1]

            with open(os.path.join(dir_vectors, 'stance-indicative', target + '_' + file + '_vectors.txt'), 'w') as f:
                df = pd.read_csv(os.path.join(dir_targets_for_stance_indicative_features, target + '_' + file + '.tsv'), sep='\t')
                df['Tweet'] = df['Tweet'].fillna('')
                for i in range(len(df['Tweet'])):
                    message_tokens = df['Tweet'][i].split()

                    for j in range(len(message_tokens)):
                        token_splitted = message_tokens[j].split('_')
                        message_tokens[j] = '_'.join(token_splitted[k] for k in range(len(token_splitted) - 1))

                    ngrams_tweet = []
                    for k in range(1, ngrams_word_num + 1):
                        if len(message_tokens) >= k:
                            for gram in ngrams(message_tokens, k):
                                gram_joined = ' '.join(x for x in gram)
                                if gram_joined in dict_ngrams:
                                    ngrams_tweet.append(gram_joined)

                    ngrams_tweet_count = Counter(ngrams_tweet)

                    feature_indexes = []
                    for key in sorted(ngrams_tweet_count):
                        feature_indexes.append(str(counter_features[target] + dict_ngrams.index(key)) + ':' + str(ngrams_tweet_count[key]))

                    f.write(text_vectors[i])

                    if len(feature_indexes) > 0:
                        f.write(' ' + ' '.join(x for x in feature_indexes))
                    f.write('\n')

                f.write('3 0:1 ' + str(counter_features[target] + len(dict_ngrams) - 1) + ':1')

        counter_features[target] += len(dict_ngrams)

    df_counter_features['Stance-indicative'] = list(counter_features.values())
    df_counter_features.to_csv(os.path.join(dir_vectors, 'stance-indicative', 'counter_features.tsv'), sep='\t', index=False)

def stylistic_features(dir_targets_src, dir_vectors):
    print('{0} Creating stylistic features'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    df_counter_features = pd.read_csv(os.path.join(dir_vectors, 'stance-indicative', 'counter_features.tsv'), sep='\t')

    counter_features = {}
    df_targets = df_counter_features['Target']
    df_stance_indicative = df_counter_features['Stance-indicative']
    for i in range(len(df_targets)):
        counter_features[df_targets[i]] = df_stance_indicative[i]

    targets = set()
    for file in os.listdir(dir_targets_src):
        if file.endswith('.tsv'):
            targets.add(file.split('_')[0])
    targets = sorted(list(targets))

    for target in targets:
        print('{0}  Building text representation model for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), target))

        filename = ['train', 'test']
        for file in filename:
            with open(os.path.join(dir_vectors, 'stance-indicative', target + '_' + file + '_vectors.txt'), 'r') as f:
                text_vectors = f.read().split('\n')
                text_vectors = text_vectors[:len(text_vectors) - 1]

            with open(os.path.join(dir_vectors, 'stylistic', target + '_' + file + '_vectors.txt'), 'w') as f:
                df = pd.read_csv(os.path.join(dir_targets_src, target + '_' + file + '.tsv'), sep='\t')
                df['Tweet'] = df['Tweet'].fillna('')
                for i in range(len(df['Tweet'])):
                    tokens = tokenizeRawTweetText(df['Tweet'][i])

                    f.write(text_vectors[i])

                    # Tweet length
                    tweet_length = len(tokens)
                    if tweet_length > 0:
                        f.write(' ' + str(counter_features[target]) + ':' + str(tweet_length))

                    # Average word lenth
                    avg_word_length = round(sum([len(token) for token in tokens])/len(tokens), 2)
                    if avg_word_length > 0:
                        f.write(' ' + str(counter_features[target] + 1) + ':' + str(avg_word_length))

                    # Number of capitalized words
                    count_capitalized_words = sum([1 if token[0].isupper() else 0 for token in tokens])
                    if count_capitalized_words > 0:
                        f.write(' ' + str(counter_features[target] + 2) + ':' + str(count_capitalized_words))

                    # Number of punctuation marks
                    punct_marks_count = {',': 0, '.': 0, ';': 0, '?': 0, '!': 0}
                    for punct in punct_marks_count.keys():
                        punct_marks_count[punct] = df['Tweet'][i].count(punct)
                    for j in range(len(punct_marks_count.keys())):
                        if punct_marks_count[list(punct_marks_count.keys())[j]] > 0:
                            f.write(' ' + str(counter_features[target] + 3 + j) + ':' +
                                    str(punct_marks_count[list(punct_marks_count.keys())[j]]))

                    # Number of mention symbols
                    count_mention_symbols = df['Tweet'][i].count('@')
                    if count_mention_symbols > 0:
                        f.write(' ' + str(counter_features[target] + 8) + ':' + str(count_mention_symbols))

                    # Number of hashtag symbols
                    count_hashtag_symbols = df['Tweet'][i].count('#')
                    if count_hashtag_symbols > 0:
                        f.write(' ' + str(counter_features[target] + 9) + ':' + str(count_hashtag_symbols))

                    f.write('\n')

                f.write('3 0:1 ' + str(counter_features[target] + 9) + ':1')

        counter_features[target] += 10

    df_counter_features['Stylistic'] = list(counter_features.values())
    df_counter_features.to_csv(os.path.join(dir_vectors, 'stylistic', 'counter_features.tsv'), sep='\t', index=False)

def sentiment_features(dir_targets_src, dir_vectors, dir_targets_preprocessed, dir_dict):
    print('{0} Creating sentiment features'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    df_counter_features = pd.read_csv(os.path.join(dir_vectors, 'stylistic', 'counter_features.tsv'), sep='\t')

    counter_features = {}
    df_targets = df_counter_features['Target']
    df_stylistic = df_counter_features['Stylistic']
    for i in range(len(df_targets)):
        counter_features[df_targets[i]] = df_stylistic[i]

    with open(os.path.join(dir_dict, 'sentiment', 'AFINN-en-165.txt'), 'r', encoding='utf-8') as f:
        lines = filter(None, f.read().split('\n'))

    dict_afinn = {}
    for line in lines:
        line_splitted = line.split('\t')
        dict_afinn[line_splitted[0]] = int(line_splitted[1])

    with open(os.path.join(dir_dict, 'sentiment', 'Sentiment140-Lexicon-v0.1.txt'), 'r', encoding='utf-8') as f:
        lines = filter(None, f.read().split('\n'))

    dict_sentiment140 = {}
    for line in lines:
        line_splitted = line.split('\t')
        dict_sentiment140[line_splitted[0]] = float(line_splitted[1])

    with open(os.path.join(dir_dict, 'sentiment', 'NRC-Hashtag-Sentiment-Lexicon-v1.0.txt'), 'r', encoding='utf-8') as f:
        lines = filter(None, f.read().split('\n'))

    dict_nrc_hashtag = {}
    for line in lines:
        line_splitted = line.split('\t')
        dict_nrc_hashtag[line_splitted[0]] = float(line_splitted[1])

    with open(os.path.join(dir_dict, 'sentiment', 'MPQA_subjectivity_clues_hltemnlp05.tff'), 'r', encoding='utf-8') as f:
        lines = filter(None, f.read().split('\n'))

    dict_mpqa = {}
    for line in lines:
        line_splitted = line.split()
        if line_splitted[5].split('=')[1] == 'positive':
            if line_splitted[0].split('=')[1] == 'strongsubj':
                dict_mpqa[line_splitted[2].split('=')[1]] = 2
            else:
                dict_mpqa[line_splitted[2].split('=')[1]] = 1
        else:
            if line_splitted[0].split('=')[1] == 'strongsubj':
                dict_mpqa[line_splitted[2].split('=')[1]] = -2
            else:
                dict_mpqa[line_splitted[2].split('=')[1]] = -1

    dict_liu = {}

    with open(os.path.join(dir_dict, 'sentiment', 'Bing Liu_opinion-lexicon-English/positive-words.txt'), 'r', encoding='utf-8') as f:
        lines = filter(None, f.read().split('\n'))

    for line in lines:
        if not line.startswith(';'):
            dict_liu[line] = 1

    with open(os.path.join(dir_dict, 'sentiment', 'Bing Liu_opinion-lexicon-English/negative-words.txt'), 'r', encoding='utf-8') as f:
        lines = filter(None, f.read().split('\n'))

    for line in lines:
        if not line.startswith(';'):
            dict_liu[line] = -1

    targets = set()
    for file in os.listdir(dir_targets_src):
        if file.endswith('.tsv'):
            targets.add(file.split('_')[0])
    targets = sorted(list(targets))

    for target in targets:
        print('{0}  Building text representation model for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), target))

        filename = ['train', 'test']
        for file in filename:
            with open(os.path.join(dir_vectors, 'stylistic', target + '_' + file + '_vectors.txt'), 'r') as f:
                text_vectors = f.read().split('\n')
                text_vectors = text_vectors[:len(text_vectors) - 1]

            with open(os.path.join(dir_vectors, 'sentiment', target + '_' + file + '_vectors.txt'), 'w') as f:
                df_src = pd.read_csv(os.path.join(dir_targets_src, target + '_' + file + '.tsv'), sep='\t')
                df_preprocessed = pd.read_csv(os.path.join(dir_targets_preprocessed, target + '_' + file + '.tsv'), sep='\t')
                df_src['Tweet'] = df_src['Tweet'].fillna('')
                for i in range(len(df_src['Tweet'])):
                    tokens = tokenizeRawTweetText(df_src['Tweet'][i])

                    for j in range(len(tokens)):
                        tokens[j] = tokens[j].lower()

                    f.write(text_vectors[i])

                    # Get score using AFINN dictionary
                    afinn_score = 0
                    for token in tokens:
                        if token in dict_afinn.keys():
                            afinn_score += dict_afinn[token]

                    if afinn_score != 0:
                        f.write(' ' + str(counter_features[target]) + ':' + str(afinn_score))

                    # Get score using Sentiment140 dictionary
                    scores = []
                    for token in tokens:
                        if token in dict_sentiment140.keys():
                            scores.append(dict_sentiment140[token])
                        else:
                            scores.append(0)

                    count_pos_words = sum([1 if x > 0 else 0 for x in scores])
                    count_neg_words = sum([1 if x < 0 else 0 for x in scores])

                    if min(scores) != 0:
                        f.write(' ' + str(counter_features[target] + 1) + ':' + str(min(scores)))
                    if max(scores) != 0:
                        f.write(' ' + str(counter_features[target] + 2) + ':' + str(max(scores)))
                    if sum([x if x > 0 else 0 for x in scores]) != 0:
                        f.write(' ' + str(counter_features[target] + 3) + ':' + str(sum([x if x > 0 else 0 for x in scores])))
                    if sum([x if x < 0 else 0 for x in scores]) != 0:
                        f.write(' ' + str(counter_features[target] + 4) + ':' + str(sum([x if x < 0 else 0 for x in scores])))
                    if sum(scores) != 0:
                        f.write(' ' + str(counter_features[target] + 5) + ':' + str(sum(scores)))
                    if count_pos_words != 0:
                        f.write(' ' + str(counter_features[target] + 6) + ':' + str(count_pos_words))
                    if count_neg_words != 0:
                        f.write(' ' + str(counter_features[target] + 7) + ':' + str(count_neg_words))
                    if count_pos_words != 0:
                        f.write(' ' + str(counter_features[target] + 8) + ':' + str(count_pos_words / len(tokens)))
                    if count_neg_words != 0:
                        f.write(' ' + str(counter_features[target] + 9) + ':' + str(count_neg_words / len(tokens)))

                    # Get score using NRC Hashtag dictionary
                    scores = []
                    for token in tokens:
                        if token in dict_nrc_hashtag.keys():
                            scores.append(dict_nrc_hashtag[token])
                        else:
                            scores.append(0)

                    count_pos_words = sum([1 if x > 0 else 0 for x in scores])
                    count_neg_words = sum([1 if x < 0 else 0 for x in scores])

                    if min(scores) != 0:
                        f.write(' ' + str(counter_features[target] + 10) + ':' + str(min(scores)))
                    if max(scores) != 0:
                        f.write(' ' + str(counter_features[target] + 11) + ':' + str(max(scores)))
                    if sum([x if x > 0 else 0 for x in scores]) != 0:
                        f.write(' ' + str(counter_features[target] + 12) + ':' + str(sum([x if x > 0 else 0 for x in scores])))
                    if sum([x if x < 0 else 0 for x in scores]) != 0:
                        f.write(' ' + str(counter_features[target] + 13) + ':' + str(sum([x if x < 0 else 0 for x in scores])))
                    if sum(scores) != 0:
                        f.write(' ' + str(counter_features[target] + 14) + ':' + str(sum(scores)))
                    if count_pos_words != 0:
                        f.write(' ' + str(counter_features[target] + 15) + ':' + str(count_pos_words))
                    if count_neg_words != 0:
                        f.write(' ' + str(counter_features[target] + 16) + ':' + str(count_neg_words))
                    if count_pos_words != 0:
                        f.write(' ' + str(counter_features[target] + 17) + ':' + str(count_pos_words / len(tokens)))
                    if count_neg_words != 0:
                        f.write(' ' + str(counter_features[target] + 18) + ':' + str(count_neg_words / len(tokens)))

                    # Get score using MPQA dictionary
                    scores = []
                    for token in tokens:
                        if token in dict_mpqa.keys():
                            scores.append(dict_mpqa[token])
                        else:
                            scores.append(0)

                    count_pos_words = sum([1 if x > 0 else 0 for x in scores])
                    count_neg_words = sum([1 if x < 0 else 0 for x in scores])

                    if count_pos_words != 0:
                        f.write(' ' + str(counter_features[target] + 19) + ':' + str(count_pos_words))
                    if count_neg_words != 0:
                        f.write(' ' + str(counter_features[target] + 20) + ':' + str(count_neg_words))
                    if count_pos_words != 0:
                        f.write(' ' + str(counter_features[target] + 21) + ':' + str(count_pos_words / len(tokens)))
                    if count_neg_words != 0:
                        f.write(' ' + str(counter_features[target] + 22) + ':' + str(count_neg_words / len(tokens)))

                    if sum(scores) > 2 or sum(scores) < -2:
                        f.write(' ' + str(counter_features[target] + 23) + ':1')

                    # Get score using Bing Liu dictionary
                    scores = []
                    for token in tokens:
                        if token in dict_liu.keys():
                            scores.append(dict_liu[token])
                        else:
                            scores.append(0)

                    count_pos_words = sum([1 if x > 0 else 0 for x in scores])
                    count_neg_words = sum([1 if x < 0 else 0 for x in scores])

                    if count_pos_words != 0:
                        f.write(' ' + str(counter_features[target] + 24) + ':' + str(count_pos_words))
                    if count_neg_words != 0:
                        f.write(' ' + str(counter_features[target] + 25) + ':' + str(count_neg_words))
                    if count_pos_words != 0:
                        f.write(' ' + str(counter_features[target] + 26) + ':' + str(count_pos_words / len(tokens)))
                    if count_neg_words / len(tokens) != 0:
                        f.write(' ' + str(counter_features[target] + 27) + ':' + str(count_neg_words / len(tokens)))

                    f.write('\n')

                f.write('3 0:1 ' + str(counter_features[target] + 27) + ':1')

        counter_features[target] += 28

    df_counter_features['Sentiment'] = list(counter_features.values())
    df_counter_features.to_csv(os.path.join(dir_vectors, 'sentiment', 'counter_features.tsv'), sep='\t', index=False)

def word2vec(dir_targets_src, dir_vectors):
    print('{0} Creating word2vec features'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    model = gensim.downloader.load('word2vec-google-news-300')

    df_counter_features = pd.read_csv(os.path.join(dir_vectors, 'sentiment', 'counter_features.tsv'), sep='\t')

    counter_features = {}
    df_targets = df_counter_features['Target']
    df_sentiment = df_counter_features['Sentiment']
    for i in range(len(df_targets)):
        counter_features[df_targets[i]] = df_sentiment[i]

    targets = set()
    for file in os.listdir(dir_targets_src):
        if file.endswith('.tsv'):
            targets.add(file.split('_')[0])
    targets = sorted(list(targets))

    for target in targets:
        print('{0}  Building text representation model for {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), target))

        filename = ['train', 'test']
        for file in filename:
            with open(os.path.join(dir_vectors, 'sentiment', target + '_' + file + '_vectors.txt'), 'r') as f:
                text_vectors = f.read().split('\n')
                text_vectors = text_vectors[:len(text_vectors) - 1]

            with open(os.path.join(dir_vectors, 'word2vec', target + '_' + file + '_vectors.txt'), 'w') as f:
                df = pd.read_csv(os.path.join(dir_targets_src, target + '_' + file + '.tsv'), sep='\t')
                df['Tweet'] = df['Tweet'].fillna('')
                for i in range(len(df['Tweet'])):
                    tokens = tokenizeRawTweetText(df['Tweet'][i])

                    vector_res = np.array([0 for j in range(300)])
                    for token in tokens:
                        if token in model.vocab:
                            vector_res = np.add(vector_res, model[token])
                    vector_res = vector_res / len(tokens)

                    f.write(text_vectors[i] + ' ' + ' '.join(str(counter_features[target] + j) + ':' + str(vector_res[j]) for j in range(len(vector_res))) + '\n')

                f.write('3 0:1 ' + str(counter_features[target] + 299) + ':0.1')

        counter_features[target] += 300

    df_counter_features['Word2vec'] = list(counter_features.values())
    df_counter_features.to_csv(os.path.join(dir_vectors, 'word2vec', 'counter_features.tsv'), sep='\t', index=False)

def run(path, dir_input):
    if len(path) == 0:
        path = '..'
    else:
        path = ''

    if len(dir_input) == 0:
        dir_targets_src = os.path.join(path, 'data', 'targets_src')
    else:
        dir_targets_src = dir_input

    dir_vectors = os.path.join(path, 'data', 'vectors')
    dir_dict = os.path.join(path, 'data', 'dictionaries')
    dir_target_dict = os.path.join(path, 'data', 'target_dictionaries')
    dir_targets_preprocessed = os.path.join(path, 'data', 'targets_preprocessed')
    dir_targets_for_stance_indicative_features = os.path.join(path, 'data', 'targets_preprocessed_for_stance_indicative_features')

    ngrams_features(dir_targets_src, dir_target_dict, dir_vectors, dir_targets_preprocessed)
    dependency_features(dir_targets_src, dir_target_dict, dir_vectors)
    linguistic_features(dir_targets_src, dir_vectors, dir_targets_preprocessed, dir_dict)
    target_features(dir_targets_src, dir_vectors)
    stance_indicative_features(dir_targets_src, dir_target_dict, dir_vectors, dir_targets_for_stance_indicative_features)
    stylistic_features(dir_targets_src, dir_vectors)
    sentiment_features(dir_targets_src, dir_vectors, dir_targets_preprocessed, dir_dict)
    word2vec(dir_targets_src, dir_vectors)


if __name__ == '__main__':
    run('', '')
