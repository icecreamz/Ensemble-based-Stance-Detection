from wordsegment import load, segment
from nltk.stem import WordNetLemmatizer
from twokenize import tokenizeRawTweetText
from datetime import datetime
import pandas as pd
import nltk
import string
import os

punctuations = [punct for punct in string.punctuation] + ['...', '\'\'', '..', '--']

def run(path, dir_input):
    if len(path) == 0:
        path = '..'
    else:
        path = ''

    if len(dir_input) == 0:
        dir_targets_src = os.path.join(path, 'data', 'targets_src')
    else:
        dir_targets_src = dir_input

    dir_targets_preprocessed = os.path.join(path, 'data', 'targets_preprocessed')
    dir_dict = os.path.join(path, 'data', 'dictionaries')

    load()
    lemmatizer = WordNetLemmatizer()

    # Load the dictionary of abbreviated words, hashtags and mentions
    dict_hashtags = {}
    with open(os.path.join(dir_dict, 'Dictionary of abbreviated words, hashtags and mentions.txt'), 'r') as f:
        lines = f.read().split('\n')
    for line in lines[1:]:
        parts = line.split('\t')
        dict_hashtags[parts[0].lower()] = parts[1].replace('# ', '').lower()

    # Load the Han-Baldwin dictionary
    dict_han = {}
    with open(os.path.join(dir_dict, 'corpus.tweet1.txt'), 'r') as f:
        lines = f.read().split('\n')
    for line in lines:
        parts = line.split('\t')
        if len(parts) == 2:
            if not parts[0] in dict_han.keys():
                dict_han[parts[0]] = parts[1]

    # Load the dictionary of stopwords
    with open(os.path.join(dir_dict, 'en_stopwords.txt'), 'r') as f:
        dict_stopwords = f.read().split('\n')
    # Load the dictionary of negations
    with open(os.path.join(dir_dict, 'en_negations.txt'), 'r') as f:
        dict_negations = f.read().split('\n')
    for word in dict_negations:
        dict_stopwords.remove(word)

    targets = set()
    for file in os.listdir(dir_targets_src):
        if file.endswith('.tsv'):
            targets.add(file.split('_')[0])
    targets = sorted(list(targets))

    for target in targets:
        print('{0} Processing {1}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), target))

        for filename in ['train', 'test']:
            df = pd.read_csv(os.path.join(dir_targets_src, target + '_' + filename + '.tsv'), sep='\t')
            for i in range(len(df['Tweet'])):
                df['Tweet'][i] = df['Tweet'][i].lower()

                # Twokenize tokenization
                tokens = tokenizeRawTweetText(df['Tweet'][i])

                # Replace words from dictionary of abbreviated words, hashtags and mentions
                for j in range(len(tokens)):
                    if tokens[j] in dict_hashtags.keys():
                        tokens[j] = dict_hashtags[tokens[j]]

                # Hashtag segmentation
                for j in range(len(tokens)):
                    if tokens[j].startswith('#'):
                        tokens[j] = ' '.join(word for word in segment(tokens[j]))

                # Replace words from Han-Baldwin dictionary
                for j in range(len(tokens)):
                    if tokens[j] in dict_han.keys():
                        tokens[j] = dict_han[tokens[j]]
                df['Tweet'][i] = ' '.join(tokens)

                # Get POS-tags
                tokens = df['Tweet'][i].split()
                tokens_pos = nltk.pos_tag(tokens)
                pos_tags = []
                for token in tokens_pos:
                    pos_tags.append(token[1])

                for j in range(len(tokens)):
                    if tokens[j].endswith('\'s'):
                        tokens[j] = tokens[j].replace('\'s', '')
                    elif tokens[j].endswith('\'ll'):
                        tokens[j] = tokens[j].replace('\'ll', '')

                # Remove punctuation and stopwords
                tokens_temp = []
                pos_tags_temp = []
                for j in range(len(tokens)):
                    if tokens[j] not in punctuations and tokens[j] not in dict_stopwords:
                        tokens_temp.append(tokens[j])
                        pos_tags_temp.append(pos_tags[j])
                tokens = tokens_temp
                pos_tags = pos_tags_temp

                # Lemmatization
                tokens_norm = [lemmatizer.lemmatize(token, pos='v') for token in tokens]

                for j in range(len(tokens_norm)):
                    tokens_norm[j] = tokens_norm[j] + '_' + pos_tags[j]

                df['Tweet'][i] = ' '.join(tokens_norm)
            df.to_csv(os.path.join(dir_targets_preprocessed, target + '_' + filename + '.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    run('', '')
