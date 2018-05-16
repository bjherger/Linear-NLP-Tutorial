#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import glob
import itertools
import logging
import ntpath
import os

import numpy
import pandas
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer

import lib
import resources


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    observations = extract()
    observations = transform(observations)
    model(observations)
    load()
    pass


def extract():
    """
    Extract necessary data / resources from upstream. This method will:
     - Validate that newsgroup data set is available, and read in
     - Validate that text embeddings are available, and read in
     - Validate that text to embedding index lookup is available, and read in
    :return: observations, embedding_matrix, word_to_index
    :rtype: (pandas.DataFrame, numpy.array, dict)
    """

    logging.info('Begin extract')
    logging.info('Performing extract for batch: {}, from newgroup_path: {}'
                 .format(lib.get_batch_name(), lib.get_conf('newsgroup_path')))

    # Download resources

    # Confirm newsgroup data set is downloaded
    resources.download_newsgroup()

    # Extract resources from file system

    # Newsgroup20: Get list of all candidate documents
    glob_pattern = os.path.join(lib.get_conf('newsgroup_path'), '*', '*')
    logging.info('Searching for glob_pattern: {}'.format(glob_pattern))
    document_candidates = glob.glob(glob_pattern)

    # Newsgroup20: Create observations data set
    observations = pandas.DataFrame(document_candidates, columns=['document_path'])
    logging.info('Shape of observations data frame created from glob matches: {}'.format(observations.shape))

    # Newsgroup20: Re-order rows
    observations = observations.sample(frac=1)

    # Newsgroup20: Subset number of observations, if it's a test run
    if lib.get_conf('test_run'):
        logging.info('Reducing file size for test run')
        observations = observations.head(100)
        logging.info('Test run number of records: {}'.format(len(observations.index)))

    # Archive schema and return
    lib.archive_dataset_schemas('extract', locals(), globals())
    logging.info('End extract')
    return observations


def transform(observations):
    logging.info('Begin transform')

    # Transform newsgroup20 data set
    # Newsgroup20: Extract article filename from document path
    observations['filename'] = observations['document_path'].apply(lambda x: ntpath.basename(x))

    # Newsgroup20: Extract article category from document path
    observations['category'] = observations['document_path'].apply(lambda x: ntpath.basename(os.path.dirname(x)))

    # Newsgroup20: Extract article text (and strip article headers), from document path
    observations['text'] = observations['document_path'].apply(lambda x: lib.strip_header(open(x).readlines()))

    # Remove non-ascii characters
    observations['text'] = observations['text'].apply(lambda x: x.decode('ascii', errors='ignore'))

    # Newsgroup20: Convert text to normalized tokens. Unknown tokens will map to 'UNK'.
    observations['tokens'] = observations['text'].apply(simple_preprocess)

    # Newsgroup20: Create bigrams
    observations['bigrams'] = observations['text'].apply(lambda x: lib.find_ngrams(x, n=2))

    # Newsgroup20: Create modeling text
    observations['modeling_text_list'] = observations['tokens'] + observations['bigrams']
    observations['modeling_text'] = observations['modeling_text_list'].apply(lambda x: ' '.join(x))

    lib.archive_dataset_schemas('transform', locals(), globals())
    logging.info('End transform')
    return observations


def model(observations):
    logging.info('Begin model')

    # Resources
    vocabulary = list(itertools.chain.from_iterable(observations['modeling_text_list']))
    vectorizer = CountVectorizer(vocabulary=vocabulary)

    # Create train, test sets
    msk = numpy.random.rand(len(observations)) < 0.8
    train = observations[msk]
    test = observations[~msk]

    # Create X, y vectors
    X_train = vectorizer.fit_transform(train['modeling_text'])
    y_train = train['category']

    X_test = vectorizer.transform(test['modeling_text'])
    y_test = test['category']

    lib.archive_dataset_schemas('model', locals(), globals())
    logging.info('End model')
    pass


def load():
    logging.info('Begin load')

    lib.archive_dataset_schemas('load', locals(), globals())
    logging.info('End load')
    pass


# Main section
if __name__ == '__main__':
    main()
