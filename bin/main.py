#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging

import lib
import resources


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    extract()
    transform()
    model()
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


def transform():
    logging.info('Begin transform')

    lib.archive_dataset_schemas('transform', locals(), globals())
    logging.info('End transform')
    pass


def model():
    logging.info('Begin model')

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
