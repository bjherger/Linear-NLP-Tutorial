import gzip
import logging
import tarfile
import os

import gensim
import requests

from lib import get_conf


def download_newsgroup():
    """
    Validate that newsgroup20 data set is available

      - Check if newsgroup20 data set is available
      - If newsgroup20 data set is not available:
        - Download files
        - Un-tar files

    :return: None
    :rtype: None
    """
    # TODO Docstring

    # Reference variables
    newsgroup_20_download_link = 'http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz'
    newsgroup_20_downloaded_path = '../resources/compressed/news20.tar.gz'

    logging.info('Attempting to either validate or download and extract newsgroup_20 data set from {}'.format(
        newsgroup_20_download_link))

    # Download and expand newsgroup 20, if necessary
    if not os.path.exists(get_conf('newsgroup_path')):
        logging.warn('newsgroup_path does not exist. Downloading and extracting data set')
        logging.info('Downloading newgroup 20 data set from: {}, to: {}'.format(newsgroup_20_download_link,
                                                                                newsgroup_20_downloaded_path))
        download_file(newsgroup_20_download_link, newsgroup_20_downloaded_path)
        logging.info('Expanding newgroup data set')
        tar = tarfile.open(newsgroup_20_downloaded_path)
        tar.extractall(os.path.dirname(get_conf('newsgroup_path')))
        tar.close()

    logging.info('Newsgroup dataset available at: {}'.format(os.path.dirname(get_conf('newsgroup_path'))))


def download_file(url, local_file_path):
    """
    Download the file at `url` in chunks, to the location at `local_file_path`
    :param url: URL to a file to be downloaded
    :type url: str
    :param local_file_path: Path to download the file to
    :type local_file_path: str
    :return: The path to the file on the local machine (same as input `local_file_path`)
    :rtype: str
    """

    # Reference variables
    chunk_count = 0

    # Create connection to the stream
    r = requests.get(url, stream=True)

    # Open output file
    with open(local_file_path, 'wb') as f:

        # Iterate through chunks of file
        for chunk in r.iter_content(chunk_size=1048576):

            logging.debug('Downloading chunk: {} for file: {}'.format(chunk_count, local_file_path))

            # If there is a chunk to write to file, write it
            if chunk:
                f.write(chunk)

            # Increase chunk counter
            chunk_count = chunk_count + 1

    r.close()
    return local_file_path
