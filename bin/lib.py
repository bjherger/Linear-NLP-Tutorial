import datetime
import logging
import os
import tempfile

import pandas
import yaml

# Global variables
CONFS = None
BATCH_NAME = None
BATCH_OUTPUT_FOLDER = None
TEMP_DIR = None


def load_confs(confs_path='../conf/conf.yaml'):
    """
    Load configurations from file.

     - If configuration file is available, load it
     - If configuraiton file is not available attempt to load configuration template

    Configurations are never explicitly validated.
    :param confs_path: Path to a configuration file, appropriately formatted for this application
    :type confs_path: str
    :return: Python native object, containing configuration names and values
    :rtype: dict
    """
    global CONFS

    if CONFS is None:

        try:
            logging.info('Attempting to load conf from path: {}'.format(confs_path))

            # Attempt to load conf from confPath
            CONFS = yaml.load(open(confs_path))

        except IOError:
            logging.warn('Unable to open user conf file. Attempting to run with default values from conf template')

            # Attempt to load conf from template path
            template_path = confs_path + '.template'
            CONFS = yaml.load(open(template_path))

    return CONFS


def get_conf(conf_name):
    """
    Get a configuration parameter by its name
    :param conf_name: Name of a configuration parameter
    :type conf_name: str
    :return: Value for that conf (no specific type information available)
    """
    return load_confs()[conf_name]


def get_batch_name():
    """
    Get the name of the current run. This is a unique identifier for each run of this application
    :return: The name of the current run. This is a unique identifier for each run of this application
    :rtype: str
    """
    global BATCH_NAME

    if BATCH_NAME is None:
        logging.info('Batch name not yet set. Setting batch name.')
        BATCH_NAME = str(datetime.datetime.utcnow()).replace(' ', '_').replace('/', '_').replace(':', '_')
        logging.info('Batch name: {}'.format(BATCH_NAME))
    return BATCH_NAME


def get_temp_dir():
    global TEMP_DIR
    if TEMP_DIR is None:
        TEMP_DIR = tempfile.mkdtemp(prefix='reddit_')
        logging.info('Created temporary directory: {}'.format(TEMP_DIR))
        print('Created temporary directory: {}'.format(TEMP_DIR))
    return TEMP_DIR


def get_batch_output_folder():
    global BATCH_OUTPUT_FOLDER
    if BATCH_OUTPUT_FOLDER is None:
        BATCH_OUTPUT_FOLDER = os.path.join(get_conf('load_path'), get_batch_name())
        os.mkdir(BATCH_OUTPUT_FOLDER)
        logging.info('Batch output folder: {}'.format(BATCH_OUTPUT_FOLDER))
    return BATCH_OUTPUT_FOLDER


def archive_dataset_schemas(step_name, local_dict, global_dict):
    """
    Archive the schema for all available Pandas DataFrames

     - Determine which objects in namespace are Pandas DataFrames
     - Pull schema for all available Pandas DataFrames
     - Write schemas to file

    :param step_name: The name of the current operation (e.g. `extract`, `transform`, `model` or `load`
    :param local_dict: A dictionary containing mappings from variable name to objects. This is usually generated by
    calling `locals`
    :type local_dict: dict
    :param global_dict: A dictionary containing mappings from variable name to objects. This is usually generated by
    calling `globals`
    :type global_dict: dict
    :return: None
    :rtype: None
    """
    logging.info('Archiving data set schema(s) for step name: {}'.format(step_name))

    # Reference variables
    data_schema_dir = get_conf('data_schema_dir')
    schema_output_path = os.path.join(data_schema_dir, step_name + '.csv')
    schema_agg = list()

    env_variables = dict()
    env_variables.update(local_dict)
    env_variables.update(global_dict)

    # Filter down to Pandas DataFrames
    data_sets = filter(lambda (k, v): type(v) == pandas.DataFrame, env_variables.iteritems())
    data_sets = dict(data_sets)

    header = pandas.DataFrame(columns=['variable', 'type', 'data_set'])
    schema_agg.append(header)

    for (data_set_name, data_set) in data_sets.iteritems():
        # Extract variable names
        logging.info('Working data_set: {}'.format(data_set_name))

        local_schema_df = pandas.DataFrame(data_set.dtypes, columns=['type'])
        local_schema_df['data_set'] = data_set_name

        schema_agg.append(local_schema_df)

    # Aggregate schema list into one data frame
    agg_schema_df = pandas.concat(schema_agg)

    # Write to file
    agg_schema_df.to_csv(schema_output_path, index_label='variable')


def strip_header(document_lines):
    """
    Newsgroup 20 documents often contain headers, which may include their classification. This (highly heuristic) method
    attempts to strip away those headers.

     - Iterate until we hit the first empty line
     - Keep all lines after the first empty line

    :param document_lines: An iterable, containing lines of text from a newsgroup 20 document
    :return: Lines of the newsgroup 20 document, with the header stripped.
    """

    first_empty_line = False

    output_agg = list()

    for line in document_lines:

        if line == '\n':
            first_empty_line = True

        if first_empty_line:
            output_agg.append(line)

    document_string = '\n'.join(output_agg)

    return document_string


def find_ngrams(input_list, n):
    # Courtesy http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
    ngrams = zip(*[input_list[i:] for i in range(n)])
    flattened_ngrams = map(lambda x: '_'.join(x), ngrams)
    return flattened_ngrams

