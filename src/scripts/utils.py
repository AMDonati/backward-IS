import logging
import os
import csv


def write_to_csv(output_dir, dic):
    """Write a python dic to csv."""
    with open(output_dir, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dic.items():
            writer.writerow([key, value])


def create_logger(out_file_log):
    logging.basicConfig(filename=out_file_log, level=logging.INFO)
    # create logger
    logger = logging.getLogger('training log')
    logger.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger

def saving_training_history(keys, values, output_path, csv_fname, logger, start_epoch):
    history = dict(zip(keys, values))
    baseline_history_fn = output_path + '/' + csv_fname
    if os.path.isdir(baseline_history_fn):
        logger.info("saving the history from the restored ckpt # {} in a new csv file...".format(start_epoch))
        baseline_history_fn = output_path + '/' + 'baseline_history_from_ckpt{}.csv'.format(start_epoch)
    write_to_csv(baseline_history_fn, history)
    logger.info('saving loss and metrics information...')