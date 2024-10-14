import logging
import os


def log(log_dirname, log_filename='', exp_id='', is_cover=None, is_iterate=None):
    project_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), './..'))
    log_path = project_path + '/' + log_dirname
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    log_save = log_path + '/' + log_filename + '_' + exp_id + '.log'
    if is_cover:
        if os.path.exists(log_save):
            os.remove(log_save)
    if is_iterate:
        while 1:
            if os.path.exists(log_save):
                exp_id += 1
                log_save = log_path + '/' + log_filename + '_' + str(exp_id) + '.log'
                continue
            fh = logging.FileHandler(log_save, encoding='utf8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            return logger, exp_id

    fh = logging.FileHandler(log_save, encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

