import logging
import os


def print_and_log_info(logger, string):
    logger.info(string)

def get_logger(file_path, name='train'):

    logger = logging.getLogger(name)
    hdlr = logging.FileHandler(file_path, mode='a')
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] %(levelname)s: %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger, hdlr


if __name__ == '__main__':
    path_1 = os.path.join(os.getcwd(), 'debugging', 'p1.log')
    logger, hand = get_logger(path_1, 'gaga')
    print_and_log_info(logger, 'gaga')
    

    path_2 = os.path.join(os.getcwd(), 'debugging', 'p2.log')
    logger2, hand2 = get_logger(path_2, 'gaga')
    print_and_log_info(logger2, 'gaga')
    logger.removeHandler(hand)
    logger2.removeHandler(hand2)