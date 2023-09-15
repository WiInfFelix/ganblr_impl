import logging
from datetime import datetime


class CSVLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

        # config logger to print to console and logfile
        self.setLevel(logging.DEBUG)

        log_handler = logging.FileHandler("log_%s.csv" % datetime.now().strftime("%Y%m%d-%H%M%S"))
        # set format to csv
        log_handler.setFormatter(logging.Formatter("%(message)s"))
        self.addHandler(log_handler)

        log_handler = logging.StreamHandler()
        log_handler.setFormatter(logging.Formatter("%(message)s"))
        self.addHandler(log_handler)

    # given args, log them to the csv file
    def log_args(self, *args):
        self.info(",".join(map(str, args)))
        