import csv
from datetime import datetime


class CSVLogger:
    def __init__(self, fieldnames, timestamp, filename_prefix="log"):
        self.timestamp = timestamp
        self.filename = f"./new_logs/{filename_prefix}_{timestamp}.csv"
        self.fieldnames = fieldnames
        self.file = open(self.filename, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, **kwargs):
        self.writer.writerow(kwargs)
        self.file.flush()
        print(", ".join([f"{k}: {v}" for k, v in kwargs.items()]))

    def __del__(self):
        self.file.close()
