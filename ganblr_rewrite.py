from datetime import datetime
import pandas as pd
import ganblr as gb
from pathlib import Path
import logging as l
import logging_utils as lu
from data_utils import convert_superstore

SUPERSTORE_PATH = Path('.\datasets\SampleSuperstore.csv')
CREDIT_RISK_PATH = Path('.\datasets\credit_risk_dataset.csv')


EPOCHS = [5, 10, 25, 50, 100 , 150]
K = [0, 1, 2, 3, 4, 5]
MAX_BINS = [10, 20, 50, None]


SUPERSTORE_DF = pd.read_csv(SUPERSTORE_PATH)
CREDIT_RISK_DF = pd.read_csv(CREDIT_RISK_PATH)

logger = lu.CSVLogger('CSVLogger', level=l.DEBUG)

logger.log_args("Eventype", "Epochs", "K", "Dataset", "Metric", "Value")


for i in MAX_BINS:

    SUPERSTORE_DISCRETIZED = convert_superstore(SUPERSTORE_DF, max_bins=i)
    CREDIT_RISK_DISCRETIZED = convert_superstore(CREDIT_RISK_DF, max_bins=i)
    
    for j in EPOCHS:
        for z in K:
            logger.log_args("INFO", j, z, "Superstore", "FID", gb.evaluate(SUPERSTORE_DISCRETIZED, epochs=j, k=z))