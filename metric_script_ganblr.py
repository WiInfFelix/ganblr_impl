import pandas as pd
import datetime
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from ganblr.models import GANBLR

# log to file and to console
import logging

from data_utils import read_in_test_data, convert_datasets, Dataset
from metric_utils import get_trtr_metrics

logging.basicConfig(
    level=logging.INFO,
    # format so it can be copied into a spreadsheet
    format="%(asctime)s, %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    handlers=[
        # log to stdout
        logging.StreamHandler(),
        # log to file with timestamp name
        logging.FileHandler(
            f"logs/analysis_results_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
        ),
    ],
)

# Constants for training
MAX_CARDINALITY = 50
MAX_BINS = 10
EPOCHS = [10, 25, 50, 100, 150]
K = [0, 1, 2]


def train_ganblr(dataset: Dataset, k: int = 2):
    model = GANBLR()
    model.fit(dataset.X_train, dataset.y_train, k=k, epochs=3)

    dataset._model = model

    return model


def train_ctgan(dataset: Dataset):
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata

    metadata = SingleTableMetadata(dataset.data)

    synthesizer = CTGANSynthesizer(metadata=metadata)
    synthesizer.fit(dataset.data)

    logging.info(f"Trianing CTGAN on dataset {dataset.name}")


def main():
    logging.info("Starting analysis")
    datasets = read_in_test_data()

    readied_datasets = convert_datasets(datasets=datasets, prune_high_cardinality=50)

    # loop 10 times
    for i in range(10):
        for d in readied_datasets:
            logging.info(f"Training GANBLR on dataset {d.name}")
            train_ganblr(d, k=0)
            logging.info(f"Model trained on dataset {d.name}")
            logging.info(f"Generating synthetic data for dataset {d.name}")
            # get synthetic data as
            d.synth_data = pd.DataFrame(
                d._model.sample(d.num_features, d.X_train.shape[0])
            )

            logging.info(f"Starting analysis on dataset {d.name}")
            get_trtr_metrics(d)
            # get_distance_metrics(d)
            # get_cramers_v(d)

    logging.info("Analysis complete for GANBLR")

    for d in readied_datasets:
        logging.info(f"Training CTGAN on dataset {d.name}")
        train_ctgan(d)
        logging.info(f"Model trained on dataset {d.name}")
        logging.info(f"Generating synthetic data for dataset {d.name}")


if __name__ == "__main__":
    main()
