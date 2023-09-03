import pandas as pd
import numpy as np
import scipy.stats as stats
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from pathlib import Path
from ganblr.models import GANBLR

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC

# log to file and to console
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        # log to stdout
        logging.StreamHandler(),
        # log to file with timestamp name
        logging.FileHandler(
            f"logs/analysis_results_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
        ),
    ],
)


class Dataset:
    def __init__(self, name: str, data: pd.DataFrame):
        self.name = name
        self.data = data
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.num_features = None
        self.unique_classes = None
        self._ordinal_encoder = OrdinalEncoder()
        self._label_encoder = LabelEncoder()

        self._model = None

        self.synth_data = None

    def __str__(self):
        return f"Dataset: {self.name}\n{self.data.head()}"


def read_in_test_data():
    paths = Path("./datasets")

    datasets_paths = []
    for file in paths.iterdir():
        if file.suffix == ".csv":
            datasets_paths.append(Dataset(file.stem, pd.read_csv(file)))

    frames = []
    for dset in datasets_paths:
        d = pd.read_csv(f"./datasets/{dset.name}.csv")
        frames.append((dset.name, d))

    return frames


def convert_datasets(
    encode_to_categorical: bool,
    datasets: list,
    prune_high_cardinality: int = 0,
    max_bins: int = 10,
):
    conv_datasets = []

    for d in datasets:
        logging.info(f"Converting dataset {d[0]}")

        dataset_class = Dataset(d[0], d[1])

        # get features as all columns except last
        X = dataset_class.data.iloc[:, :-1]
        y = dataset_class.data.iloc[:, -1]

        if encode_to_categorical:
            # if y is not already categorical, bin it then encode it
            if y.dtype == np.float64 or y.dtype == np.int64:
                y = pd.cut(y, bins=np.min([y.nunique(), max_bins]), labels=False)

            dataset_class.y = dataset_class._label_encoder.fit_transform(y).astype(int)
            # dataset_class.y = y

            # encode continuous features to categorical with max_bins
            for col in X.columns:
                if X[col].dtype == np.float64 or X[col].dtype == np.int64:
                    if X[col].nunique() > max_bins:
                        X[col] = pd.cut(X[col], bins=max_bins, labels=False)
                    else:
                        X[col] = pd.cut(
                            X[col],
                            bins=np.min([X[col].nunique(), max_bins]),
                            labels=False,
                        )

            # drop columns where unique values exceed high cardinality threshold
            if prune_high_cardinality > 0:
                X = X.loc[:, X.nunique() < prune_high_cardinality]

            dataset_class.X = dataset_class._ordinal_encoder.fit_transform(X).astype(
                int
            )

            dataset_class.num_features = dataset_class.X.shape[1]

            # get sum of number of unique classes in all columns
            dataset_class.unique_classes = sum([X[col].nunique() for col in X.columns])

            # split into train and test
            (
                dataset_class.X_train,
                dataset_class.X_test,
                dataset_class.y_train,
                dataset_class.y_test,
            ) = train_test_split(
                dataset_class.X, dataset_class.y, test_size=0.2, random_state=42
            )

        else:
            dataset_class.X = X

        logging.info(f"Dataset {d[0]} converted")
        logging.info(
            f"Dataset {d[0]} has {np.unique(dataset_class.y)} unique target classes"
        )
        logging.info(f"Dataset {d[0]} has {dataset_class.num_features} features")
        logging.info(
            f"Dataset {d[0]} has {dataset_class.unique_classes} unique classes"
        )

        logging.info(f"Dataset {d[0]} has {dataset_class.X.shape[0]} rows")

        conv_datasets.append(dataset_class)

    return conv_datasets


def train_ganblr(dataset: Dataset, k: int = 2):
    model = GANBLR()
    model.fit(dataset.X_train, dataset.y_train, k=k)

    dataset._model = model

    return model


def get_trtr_metrics(dataset: Dataset):
    logging.info(f"Getting metrics for dataset {dataset.name}")

    synth_X = dataset.synth_data.iloc[:, :-1]

    # select y and convert to int
    synth_y = dataset.synth_data.iloc[:, -1].astype(int)

    # train adaboost, random forest and svm classifier on real data
    adaboost = AdaBoostClassifier()
    adaboost.fit(dataset.X_train, dataset.y_train)
    adaboost_acc = adaboost.score(dataset.X_test, dataset.y_test)
    logging.info(
        f"Dataset {dataset.name} - Adaboost accuracy on real data: {adaboost_acc}"
    )

    random_forest = RandomForestClassifier()
    random_forest.fit(dataset.X_train, dataset.y_train)
    random_forest_acc = random_forest.score(dataset.X_test, dataset.y_test)
    logging.info(
        f"Dataset {dataset.name} - Random Forest accuracy on real data: {random_forest_acc}"
    )

    svm = SVC()
    svm.fit(dataset.X_train, dataset.y_train)
    svm_acc = svm.score(dataset.X_test, dataset.y_test)
    logging.info(f"Dataset {dataset.name} - SVM accuracy on real data: {svm_acc}")

    # train adaboost, random forest and svm classifier on synthetic data
    adaboost_synth = AdaBoostClassifier()
    adaboost_synth.fit(synth_X, synth_y)
    adaboost_synth_acc = adaboost_synth.score(dataset.X_test, dataset.y_test)
    logging.info(
        f"Dataset {dataset.name} - Adaboost accuracy on synthetic data: {adaboost_synth_acc}"
    )

    random_forest_synth = RandomForestClassifier()
    random_forest_synth.fit(synth_X, synth_y)
    random_forest_synth_acc = random_forest_synth.score(dataset.X_test, dataset.y_test)
    logging.info(
        f"Dataset {dataset.name} - Random Forest accuracy on synthetic data: {random_forest_synth_acc}"
    )

    # svm_synth = SVC()
    # svm_synth.fit(synth_X, synth_y)
    # svm_synth_acc = svm_synth.score(dataset.X_test, dataset.y_test)
    # logging.info(
    #     f"Dataset {dataset.name} - SVM accuracy on synthetic data: {svm_synth_acc}"
    # )


def get_distance_metrics(dataset):
    """Gets wasserstein, chi squared and kolmogorov smirnov distance metrics for real and synthetic data"""

    logging.info(f"Getting distance metrics for dataset {dataset.name}")

    # get real data
    real_data = dataset.data.iloc[:, :-1]

    # get synthetic data
    synth_data = dataset.synth_data.iloc[:, :-1]

    # get wasserstein distance
    wasserstein = stats.wasserstein_distance(real_data, synth_data)

    # get chi squared distance
    chi_squared = stats.chisquare(real_data, synth_data)

    # get kolmogorov smirnov distance
    kolmogorov_smirnov = stats.ks_2samp(real_data, synth_data)

    # log all metrics
    logging.info(f"Dataset {dataset.name} - Wasserstein distance: {wasserstein}")
    logging.info(f"Dataset {dataset.name} - Chi squared distance: {chi_squared}")
    logging.info(
        f"Dataset {dataset.name} - Kolmogorov smirnov distance: {kolmogorov_smirnov}"
    )


def get_cramers_v(dataset: Dataset):
    """Gets cramers v metric for real and synthetic data"""

    logging.info(f"Getting Cramers V metric for dataset {dataset.name}")

    # get real data
    real_data = dataset.data.iloc[:, :-1]

    # get synthetic data
    synth_data = dataset.synth_data.iloc[:, :-1]

    # get cramers v
    cramers_v = stats.chi2_contingency(real_data, synth_data)

    # log all metrics
    logging.info(f"Dataset {dataset.name} - Cramers V: {cramers_v}")


def get_sdv_metrics(dataset: Dataset):
    """Calculates the evaluation metrics using the SDV library"""


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

    readied_datasets = convert_datasets(
        encode_to_categorical=True, datasets=datasets, prune_high_cardinality=50
    )

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
        train_ctgan(d, k=0)
        logging.info(f"Model trained on dataset {d.name}")
        logging.info(f"Generating synthetic data for dataset {d.name}")


if __name__ == "__main__":
    main()
