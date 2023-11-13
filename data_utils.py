import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer


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
        self._feature_encoder = LabelEncoder()
        self._target_encoder = LabelEncoder()

        self._model = None

        self.synth_data = None

    def __str__(self):
        return f"Dataset: {self.name}\n{self.data.head()}"


def read_in_test_data():
    """Read in test data from datasets folder and return as list of tuples"""
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

        # if y is not already categorical, bin it then encode it
        if y.dtype == np.float64 or y.dtype == np.int64:
            y = pd.cut(y, bins=np.min([y.nunique(), max_bins]), labels=False)

        dataset_class.y = dataset_class._target_encoder.fit_transform(y).astype(int)

        # encode continuous features to categorical with max_bins
        for col in X.select_dtypes(include=[np.float64, np.int64]).columns:
            X[col] = pd.cut(
                X[col].values, bins=np.min([X[col].nunique(), max_bins]), labels=False
            )

        # drop columns where unique values exceed high cardinality threshold
        if prune_high_cardinality > 0:
            X = X.loc[:, X.nunique() < prune_high_cardinality]

        dataset_class.X = dataset_class._feature_encoder.fit_transform(X).astype(int)

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

        logging.info(f"Dataset {d[0]} converted")
        logging.info(
            f"Dataset {d[0]} has {np.unique(dataset_class.y, return_counts=True)} unique target classes"
        )
        logging.info(f"Dataset {d[0]} has {dataset_class.num_features} features")
        logging.info(
            f"Dataset {d[0]} has {dataset_class.unique_classes} unique classes"
        )

        logging.info(f"Dataset {d[0]} has {dataset_class.X.shape[0]} rows")

        logging.info(f"Dataset {d[0]} feature head:\n{dataset_class.X[:5, :]}")
        logging.info(f"Dataset {d[0]} target head:\n{dataset_class.y[:5]}")

        conv_datasets.append(dataset_class)

    return conv_datasets


def preprocess_superstore(frame: pd.DataFrame):
    preproc_frame = frame.copy()
    
    preproc_frame["Postal Code"] = round(preproc_frame["Postal Code"] / 1000) * 1000

    # convert postal code to string
    preproc_frame["Postal Code"] = preproc_frame["Postal Code"].astype(str)
    
    preproc_frame = preproc_frame.drop("City", axis=1)

    return preproc_frame


def preprocess_credit_risk(frame: pd.DataFrame):
    preproc_frame = frame.copy()

    preproc_frame = frame.dropna()

    # move loan status to last column
    preproc_frame = preproc_frame[
        [col for col in preproc_frame.columns if col != "loan_status"] + ["loan_status"]
    ]

    return preproc_frame


def preprocess_mushroom(frame: pd.DataFrame):
    preproc_frame = frame.copy()

    preproc_frame = frame.dropna()
    
    preproc_frame = preproc_frame[
        [col for col in preproc_frame.columns if col != "class"] + ["class"]
    ]

    return preproc_frame


def transfrom_dataframe_discrete(
    frame: pd.DataFrame,
    max_bins: int = 10,
    encode: str = "ordinal",
    strategy: str = "kmeans",
):
    enc_frame = frame.copy()
    encoders = {}

    for col_n, col_d in frame.items():
        if np.issubdtype(col_d, np.number):
            if col_d.nunique() <= max_bins:
                kbins = KBinsDiscretizer(
                    n_bins=col_d.nunique(), encode=encode, strategy=strategy
                )
            else:
                kbins = KBinsDiscretizer(n_bins=max_bins, encode=encode, strategy=strategy)
            trans_data = kbins.fit_transform(col_d.values.reshape(-1, 1))
            enc_frame[col_n] = trans_data
            encoders[col_n] = kbins
        else:
            le = LabelEncoder()
            trans_data = le.fit_transform(col_d.values)
            enc_frame[col_n] = trans_data
            encoders[col_n] = le

    return enc_frame, encoders
