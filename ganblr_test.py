from time import sleep
import pandas as pd
import tracemalloc
# import numpy as np
from ganblr.models import GANBLR
from data_utils import (
    transfrom_dataframe_discrete,
    preprocess_superstore,
    preprocess_credit_risk,
    preprocess_mushroom
)
from logger_utils import CSVLogger
from sklearn.model_selection import train_test_split
from metric_utils import get_trtr_metrics, get_sdv_metrics
from datetime import datetime
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from pathlib import Path

import os
import gc

EPOCHS = [10, 25, 50, 100,  150]
K = [0,1]  #

overall_logfile = Path(f"./new_logs/log_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/' 

timestamp_id = datetime.now().strftime("%Y%m%d-%H%M%S")

with open(overall_logfile, "w") as f:
    # write with ; as delimiter
    f.write("Event;Model;Epochs;K;Dataset;Test;Metric;Value\n")

tracemalloc.start()

def process_dataset(dataset_name, X, y, df_enc, encoders, X_train, X_test, y_train, y_test, epochs, K, timestamp_id, overall_logfile):
    for epoch in epochs:
        for k in K:
            for i in range(1, 4):
                ganblr = GANBLR()
                ganblr.fit(X, y, epochs=epoch, k=k)

                # sample as many rows as the original dataset
                synth_data = pd.DataFrame(
                    ganblr.sample(X.shape[0]),
                    columns=df_enc.columns,
                )

                # decode the categorical columns
                synth_data_clear = synth_data.copy()
                for col in df_enc.columns:
                    synth_data_clear[col] = encoders[col].inverse_transform(
                        synth_data[[col]].astype(int)
                    )

                synth_data_clear.to_csv(f"./synth_data/{timestamp_id}_ganblr_synth_data_{dataset_name}_{epoch}_{k}_{i}.csv")

                # get metrics
                get_trtr_metrics(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    synth_data,
                    dataset_name,
                    "GANBLR",
                    overall_logfile,
                    epoch,
                    k,
                )

                get_sdv_metrics(
                    real_data=df_enc,
                    synth_data=synth_data,
                    dataset_name=dataset_name,
                    model="GANBLR",
                    overall_logfile=overall_logfile,
                    epochs=epoch,
                    k=k,
                    timestamp=timestamp_id,
                    i=i
                )

                del ganblr
                gc.collect()
        
        for i in range(1, 4):
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data=df_enc)
            ctgan = CTGANSynthesizer(metadata, epochs=epoch)
            ctgan.fit(df_enc)

            synth_data_ctgan = pd.DataFrame(
                ctgan.sample(X.shape[0]),
                columns=df_enc.columns,
            )

            synth_data_ctgan_clear = synth_data_ctgan.copy()
            for col in df_enc.columns:
                synth_data_ctgan_clear[col] = encoders[col].inverse_transform(
                    synth_data_ctgan[[col]].astype(int)
                )

            synth_data_ctgan_clear.to_csv(f"./synth_data/{timestamp_id}_ctgan_synth_data_{dataset_name}_{epoch}_{k}_{i}.csv")

            get_trtr_metrics(
                X_train,
                X_test,
                y_train,
                y_test,
                synth_data_ctgan,
                dataset_name,
                "CTGAN",
                overall_logfile,
                epoch,
                0,
            )

            get_sdv_metrics(
                real_data=df_enc,
                synth_data=synth_data_ctgan,
                dataset_name=dataset_name,
                model="CTGAN",
                csv_logger=overall_logfile,
                epochs=epoch,
                k=0,
                timestamp=timestamp_id,
                i=i
            )

            del ctgan
            gc.collect()





SUPERSTORE_PATH = Path("datasets/SampleSuperstore.csv")
CREDIT_RISK_PATH = Path("datasets/credit_risk_dataset.csv")
MUSHROOMS_PATH = Path("datasets/mushrooms.csv")

SUPERSTORE_DF = pd.read_csv(SUPERSTORE_PATH)
CREDIT_RISK_DF = pd.read_csv(CREDIT_RISK_PATH)
MUSHROOMS_DF = pd.read_csv(MUSHROOMS_PATH)

SUPERSTORE_DF = preprocess_superstore(SUPERSTORE_DF)
CREDIT_RISK_DF = preprocess_credit_risk(CREDIT_RISK_DF)
MUSHROOMS_DF = preprocess_mushroom(MUSHROOMS_DF)

SUPERSTORE_DF_ENC, SUPERSTORE_ENCODERS = transfrom_dataframe_discrete(SUPERSTORE_DF)
CREDIT_RISK_DF_ENC, CREDIT_RISK_ENCODERS = transfrom_dataframe_discrete(CREDIT_RISK_DF)
MUSHROOMS_DF_ENC, MUSHROOMS_ENCODERS = transfrom_dataframe_discrete(MUSHROOMS_DF)


# cast all columns to categorical
# SUPERSTORE_DF_ENC = SUPERSTORE_DF_ENC.astype("category")
# CREDIT_RISK_DF_ENC = CREDIT_RISK_DF_ENC.astype("category")
# MUSHROOMS_DF_ENC = MUSHROOMS_DF_ENC.astype("category")

X_super = SUPERSTORE_DF_ENC.drop("Profit", axis=1)
y_super = SUPERSTORE_DF_ENC["Profit"]

X_super_train, X_super_test, y_super_train, y_super_test = train_test_split(
    X_super, y_super, test_size=0.2, random_state=42
)


X_credit = CREDIT_RISK_DF_ENC.drop("loan_status", axis=1)
y_credit = CREDIT_RISK_DF_ENC["loan_status"]

X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(
    X_credit, y_credit, test_size=0.2, random_state=42
)

X_mushrooms = MUSHROOMS_DF_ENC.drop("class", axis=1)
y_mushrooms = MUSHROOMS_DF_ENC["class"]

X_mushrooms_train, X_mushrooms_test, y_mushrooms_train, y_mushrooms_test = train_test_split(
    X_mushrooms, y_mushrooms, test_size=0.2, random_state=42
)


process_dataset(
    "superstore",
    X_super,
    y_super,
    SUPERSTORE_DF_ENC,
    SUPERSTORE_ENCODERS,
    X_super_train,
    X_super_test,
    y_super_train,
    y_super_test,
    EPOCHS,
    K,
    timestamp_id,
    overall_logfile
)

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)


process_dataset(
    "credit_risk",
    X_credit,
    y_credit,
    CREDIT_RISK_DF_ENC,
    CREDIT_RISK_ENCODERS,
    X_credit_train,
    X_credit_test,
    y_credit_train,
    y_credit_test,
    EPOCHS,
    K,
    timestamp_id,
    overall_logfile
)

process_dataset(
    "mushrooms",
    X_mushrooms,
    y_mushrooms,
    MUSHROOMS_DF_ENC,
    MUSHROOMS_ENCODERS,
    X_mushrooms_train,
    X_mushrooms_test,
    y_mushrooms_train,
    y_mushrooms_test,
    EPOCHS,
    K,
    timestamp_id,
    overall_logfile
)