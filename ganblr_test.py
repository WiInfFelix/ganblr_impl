import pandas as pd

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

EPOCHS = [10, 25, 50, 100, 150]  
K = [0, 1, 2, 3, 4, 5]  #

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

csv_logger = CSVLogger(timestamp=timestamp,
    fieldnames=["Event", "Model", "Epochs", "K", "Dataset", "Test", "Metric", "Value"]
)


SUPERSTORE_DF = pd.read_csv("datasets\SampleSuperstore.csv")
CREDIT_RISK_DF = pd.read_csv("datasets\credit_risk_dataset.csv")
MUSHROOMS_DF = pd.read_csv("datasets\mushrooms.csv")

SUPERSTORE_DF = preprocess_superstore(SUPERSTORE_DF)
CREDIT_RISK_DF = preprocess_credit_risk(CREDIT_RISK_DF)
MUSHROOMS_DF = preprocess_mushroom(MUSHROOMS_DF)

SUPERSTORE_DF_ENC, SUPERSTORE_ENCODERS = transfrom_dataframe_discrete(SUPERSTORE_DF)
CREDIT_RISK_DF_ENC, CREDIT_RISK_ENCODERS = transfrom_dataframe_discrete(CREDIT_RISK_DF)
MUSHROOMS_DF_ENC, MUSHROOMS_ENCODERS = transfrom_dataframe_discrete(MUSHROOMS_DF)

print(SUPERSTORE_DF_ENC.head())


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

for epochs in EPOCHS:
    for k in K:
        ganblr_superstore = GANBLR()
        ganblr_superstore.fit(X_super, y_super, epochs=epochs, k=k)

        # sample as many rows as the original dataset
        synth_data_super = pd.DataFrame(
            ganblr_superstore.sample(X_super.shape[0]),
            columns=SUPERSTORE_DF_ENC.columns,
        )

        # get metrics
        get_trtr_metrics(
            X_super_train,
            X_super_test,
            y_super_train,
            y_super_test,
            synth_data_super,
            "superstore",
            "GANBLR",
            csv_logger,
            epochs,
            k,
        )
        
        get_sdv_metrics(
            real_data=SUPERSTORE_DF_ENC,
            synth_data=synth_data_super,
            dataset_name="superstore",
            model="GANBLR",
            csv_logger=csv_logger,
            epochs=epochs,
            k=k,
            timestamp=timestamp,
        )

        ganblr_credit_risk = GANBLR()
        ganblr_credit_risk.fit(X_credit, y_credit, epochs=epochs, k=k)

        # sample as many rows as the original dataset
        synth_data_credit = pd.DataFrame(
            ganblr_credit_risk.sample(X_credit.shape[0]),
            columns=CREDIT_RISK_DF_ENC.columns,
        )

        get_trtr_metrics(
            X_credit_train,
            X_credit_test,
            y_credit_train,
            y_credit_test,
            synth_data_credit,
            "credit_risk",
            "GANBLR",
            csv_logger,
            epochs,
            k,
        )
        
        get_sdv_metrics(
            real_data=CREDIT_RISK_DF_ENC,
            synth_data=synth_data_credit,
            dataset_name="credit_risk",
            model="GANBLR",
            csv_logger=csv_logger,
            epochs=epochs,
            k=k,
            timestamp=timestamp,
        )
        
        
        ganblr_mushrooms = GANBLR()
        ganblr_mushrooms.fit(X_mushrooms, y_mushrooms, epochs=epochs, k=k)
        
        # sample as many rows as the original dataset
        synth_data_mushrooms = pd.DataFrame(
            ganblr_mushrooms.sample(X_mushrooms.shape[0]),
            columns=MUSHROOMS_DF.columns,
        )
        
        get_trtr_metrics(
            X_mushrooms_train,
            X_mushrooms_test,
            y_mushrooms_train,
            y_mushrooms_test,
            synth_data_mushrooms,
            "mushrooms",
            "GANBLR",
            csv_logger,
            epochs,
            k,
        )
        
        get_sdv_metrics(
            real_data=MUSHROOMS_DF_ENC,
            synth_data=synth_data_mushrooms,
            dataset_name="mushrooms",
            model="GANBLR",
            csv_logger=csv_logger,
            epochs=epochs,
            k=k,
            timestamp=timestamp,
        )
        
    superstore_metadata = SingleTableMetadata()
    superstore_metadata.detect_from_dataframe(data=SUPERSTORE_DF_ENC)
    superstore_ctgan = CTGANSynthesizer(superstore_metadata)
    superstore_ctgan.fit(SUPERSTORE_DF_ENC)
    
    synth_data_superstore_ctgan = pd.DataFrame(
        superstore_ctgan.sample(X_super.shape[0]),
        columns=SUPERSTORE_DF_ENC.columns,
    )
    
    get_trtr_metrics(
        X_super_train,
        X_super_test,
        y_super_train,
        y_super_test,
        synth_data_superstore_ctgan,
        "superstore",
        "CTGAN",
        csv_logger,
        epochs,
        0,
    )
    
    get_sdv_metrics(
        real_data=SUPERSTORE_DF_ENC,
        synth_data=synth_data_superstore_ctgan,
        dataset_name="superstore",
        model="CTGAN",
        csv_logger=csv_logger,
        epochs=epochs,
        k=0,
        timestamp=timestamp,
    )
    
    credit_risk_metadata = SingleTableMetadata()
    credit_risk_metadata.detect_from_dataframe(data=CREDIT_RISK_DF_ENC)
    credit_risk_ctgan = CTGANSynthesizer(credit_risk_metadata)
    credit_risk_ctgan.fit(CREDIT_RISK_DF_ENC)
    
    synth_data_credit_risk_ctgan = pd.DataFrame(
        credit_risk_ctgan.sample(X_credit.shape[0]),
        columns=CREDIT_RISK_DF_ENC.columns,
    )
    
    get_trtr_metrics(
        X_credit_train,
        X_credit_test,
        y_credit_train,
        y_credit_test,
        synth_data_credit_risk_ctgan,
        "credit_risk",
        "CTGAN",
        csv_logger,
        epochs,
        0,
    )
    
    get_sdv_metrics(
        real_data=CREDIT_RISK_DF_ENC,
        synth_data=synth_data_credit_risk_ctgan,
        dataset_name="credit_risk",
        model="CTGAN",
        csv_logger=csv_logger,
        epochs=epochs,
        k=0,
        timestamp=timestamp,
    )
    
    mushrooms_metadata = SingleTableMetadata()
    mushrooms_metadata.detect_from_dataframe(data=MUSHROOMS_DF_ENC)
    mushrooms_ctgan = CTGANSynthesizer(mushrooms_metadata)
    mushrooms_ctgan.fit(MUSHROOMS_DF_ENC)
    
    synth_data_mushrooms_ctgan = pd.DataFrame(
        mushrooms_ctgan.sample(X_mushrooms.shape[0]),
        columns=MUSHROOMS_DF_ENC.columns,
    )
    
    get_trtr_metrics(
        X_mushrooms_train,
        X_mushrooms_test,
        y_mushrooms_train,
        y_mushrooms_test,
        synth_data_mushrooms_ctgan,
        "mushrooms",
        "CTGAN",
        csv_logger,
        epochs,
        0,
    )
    
    get_sdv_metrics(
        real_data=MUSHROOMS_DF_ENC,
        synth_data=synth_data_mushrooms_ctgan,
        dataset_name="mushrooms",
        model="CTGAN",
        csv_logger=csv_logger,
        epochs=epochs,
        k=0,
        timestamp=timestamp,
    )
    
