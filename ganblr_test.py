import pandas as pd

# import numpy as np
from ganblr.models import GANBLR
from data_utils import (
    transfrom_dataframe_discrete,
    preprocess_superstore,
    preprocess_credit_risk,
)
from logger_utils import CSVLogger
from sklearn.model_selection import train_test_split
from metric_utils import get_trtr_metrics


EPOCHS = [10, 25, 50, 100, 150]  
K = [0, 1, 2]  # , 3, 4, 5

csv_logger = CSVLogger(
    ["Event", "Model", "Epochs", "K", "Dataset", "Test", "Metric", "Value"]
)


SUPERSTORE_DF = pd.read_csv("datasets\SampleSuperstore.csv")
CREDIT_RISK_DF = pd.read_csv("datasets\credit_risk_dataset.csv")
# CALIFORNIA_HOUSING_DF = pd.read_csv("datasets\california_housing.csv")

SUPERSTORE_DF = preprocess_superstore(SUPERSTORE_DF)
CREDIT_RISK_DF = preprocess_credit_risk(CREDIT_RISK_DF)
# CALIFORNIA_HOUSING_DF = preprocess_california_housing(CALIFORNIA_HOUSING_DF)

SUPERSTORE_DF_ENC, SUPERSTORE_ENCODERS = transfrom_dataframe_discrete(SUPERSTORE_DF)
CREDIT_RISK_DF_ENC, CREDIT_RISK_ENCODERS = transfrom_dataframe_discrete(CREDIT_RISK_DF)

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

for epochs in EPOCHS:
    for k in K:
        # ganblr_superstore = GANBLR()
        # ganblr_superstore.fit(X_super, y_super, epochs=epochs, k=k)

        # # print(ganblr_superstore.evaluate(X_super, y_super))

        # # sample as many rows as the original dataset
        # synth_data_super = pd.DataFrame(
        #     ganblr_superstore.sample(X_super.shape[0]),
        #     columns=SUPERSTORE_DF_ENC.columns,
        # )

        # # get metrics
        # get_trtr_metrics(
        #     X_super_train,
        #     X_super_test,
        #     y_super_train,
        #     y_super_test,
        #     synth_data_super,
        #     "superstore",
        #     "GANBLR",
        #     csv_logger,
        #     epochs,
        #     k,
        # )

        ganblr_credit_risk = GANBLR()
        ganblr_credit_risk.fit(X_credit, y_credit, epochs=epochs, k=k)

        # sample as many rows as the original dataset
        synth_data_credit = pd.DataFrame(
            ganblr_credit_risk.sample(X_credit.shape[0]),
            columns=CREDIT_RISK_DF_ENC.columns,
        )

        # get metrics
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
