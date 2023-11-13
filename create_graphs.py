import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

SUPERSTORE_TRTR_LOGS = Path("new_logs/log_20231109-074854.csv")
CREDIT_RISK_TRTR_LOGS = Path("new_logs/log_20231109-204608.csv")
MUSHROOMS_TRTR_LOGS = Path("new_logs/log_20231110-183536.csv")


def create_graph_trtr_one_model(frame: pd.DataFrame, dataset_name: str):
    f = frame.copy()
    
    # drop rows where "Event" column is "sdv_eval"
    #f = f[f["Event"] != "sdv_eval"]

    # drop column "Dataset"
    f = f.drop(columns=["Dataset"])

    # set "Event" column every 4th, 5th and 6th row to "tstr"
    f.iloc[3::6, f.columns.get_loc("Event")] = "tstr"
    f.iloc[4::6, f.columns.get_loc("Event")] = "tstr"
    f.iloc[5::6, f.columns.get_loc("Event")] = "tstr"


    f_group = f.groupby(["Model", "Epochs", "K", "Test", "Metric", "Event", "Value"]).mean().reset_index()

    # query for "Test" == "adaboost"
    f_group_ada = f_group.query("Test == 'adaboost'")

    # plot the graph w
    fig, ax = plt.subplots(figsize=(10, 10))
    
    sns.catplot(
        data=f_group_ada,
        x="Epochs",
        y="Value",
        hue="K",
        col="Metric",
        row="Model",
        kind="bar",
        sharey=False,
        ax=ax,
    )

    # set y label to "Accuracy"
    ax.set(ylabel="Accuracy")

    plt.title("AdaBoost TRTR/TSTR Performance")
    
    plt.tight_layout()
    plt.savefig(f"./graphs/{dataset_name}_trtr_ada.png", dpi=300)

    f_group_random_forest = f_group.query("Test == 'random_forest'")

    # plot the graph w
    fig, ax = plt.subplots(figsize=(10, 10))

    sns.catplot(
        data=f_group_random_forest,
        x="Epochs",
        y="Value",
        hue="K",
        col="Metric",
        row="Model",
        kind="bar",
        sharey=False,
        ax=ax,
    )

    # set y label to "Accuracy"
    ax.set(ylabel="Accuracy")

    plt.title("Random Forest TRTR/TSTR Performance")
    plt.tight_layout()
    plt.savefig(f"./graphs/{dataset_name}_trtr_forest.png", dpi=300)

    f_group_svm = f_group.query("Test == 'svm'")

    # plot the graph w
    fig, ax = plt.subplots(figsize=(10, 10))

    sns.catplot(
        data=f_group_svm,
        x="Epochs",
        y="Value",
        hue="K",
        col="Metric",
        row="Model",
        kind="bar",
        sharey=False,
        ax=ax,
    )

    # set y label to "Accuracy"
    ax.set(ylabel="Accuracy")

    plt.title("SVM TRTR/TSTR Performance")
    plt.tight_layout()
    plt.savefig(f"./graphs/{dataset_name}_trtr_svm.png", dpi=300)


    f_group_sdv_eval = f_group.query("Test == 'sdv_eval'")

    print(f_group_sdv_eval)

    # plot the graph w
    fig, ax = plt.subplots(figsize=(10, 10))

    sns.catplot(
        data=f_group_sdv_eval,
        x="Epochs",
        y="Value",
        hue="K",
        col="Metric",
        row="Model",
        kind="bar",
        sharey=False,
        ax=ax,
    )

    # set y label to "Accuracy"
    ax.set(ylabel="Quality Score")

    plt.title("SDV Quality Performance")
    plt.tight_layout()
    plt.savefig(f"./graphs/{dataset_name}_quality.png", dpi=300)


superstore = pd.read_csv(SUPERSTORE_TRTR_LOGS, delimiter=";")
credit_risk = pd.read_csv(CREDIT_RISK_TRTR_LOGS, delimiter=";")
mushrooms = pd.read_csv(MUSHROOMS_TRTR_LOGS, delimiter=";")

create_graph_trtr_one_model(superstore, "superstore")
create_graph_trtr_one_model(credit_risk, "credit_risk")
create_graph_trtr_one_model(mushrooms, "mushrooms")