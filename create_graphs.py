import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from pathlib import Path

sns.set_theme(style="whitegrid")

SUPERSTORE_TRTR_LOGS = Path("new_logs/log_20231109-074854.csv")
CREDIT_RISK_TRTR_LOGS = Path("new_logs/log_20231109-204608.csv")
MUSHROOMS_TRTR_LOGS = Path("new_logs/log_20231110-183536.csv")


def create_super_table_for_dataset(frame: pd.DataFrame, dataset_name: str):
    f = frame.copy()
    f = f.drop(columns=["Dataset"])

    # drop all where Metric == sdv_eval
    f = f[f["Test"] != "sdv_eval"]

    f.iloc[3::6, f.columns.get_loc("Event")] = "tstr"
    f.iloc[4::6, f.columns.get_loc("Event")] = "tstr"
    f.iloc[5::6, f.columns.get_loc("Event")] = "tstr"

    f_group = f.groupby(["Model", "Epochs", "K", "Event", "Test", "Metric"])[
        "Value"
    ].mean()

    # write to latex with caption
    f_group.to_latex(
        f"tables/{dataset_name}.tex",
        caption=f"Results for {dataset_name}",
        longtable=True,
    )


def create_comparison_accuracy_test_dataset(
    frame: pd.DataFrame, dataset_name: str, metric: str
):
    f = frame.copy()

    # drop all where Metric != adaboost
    f = f[f["Test"] == metric]

    # drop col Dataset
    f = f.drop(columns=["Dataset"])

    f.iloc[3::6, f.columns.get_loc("Event")] = "tstr"
    f.iloc[4::6, f.columns.get_loc("Event")] = "tstr"
    f.iloc[5::6, f.columns.get_loc("Event")] = "tstr"

    ctgan_frame = f[f["Model"] == "CTGAN"]
    ganblr_frame_k_0 = f[(f["Model"] == "GANBLR") & (f["K"] == 0)]
    ganblr_frame_k_1 = f[(f["Model"] == "GANBLR") & (f["K"] == 1)]

    fig, ax = plt.subplots(ncols=3, figsize=(12, 5))

    # plot for GANBLR with k=0, hue by Event
    sns.barplot(
        data=ganblr_frame_k_0,
        x="Epochs",
        y="Value",
        hue="Event",
        ax=ax[0],
        palette="Set2",
    )

    # plot for GANBLR with k=1, hue by Event
    sns.barplot(
        data=ganblr_frame_k_1,
        x="Epochs",
        y="Value",
        hue="Event",
        ax=ax[1],
        palette="Set2",
    )

    # plot for CTGAN, hue by Event
    sns.barplot(
        data=ctgan_frame,
        x="Epochs",
        y="Value",
        hue="Event",
        ax=ax[2],
        palette="Set2",
    )

    # set Title
    ax[0].set_title("GANBLR with K=0")
    ax[1].set_title("GANBLR with K=1")
    ax[2].set_title("CTGAN")

    # set Figure Title
    fig.suptitle(f"Accuracy for {dataset_name} with {metric} as Test")

    # set Y-Label f ax0, remove for ax1 and ax2
    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("")
    ax[2].set_ylabel("")

    plt.tight_layout()

    # save figure
    fig.savefig(f"graphs/{dataset_name}_{metric}.png", dpi=1200)


def create_comparison_sdv_eval_dataset(frame: pd.DataFrame, dataset_name: str):
    f = frame.copy()

    # drop all where Metric != sdv_eval
    f = f[f["Test"] == "sdv_eval"]

    # drop col Dataset
    f = f.drop(columns=["Dataset"])

    # drop all where Metric != sdv_eval
    f = f[f["Test"] == "sdv_eval"]

    sns.catplot(
        data=f,
        x="Epochs",
        y="Value",
        col="Model",
        hue="K",
        palette="Set2",
        kind="bar",
    )

    # set Figure Title
    plt.suptitle(f"SDV-Eval for {dataset_name}")

    # set Y-Label f ax0, remove for ax1 and ax2
    plt.ylabel("SDV-Eval")

    plt.tight_layout()

    # save figure
    plt.savefig(f"graphs/{dataset_name}_sdv_eval.png", dpi=1200)


superstore = pd.read_csv(SUPERSTORE_TRTR_LOGS, delimiter=";")
credit_risk = pd.read_csv(CREDIT_RISK_TRTR_LOGS, delimiter=";")
mushrooms = pd.read_csv(MUSHROOMS_TRTR_LOGS, delimiter=";")

# create_super_table_for_dataset(superstore, "superstore")
# create_super_table_for_dataset(credit_risk, "credit_risk")
# create_super_table_for_dataset(mushrooms, "mushrooms")

create_comparison_accuracy_test_dataset(superstore, "superstore", "adaboost")
create_comparison_accuracy_test_dataset(credit_risk, "credit_risk", "adaboost")
create_comparison_accuracy_test_dataset(mushrooms, "mushrooms", "adaboost")

create_comparison_accuracy_test_dataset(superstore, "superstore", "svm")
create_comparison_accuracy_test_dataset(credit_risk, "credit_risk", "svm")
create_comparison_accuracy_test_dataset(mushrooms, "mushrooms", "svm")

create_comparison_accuracy_test_dataset(superstore, "superstore", "random_forest")
create_comparison_accuracy_test_dataset(credit_risk, "credit_risk", "random_forest")
create_comparison_accuracy_test_dataset(mushrooms, "mushrooms", "random_forest")

create_comparison_sdv_eval_dataset(superstore, "superstore")
create_comparison_sdv_eval_dataset(credit_risk, "credit_risk")
create_comparison_sdv_eval_dataset(mushrooms, "mushrooms")
