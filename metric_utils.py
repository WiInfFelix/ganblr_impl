import logging

from scipy import stats as stats
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC

from data_utils import Dataset


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

    svm_synth = SVC()
    svm_synth.fit(synth_X, synth_y)
    svm_synth_acc = svm_synth.score(dataset.X_test, dataset.y_test)
    logging.info(
        f"Dataset {dataset.name} - SVM accuracy on synthetic data: {svm_synth_acc}"
    )


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
