import logging

from scipy import stats as stats
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC

from data_utils import Dataset
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

def get_trtr_metrics(
    X_train,
    X_test,
    y_train,
    y_test,
    synth_data,
    dataset_name,
    model,
    overall_logfile,
    epochs,
    k,
):
    print(f"Getting metrics for dataset {dataset_name}")



    synth_X = synth_data.iloc[:, :-1]

    # select y and convert to int
    synth_y = synth_data.iloc[:, -1].astype(int)

    with open (overall_logfile, "a") as f:


        # train adaboost, random forest and svm classifier on real data
        adaboost = AdaBoostClassifier()
        adaboost.fit(X_train, y_train)
        adaboost_acc = adaboost.score(X_test, y_test)

        f.write(
            f"trtr;{model};{epochs};{k};{dataset_name};adaboost;accuracy;{adaboost_acc}\n"
        )
        

        random_forest = RandomForestClassifier()
        random_forest.fit(X_train, y_train)
        random_forest_acc = random_forest.score(X_test, y_test)
        f.write(
            f"trtr;{model};{epochs};{k};{dataset_name};random_forest;accuracy;{random_forest_acc}\n"
        )

        svm = SVC()
        svm.fit(X_train, y_train)
        svm_acc = svm.score(X_test, y_test)
        f.write(f"trtr;{model};{epochs};{k};{dataset_name};svm;accuracy;{svm_acc}\n")

        # train adaboost, random forest and svm classifier on synthetic data
        adaboost_synth = AdaBoostClassifier()
        adaboost_synth.fit(synth_X, synth_y)
        adaboost_synth_acc = adaboost_synth.score(X_test, y_test)
        f.write(
            f"trtr;{model};{epochs};{k};{dataset_name};adaboost;accuracy;{adaboost_synth_acc}\n"
        )

        random_forest_synth = RandomForestClassifier()
        random_forest_synth.fit(synth_X, synth_y)
        random_forest_synth_acc = random_forest_synth.score(X_test, y_test)
        f.write(
            f"trtr;{model};{epochs};{k};{dataset_name};random_forest;accuracy;{random_forest_synth_acc}\n"
        )

        svm_synth = SVC()
        svm_synth.fit(synth_X, synth_y)
        svm_synth_acc = svm_synth.score(X_test, y_test)
        f.write(f"trtr;{model};{epochs};{k};{dataset_name};svm;accuracy;{svm_synth_acc}\n")


def get_distance_metrics(real_data, synth_data):
    """Gets wasserstein, chi squared and kolmogorov smirnov distance metrics for real and synthetic data"""

    logging.info(f"Getting distance metrics for dataset {name}")

    # get wasserstein distance from dataframes
    wasserstein = stats.wasserstein_distance(real_data, synth_data)
    
    # get chi squared distance from dataframes
    chi_squared = stats.chisquare(real_data, synth_data)
    
    # get kolmogorov smirnov distance from dataframes
    kolmogorov_smirnov = stats.ks_2samp(real_data, synth_data)

    # log all metrics
    logging.info(f"Dataset {name} - Wasserstein distance: {wasserstein}")
    logging.info(f"Dataset {name} - Chi squared distance: {chi_squared}")
    logging.info(f"Dataset {name} - Kolmogorov smirnov distance: {kolmogorov_smirnov}")


def get_cramers_v(real_data, synth_data):
    """Gets cramers v metric for real and synthetic data"""

    logging.info(f"Getting Cramers V metric for dataset {name}")

    # get real data
    real_data = data.iloc[:, :-1]

    # get synthetic data
    synth_data = synth_data.iloc[:, :-1]

    # get cramers v
    cramers_v = stats.chi2_contingency(real_data, synth_data)

    # log all metrics
    logging.info(f"Dataset {name} - Cramers V: {cramers_v}")


def get_sdv_metrics(real_data, synth_data, epochs, k, dataset_name, model, overall_logfile, timestamp,i):
    """Calculates the evaluation metrics using the SDV library"""

    # cast real data and synth data to categorical
    real_data = real_data.astype("category")
    synth_data = synth_data.astype("category")


    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)

    with open (overall_logfile, "a") as f:
    
    #metadata = metadata.to_dict()
    
        quality_report = evaluate_quality(synth_data, real_data, metadata)
        
        f.write(
            f"sdv_eval;{model};{epochs};{k};{dataset_name};sdv_eval;quality_score;{quality_report.get_score()}\n"
        )

        
        report_columns_shapes = quality_report.get_details("Column Shapes")
        report_column_trends = quality_report.get_details("Column Pair Trends")
        
        report_columns_shapes.to_csv(f"./sdv_frames/{timestamp}_{model}_report_columns_shapes_{dataset_name}_{epochs}_{k}_{i}.csv")
        report_column_trends.to_csv(f"./sdv_frames/{timestamp}_{model}_report_column_trends_{dataset_name}_{epochs}_{k}_{i}.csv")
        
        
        #quality_report.save(f"quality_report_{dataset_name}_{epochs}_{k}.json")