from ganblr.models import GANBLR
import pandas as pd
import os
import numpy as np
import scipy.stats as stats
import datetime
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

import logging

# build folder for logs
if not os.path.exists("logs"):
    os.mkdir("logs")

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

ordinal_encoder = OrdinalEncoder()
label_encoder = LabelEncoder()

# Set up logging to file with timestamp name
logging.basicConfig(filename=f"logs/analysis_{timestamp}.log", level=logging.INFO, format='%(asctime)s %(message)s')



def read_in_test_data():
    superstore = pd.read_csv("./SampleSuperstore.csv")

    print(superstore.columns)

    return superstore

def encode_columns(dframe: pd.DataFrame):
    # drop column city and postal code
    dframe = dframe.drop(['City', 'Postal Code', "State"], axis=1)

    # encode last column with label encoder
    dframe['Profit'] = label_encoder.fit_transform(dframe['Profit'])

    # remove upper and lower outliers
    for col in dframe.columns:
        if dframe[col].dtype == 'float64':
            q1 = dframe[col].quantile(0.25)
            q3 = dframe[col].quantile(0.75)
            iqr = q3 - q1
            dframe = dframe[(dframe[col] > (q1 - 1.5 * iqr)) & (dframe[col] < (q3 + 1.5 * iqr))]


    numeric_columns = dframe.select_dtypes(include=['int64', 'float64']).columns

    # bin the numeric columns
    for col in numeric_columns:
        dframe[col] = pd.qcut(dframe[col], q=10, duplicates='drop')

    # encode the non-numeric columns
    for col in dframe.columns:
        if dframe[col].dtype == 'object':
            dframe[col] = ordinal_encoder.fit_transform(dframe[[col]])

    # log all columns with their number of unique values
    for col in dframe.columns:
        logging.info(f"{col}: {dframe[col].nunique()}")


def bin_freedman(data: pd.Series):
    """Bins a series of data using the Freedman-Diaconis rule"""
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    h = 2 * iqr * (len(data)**(-1/3))
    bins = int((data.max() - data.min()) / h)
    return bins

def get_cramers_v(traindata: pd.DataFrame, synth_data: pd.DataFrame):
    # get the number of rows and columns
    rows = traindata.shape[0]
    cols = traindata.shape[1]

    # get the chi-squared statistic
    chi2 = stats.chi2_contingency(np.array([traindata, synth_data]), correction=False)[0]


    # calculate cramers v
    cramers_v = np.sqrt(chi2 / (rows * (min(cols - 1, rows - 1))))

    return cramers_v


def run_metrics(traindata: pd.DataFrame, synth_data: pd.DataFrame):
    # get wasserstein distance
    wasserstein = stats.wasserstein_distance(traindata, synth_data)

    # get kolmogorov-smirnov statistic
    ks_statistic = stats.ks_2samp(traindata, synth_data)

    # get chi-squared statistic
    chi2_statistic = stats.chisquare(traindata, synth_data)

    # cramers v lambda
    cramers = get_cramers_v(traindata, synth_data)

    # log the metrics
    logging.info(f"Wasserstein distance: {wasserstein}")
    logging.info(f"KS statistic: {ks_statistic}")
    logging.info(f"Chi-squared statistic: {chi2_statistic}")


    logging.info("Starting TRTR/TSTR")
    
def main():
    logging.info('Starting analysis')

    data = read_in_test_data()
    # data = get_demo_data("adult")
    logging.info('Read in data')

    encode_columns(data)
    logging.info('Encoded columns')

    # Split data into train and test
    train = data.sample(frac=0.8, random_state=200)
    test = data.drop(train.index)
    logging.info('Split data into train and test')

    # Get the train data
    x_train = train.drop('Profit', axis=1).to_numpy()
    y_train = train['Profit'].to_numpy()
    logging.info('Got train data')

    # Get the test data
    x_test = test.drop('Profit', axis=1).to_numpy()
    y_test = test['Profit'].to_numpy()
    logging.info('Got test data')

    # Create the model
    ganblr = GANBLR()
    logging.info('Starting training')
    # start timer
    start = datetime.datetime.now()
    ganblr.fit(x_train[:1000], y_train[:1000], k=1)
    logging.info('Finished training, took %s', datetime.datetime.now() - start)

    # concatenate x_train and y_train
    traindata = np.concatenate((x_train, y_train.reshape(-1, 1)), axis=1)

    # Get the predictions
    synth_data = ganblr.sample(1000)

    # Get the metrics
    run_metrics(traindata, synth_data)


if __name__ == "__main__":
    main()


# loop 10 times
