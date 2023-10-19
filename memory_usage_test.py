import pandas as pd
from timeit import default_timer as timer
import random
from ganblr.kdb import build_graph
import psutil
import numpy as np

def create_mock_dataset(r, d):
    """Creates a mock dataset with r cols and d as highest dimension possible"""
    df = pd.DataFrame()

    for i in range(r + 1):
        # append 10000 rows to col with random values between 0 and d
        col = np.random.randint(1, d, size=10000)
        df[i] = col

    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    return X.to_numpy(), y.to_numpy()


def call_with_profiling(r, d, k):
    X, y = create_mock_dataset(r, d)
    start = timer()
    ram_before = psutil.Process().memory_info().rss
    build_graph(X, y, k=k)
    ram_after = psutil.Process().memory_info().rss

    return timer() - start, ram_after - ram_before


R = [5, 10, 20, 50, 100, 200, 500, 1000]
D = [5, 10, 20, 50, 100, 200, 500, 1000]
K = [0, 1, 2, 3, 4, 5]

for r in R:
    for d in D:
        for k in K:
            time, ram = call_with_profiling(r, d, k)
            print("r:", r, "d:", d, "k:", k, "time:", time, "ram:", ram)
            with open("memory_usage.csv", "a") as f:
                f.write(f"{r};{d};{k};{time};{ram}\n")
