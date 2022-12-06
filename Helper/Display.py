import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")


def plot_data(df_, fraction_test_=0.1):

    nb_test = int(fraction_test_ * len(df_))

    plt.figure(figsize=(15, 9))

    plt.subplot(3, 1, 1)
    plt.title("Exchange rate")
    plt.plot(
        df_.Date.values[:-nb_test], df_.Close.values[:-nb_test], label="train data"
    )
    plt.plot(
        df_.Date.values[-nb_test:],
        df_.Close.values[-nb_test:],
        color="r",
        label="test data",
    )
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("USD/CHF", fontsize=10)
    plt.grid()
    plt.legend(fontsize=10)

    plt.subplot(3, 1, 2)
    plt.title("Inflation")
    plt.plot(
        df_.Date.values[:-nb_test],
        df_.US_infl.values[:-nb_test],
        label="train data (US)",
    )
    plt.plot(
        df_.Date.values[-nb_test:],
        df_.US_infl.values[-nb_test:],
        color="r",
        label="test data (US)",
    )
    plt.plot(
        df_.Date.values[:-nb_test],
        df_.CH_infl.values[:-nb_test],
        label="train data (CH)",
    )
    plt.plot(
        df_.Date.values[-nb_test:],
        df_.CH_infl.values[-nb_test:],
        color="g",
        label="test data (CH)",
    )
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("%", fontsize=10)
    plt.grid()
    plt.legend(fontsize=10)

    plt.subplot(3, 1, 3)
    plt.title("Interest rate")
    plt.plot(
        df_.Date.values[:-nb_test],
        df_.US_IR.values[:-nb_test] - df_.SARON_close.values[:-nb_test],
        label="train data (US)",
    )
    plt.plot(
        df_.Date.values[-nb_test:],
        df_.US_IR.values[-nb_test:] - df_.SARON_close.values[-nb_test:],
        color="r",
        label="test data (US)",
    )
    """
    plt.plot(
        df_.Date.values[:-nb_test],
        df_.SARON_close.values[:-nb_test],
        label="train data (CH)",
    )
    plt.plot(
        df_.Date.values[-nb_test:],
        df_.SARON_close.values[-nb_test:],
        color="g",
        label="test data (CH)",
    )
    """
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("%", fontsize=10)
    plt.grid()
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_log_ret(df_):
    data_plt = [df_.Close, df_.US_IR, df_.SARON_close]
    name = ["Close", "US_IR", "SARON_close"]
    fig = plt.figure(figsize=(12, 9))
    c = 0
    for d, n in zip(data_plt, name):

        plt.subplot(3, 3, 1 + c)
        plt.hist(d, bins=150)
        plt.title("Price " + n)

        plt.subplot(3, 3, 2 + c)
        ret = d.shift(1) / d
        ret.replace([np.inf, -np.inf], 0, inplace=True)
        plt.hist(ret.iloc[1:] - 1, bins=150)
        plt.title("Return " + n)

        plt.subplot(3, 3, 3 + c)
        log_ret = np.log(ret)
        log_ret.replace([np.inf, -np.inf], 0, inplace=True)
        plt.hist(log_ret, bins=150)
        plt.title("Log Return " + n)

        c += 3
    plt.tight_layout()
    plt.show()


def plot_result_ret(outputs, targets, nb=30):
    plt.figure(figsize=(14, 10))

    # plt.subplot(2,1,1)
    plt.bar(np.arange(nb), outputs[-nb:], color="g", label="Predicted ")
    # plt.plot(np.cumsum(outputs[-nb:]), "-o", color="g", label="Predicted reg")

    # plt.subplot(2,1,2)
    plt.bar(np.arange(nb), targets[-nb:], width=0.2, color="k", label="Actual")
    # plt.plot(np.cumsum(targets[-nb:]),"-o", color="k", label="Actual")
    plt.ylabel("USD/CHF")
    plt.grid()
    plt.legend()
    plt.plot()


def plot_result_price(outputs, targets, range_=range(0, 30)):

    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(outputs[range_], "-o", color="g", label="Predicted ")
    plt.grid()
    plt.ylabel("USD/CHF")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(targets[range_], "-o", color="k", label="Actual")
    plt.grid()
    plt.ylabel("USD/CHF")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(targets[range_], "-o", color="k", label="Actual")
    plt.plot(outputs[range_], "-o", color="g", label="Predicted ")

    plt.ylabel("USD/CHF")
    plt.grid()
    plt.legend()
    plt.plot()
