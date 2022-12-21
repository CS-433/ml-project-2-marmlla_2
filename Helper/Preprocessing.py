import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import tqdm


def generate_dataset(
    data,
    lookback_=10,
    norm_=False,
    trend_=False,
    span_trend_=1,
    span_back_trend_=0,
    fraction_val_=0.2,
    fraction_test_=0.2,
    verbose=1,
):
    """Generate dataset in order to train your network.
       return the train, validation, test data.

    Args:
        data: array of series of shape=(N, ), feature values where N is the number of samples.
        lookback_: scalar, the lookback window size.
        norm_: boolean, decide if apply normalisation.
        fraction_val_: scalar, percentage of validation data.
        fraction_test_: scalar, percentage of test data.

    Returns:
        train_x: numpy array of shape=(D1, lookback_, nb_feature), where D1 is the train len.
        val_x: numpy array of shape=(D2, lookback_, nb_feature), where D2 is the validation len.
        test_x: numpy array of shape=(D3, lookback_, nb_feature), where D3 is the test len.
        train_y: numpy array of shape=(D1, ), train vector label.
        val_y: numpy array of shape=(D2, ), validation vector label.
        test_y: numpy array of shape=(D3,), test vector label.
        norm: array of tuple, min max scaler and standar scaler object.
    """

    nb_test = int(fraction_test_ * len(data[0]))

    train_x_list = []
    val_x_list = []
    test_x_list = []
    train_y_list = []
    val_y_list = []
    test_y_list = []

    # save usd/chf price: (! first element of "data" list)
    price = []
    p = True

    # iterate over all features Series and extract value
    for df_ in data:

        train_x = []
        train_y = []
        test_x = []
        test_y = []

        # generate sliding window
        inputs = np.zeros((len(df_) - lookback_ - span_trend_, lookback_))
        labels = np.zeros((len(df_) - lookback_ - span_trend_, 1))
        for i in range(lookback_, len(df_) - span_trend_):

            if p:
                price.append(df_.iloc[i - 1])

            if trend_:
                # set trend label: 1 or 0 based on mean span_trend_ price
                tar = df_.iloc[i : i + span_trend_].mean()
                if tar <= df_.iloc[i - 1]:
                    labels[i - lookback_, 0] = 0
                else:
                    labels[i - lookback_, 0] = 1
            else:
                # set price label: price at t+1
                labels[i - lookback_, 0] = df_.iloc[i]

            inputs[i - lookback_] = df_.iloc[i - lookback_ : i].values

        p = False

        # split data into train and test
        train_x = inputs[:-nb_test]
        train_y = labels[:-nb_test]
        test_x = inputs[-nb_test:]
        test_y = labels[-nb_test:]

        # reshape data in the correct shape: N (nb sample), T (time), D (feature)
        train_x = train_x.reshape(-1, lookback_, 1)
        test_x = test_x.reshape(-1, lookback_, 1)

        # split data in train and validation
        train_x, val_x, train_y, val_y = train_test_split(
            train_x, train_y, test_size=fraction_val_, shuffle=False
        )

        train_x_list.append(train_x)
        val_x_list.append(val_x)
        test_x_list.append(test_x)
        train_y_list.append(train_y)
        val_y_list.append(val_y)
        test_y_list.append(test_y)

    # concatenate all features
    train_x_fin = np.concatenate(train_x_list, axis=2)
    val_x_fin = np.concatenate(val_x_list, axis=2)
    test_x_fin = np.concatenate(test_x_list, axis=2)
    train_y_fin = np.concatenate(train_y_list, axis=1)
    val_y_fin = np.concatenate(val_y_list, axis=1)
    test_y_fin = np.concatenate(test_y_list, axis=1)

    L1 = len(train_x_fin)
    L2 = len(val_x_fin) + L1
    L3 = len(test_x_fin) + L2
    price_train = price[:L1]
    price_val = price[L1:L2]
    price_test = price[L2:]

    norm = []
    if norm_:
        # calculate min max norm
        train_x_fin, train_y_fin, min_train, max_train = min_max_norm(
            train_x_fin, train_y_fin, norm_output_=not trend_
        )
        norm.append((min_train, max_train))

        val_x_fin, val_y_fin, min_val, max_val = min_max_norm(
            val_x_fin, val_y_fin, norm_output_=not trend_
        )
        norm.append((min_val, max_val))

        test_x_fin, test_y_fin, min_test, max_test = min_max_norm(
            test_x_fin, test_y_fin, norm_output_=not trend_
        )
        norm.append((min_test, max_test))

    # add trend feature to help the model
    train_x_fin, train_y_fin = add_label_data(train_x_fin, train_y_fin, span_trend_)
    val_x_fin, val_y_fin = add_label_data(val_x_fin, val_y_fin, span_trend_)
    test_x_fin, test_y_fin = add_label_data(test_x_fin, test_y_fin, span_trend_)
    # shift price of T x extract trend label [trend_1, ..., trend_D]
    price_train = price_train[train_x_fin.shape[1] + span_trend_ - 1 :]
    price_val = price_val[train_x_fin.shape[1] + span_trend_ - 1 :]
    price_test = price_test[train_x_fin.shape[1] + span_trend_ - 1 :]

    if verbose == 1:
        print(
            f"Shape: \
            \nX train     {train_x_fin.shape}, y train     {train_y_fin.shape}\
            \nX train val {val_x_fin.shape} , y train val {val_y_fin.shape} \
            \nX test      {test_x_fin.shape} , y test      {test_y_fin.shape} \
            \nPrice       {len(price)}"
        )

    return (
        train_x_fin,
        val_x_fin,
        test_x_fin,
        train_y_fin,
        val_y_fin,
        test_y_fin,
        norm,
        (price_train, price_val, price_test),
    )


def min_max_norm(x_, y_, norm_output_=True):

    seq_len = x_.shape[1]

    y_ = y_.copy()
    min = x_.min(axis=1)
    max = x_.max(axis=1)
    print(f"Number nan: {(min == max).sum()} (for var len = {len(x_)})")

    d = []

    for i in range(x_.shape[2]):

        res = (x_[:, :, i] - min[:, i].reshape(-1, 1)) / (
            max[:, i].reshape(-1, 1) - min[:, i].reshape(-1, 1)
        )
        d.append(res.reshape(-1, seq_len, 1))

        if norm_output_:
            y_[:, i] = (
                (y_[:, i].reshape(-1, 1) - min[:, i].reshape(-1, 1))
                / (max[:, i].reshape(-1, 1) - min[:, i].reshape(-1, 1))
            ).reshape(
                -1,
            )

    return np.concatenate(d, axis=2), y_, min, max


def min_max_norm_inverse(
    x_, tuple_min_max_, span_=9
):  # TODO: SPAN TRY TO REMOVE AND ADJUST GENERATE DATASET

    min = tuple_min_max_[0]
    max = tuple_min_max_[1]

    inv = x_ * (max[span_:, 0] - min[span_:, 0]) + min[span_:, 0]

    return inv


def add_label_data(df_x, df_y, span):

    X_ = np.zeros((df_x.shape[0], df_x.shape[1], df_x.shape[2] * 2))
    c = 0
    for i in range(df_x.shape[2] * 2):
        if i % 2 == 0:
            # copy the same feature price value: [p_1, ..., p_D]
            X_[:, :, i] = df_x[:, :, c]
        else:
            # add trend feature trend: [trend_1, ..., trend_D]
            for j in range(df_x.shape[1] + span - 1, len(X_)):
                X_[j, :, i] = df_y[j - df_x.shape[1] - span + 1 : j - span + 1, c]
            c += 1

    return X_[df_x.shape[1] + span - 1 :], df_y[df_x.shape[1] + span - 1 :]


def get_log_ret(df_):
    ret = df_.shift(1) / df_
    ret.replace([np.inf, np.nan, -np.inf], 0, inplace=True)
    log_ret = np.log(ret)
    log_ret.replace([np.inf, np.nan, -np.inf], 0, inplace=True)
    return log_ret.iloc[1:]


def pca_lookback(data_, lookback_, n_comp_=2):
    train_x, _, _, train_y, _, _, _ = generate_dataset(
        data_,
        lookback_=lookback_,
        norm_=False,
        fraction_val_=0.01,
        fraction_test_=0.01,
        verbose=0,
    )
    # pca
    pca = PCA(n_components=n_comp_)
    pComp = pca.fit_transform(train_x[:, :, 0])
    pr_df = pd.DataFrame(data=pComp, columns=[np.arange(n_comp_).astype(str).tolist()])

    # Regression
    X = pr_df.values
    y = train_y[:, 0].reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    score = reg.score(X, y)
    pred = reg.predict(X)
    return score, pred, X, y


def best_lookback(data, range_=range(5, 50), n_comp_=2):
    s = []
    s_top = 0.0
    pred_top = 0
    X_top, y_top = 0, 0
    i_top = 0
    for i in tqdm.tqdm(range_):
        score, pred, X, y = pca_lookback(data, lookback_=i, n_comp_=n_comp_)
        s.append(score)
        if s[-1] > s_top:
            s_top = s[-1]
            i_top = i
    return s, i_top, s_top
