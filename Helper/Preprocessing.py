import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


def generate_dataset(
    data, lookback_=10, norm_=False, fraction_val_=0.2, fraction_test_=0.2
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

    norm = []
    for df_ in data:
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        # generate sliding window
        inputs = np.zeros((len(df_) - lookback_, lookback_))
        labels = np.zeros((len(df_) - lookback_, 1))

        for i in range(lookback_, len(df_) - 2):
            inputs[i - lookback_] = df_.iloc[i - lookback_ : i].values
            labels[i - lookback_, 0] = df_.iloc[i]
            # labels[i - lookback_, 1] = 0 if df_.iloc[i + 2] <= df_.iloc[i - 1] else 1

        # Split data into train and test
        train_x = inputs[:-nb_test]
        train_y = labels[:-nb_test]
        test_x = inputs[-nb_test:]
        test_y = labels[-nb_test:]

        mm = MinMaxScaler()
        ss = StandardScaler()
        if norm_:
            train_x = mm.fit_transform(train_x)
            train_y = ss.fit_transform(train_y)
            test_x = mm.transform(test_x)
            test_y = ss.transform(test_y)
            norm.append((mm, ss))

        # reshape data in the correct shape: N (nb sample), T (time), D (feature)
        train_x = train_x.reshape(-1, lookback_, 1)
        test_x = test_x.reshape(-1, lookback_, 1)

        train_x, val_x, train_y, val_y = train_test_split(
            train_x, train_y, test_size=fraction_val_, shuffle=False
        )  # random_state=0

        train_x_list.append(train_x)
        val_x_list.append(val_x)
        test_x_list.append(test_x)
        train_y_list.append(train_y)
        val_y_list.append(val_y)
        test_y_list.append(test_y)

    train_x_fin = np.concatenate(train_x_list, axis=2)
    val_x_fin = np.concatenate(val_x_list, axis=2)
    test_x_fin = np.concatenate(test_x_list, axis=2)
    train_y_fin = np.concatenate(train_y_list, axis=1)
    val_y_fin = np.concatenate(val_y_list, axis=1)
    test_y_fin = np.concatenate(test_y_list, axis=1)

    print(
        f"Shape: \
        \nX train     {train_x_fin.shape}, y train     {train_y_fin.shape}\
        \nX train val {val_x_fin.shape} , y train val {val_y_fin.shape} \
        \nX test      {test_x_fin.shape} , y test      {test_y_fin.shape}"
    )

    return train_x_fin, val_x_fin, test_x_fin, train_y_fin, val_y_fin, test_y_fin, norm
