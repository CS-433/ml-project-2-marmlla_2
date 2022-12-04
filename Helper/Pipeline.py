import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

import statsmodels.api as sm


def train(
    model,
    train_x_,
    train_y_,
    val_x_,
    val_y_,
    batch_size_=256,
    num_epochs_=1000,
    lr=0.0001,
    device_="cpu",
):

    train_data = TensorDataset(torch.from_numpy(train_x_), torch.from_numpy(train_y_))
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=batch_size_, drop_last=True
    )

    device = device_
    print_nb = int(num_epochs_ / 5)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_loss = []
    val_loss = []

    for epoch in tqdm.tqdm(range(1, num_epochs_ + 1)):

        avg_loss = 0.0
        model.train()
        for x, label in train_loader:
            outputs = model(x.to(device).float())
            optimizer.zero_grad()
            loss = criterion(outputs, label.to(device).float()[:, 0].unsqueeze(1))
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss.append(avg_loss / (len(train_loader)))

        model.eval()
        inp = torch.from_numpy(np.array(val_x_))
        labs = torch.from_numpy(np.array(val_y_[:, 0]))
        out = model(inp.to(device).float())
        outputs = out.cpu().detach().numpy().reshape(-1)
        targets = labs.numpy().reshape(-1)
        MSE = np.mean((outputs - targets) ** 2)
        val_loss.append(MSE)

        if epoch % print_nb == 0:
            print(
                f"Epoch: {epoch}/{num_epochs_}\nMSE = [train loss mean : {np.mean(train_loss[-print_nb:]): .08f}] , [val loss mean: {np.mean(val_loss[-print_nb:]): .08f}, MSE (last){MSE*100: .05f}%]"
            )


        
    return train_loss, val_loss


def smooth_loss(val, chunksize=100):
    mean_list = []
    for i in range(chunksize, len(val), chunksize):
        mean_list.append(np.mean(val[i - chunksize : i]))
    return mean_list


def evaluate(model, x_, y_, device="cpu"):
    model.eval()

    outputs = []
    targets = []

    inp = torch.from_numpy(np.array(x_))
    labs = torch.from_numpy(np.array(y_[:, 0]))

    out = model(inp.to(device).float())
    outputs = out.cpu().detach().numpy().reshape(-1)
    targets = labs.numpy().reshape(-1)

    MSE = np.mean((outputs - targets) ** 2)
    print("MSE: {}%".format(MSE * 100))

    return outputs, targets, MSE


def regression_result(targets, outputs):
    Y = targets
    X = outputs
    X = sm.add_constant(X, has_constant='add')
    
    model = sm.OLS(Y, X)
    res = model.fit()
    print(res.summary())

    plt.plot(outputs, targets, ".")
    plt.plot(X[:, 1], res.predict(X))
    plt.xlabel("outputs")
    plt.ylabel("targets")
    plt.grid()
    plt.show()


def direction_score(outputs, targets):
    outp = outputs.copy()
    outp[outp > 0] = 1
    outp[outp <= 0] = 0

    targ = targets.copy()
    targ[targ > 0] = 1
    targ[targ <= 0] = 0

    res = (targ == outp).sum() / len(targ)
    print(f"accuracy {accuracy_score(outp, targ)*100 : 0.2f} %")
    sns.heatmap(data=confusion_matrix(outp, targ), annot=True)
    plt.plot()
