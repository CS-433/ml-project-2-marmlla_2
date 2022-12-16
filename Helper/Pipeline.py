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
    lr_=0.0001,
    criterion_=nn.MSELoss(),
    device_="cpu",
    verbose=1,
    gru_trend=True,
):

    train_data = TensorDataset(torch.from_numpy(train_x_), torch.from_numpy(train_y_))
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=batch_size_, drop_last=False
    )

    device = device_
    print_nb = int(num_epochs_ / 5)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_)

    train_loss = []
    val_loss = []

    for epoch in tqdm.tqdm(range(1, num_epochs_ + 1)):

        avg_loss = 0.0
        model.train()
        for x, label in train_loader:

            outputs = model(x.to(device).float())
            optimizer.zero_grad()

            loss = criterion_(outputs, label.to(device).float())
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss.append(avg_loss / (len(train_loader)))

        _, _, MSE = evaluate(
            model, val_x_, val_y_, criterion_=criterion_, device=device, verbose=0
        )
        val_loss.append(MSE)

        # hearly stopping
        if len(val_loss) > 51 and np.mean(val_loss[-50:]) < np.mean(val_loss[-25:]):
            print(np.mean(val_loss[-50:]), "<", np.mean(val_loss[-25:]))
            print(
                f"Epoch: {epoch}/{num_epochs_}\nMSE = [train loss mean : {np.mean(train_loss[-print_nb:]): .08f}] , [val loss mean: {np.mean(val_loss[-print_nb:]): .08f}, MSE (last){MSE*100: .05f}%]"
            )
            return train_loss, val_loss

        if epoch % print_nb == 0 and verbose == 1:
            print(
                f"Epoch: {epoch}/{num_epochs_}\nMSE = [train loss mean : {np.mean(train_loss[-print_nb:]): .08f}] , [val loss mean: {np.mean(val_loss[-print_nb:]): .08f}, MSE (last){MSE*100: .05f}%]"
            )

    return train_loss, val_loss


def train_regularized(
    model,
    train_x_,
    train_y_,
    val_x_,
    val_y_,
    batch_size_=256,
    num_epochs_=1000,
    lr_=0.0001,
    criterion_=nn.MSELoss(reduction="sum"),
    lambda_=0.01,
    device_="cpu",
    verbose=1,
):

    train_data = TensorDataset(
        torch.from_numpy(train_x_[1:]),
        torch.from_numpy(train_y_[1:]),
        torch.from_numpy(train_y_[:-1]),
    )
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=batch_size_, drop_last=False
    )

    device = device_
    print_nb = int(num_epochs_ / 5)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_)

    train_loss = []
    val_loss = []

    for epoch in tqdm.tqdm(range(1, num_epochs_ + 1)):

        avg_loss = 0.0
        model.train()
        for x, label, label_prev in train_loader:

            outputs = model(x.to(device).float())
            optimizer.zero_grad()

            loss = criterion_(outputs, label.to(device).float()) + lambda_ * torch.sum(
                ((label - label_prev) * (label - outputs)) ** 2
            )
            avg_loss += loss.item() / len(x)
            loss.backward()
            optimizer.step()

        train_loss.append(avg_loss / (len(train_loader)))

        _, _, MSE = evaluate(
            model, val_x_, val_y_, criterion_=criterion_, device=device, verbose=0
        )
        val_loss.append(MSE / len(val_x_))

        # hearly stopping
        if len(val_loss) > 100 and np.mean(val_loss[-50:]) < np.mean(val_loss[-25:]):
            print(np.mean(val_loss[-50:]), "<", np.mean(val_loss[-25:]))
            print(
                f"Epoch: {epoch}/{num_epochs_}\nMSE = [train loss mean : {np.mean(train_loss[-print_nb:]): .08f}] , [val loss mean: {np.mean(val_loss[-print_nb:]): .08f}, MSE (last){MSE*100: .05f}%]"
            )
            return train_loss, val_loss

        if epoch % print_nb == 0 and verbose == 1:
            print(
                f"Epoch: {epoch}/{num_epochs_}\nMSE = [train loss mean : {np.mean(train_loss[-print_nb:]): .08f}] , [val loss mean: {np.mean(val_loss[-print_nb:]): .08f}, MSE (last){MSE*100: .05f}%]"
            )

    return train_loss, val_loss


def train_with_trend(
    model,
    train_x_,
    train_y_,
    train_x_trend_,
    val_x_,
    val_y_,
    val_x_trend_,
    batch_size_=256,
    num_epochs_=1000,
    lr_=0.0001,
    criterion_=nn.MSELoss(reduction="sum"),
    device_="cpu",
    verbose=1,
    gru_trend=True,
    lambda_=0.01,
):

    train_data = TensorDataset(
        torch.from_numpy(train_x_[1:]),
        torch.from_numpy(train_x_trend_[1:]),
        torch.from_numpy(train_y_[1:]),
        torch.from_numpy(train_y_[:-1]),
    )

    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=batch_size_, drop_last=False
    )

    device = device_
    print_nb = 2  # int(num_epochs_ / 5)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_)

    train_loss = []
    val_loss = []

    for epoch in tqdm.tqdm(range(1, num_epochs_ + 1)):

        avg_loss = 0.0
        model.train()
        for x, x_trend, label, label_prev in train_loader:

            outputs = model(x.to(device).float(), x_trend.to(device).float())
            optimizer.zero_grad()

            label = label.to(device).float()
            label_prev = label_prev.to(device).float()

            loss = criterion_(outputs, label.to(device).float()) + lambda_ * torch.sum(
                ((label - label_prev) * (label - outputs)) ** 2
            )
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss.append(avg_loss / (len(train_loader)))

        _, _, MSE = evaluate(
            model,
            val_x_,
            val_y_,
            criterion_=criterion_,
            device=device,
            verbose=0,
            x_trend_=val_x_trend_,
        )
        val_loss.append(MSE)

        if len(val_loss) > 20 and np.mean(val_loss[-15:]) < np.mean(val_loss[-5:]):
            print(np.mean(val_loss[-15:]), "<", np.mean(val_loss[-5:]))
            print(
                f"Epoch: {epoch}/{num_epochs_}\nMSE = [train loss mean : {np.mean(train_loss[-print_nb:]): .08f}] , [val loss mean: {np.mean(val_loss[-print_nb:]): .08f}, MSE (last){MSE*100: .05f}%]"
            )
            return train_loss, val_loss

        if epoch % print_nb == 0 and verbose == 1:
            print(
                f"Epoch: {epoch}/{num_epochs_}\nMSE = [train loss mean : {np.mean(train_loss[-print_nb:]): .08f}] , [val loss mean: {np.mean(val_loss[-print_nb:]): .08f}, MSE (last){MSE*100: .05f}%]"
            )

    return train_loss, val_loss


def evaluate(
    model, x_, y_, criterion_=nn.MSELoss(), device="cpu", verbose=0, x_trend_=[]
):

    model.eval()

    inp = torch.from_numpy(x_)
    labs = torch.from_numpy(y_)

    if len(x_trend_) > 0:
        inp2 = torch.from_numpy(x_trend_)
        out = model(inp.to(device).float(), inp2.to(device).float())
    else:
        out = model(inp.to(device).float())

    MSE = criterion_(out, labs.to(device).float()).item()

    outputs = out.cpu().detach().numpy()
    targets = labs.cpu().detach().numpy()

    if verbose == 1:
        print(f"MSE: {MSE: 0.05f}")
    return outputs, targets, MSE


def direction_accuracy(outputs, targets):
    out_dir = [
        1 if outputs[i] >= targets[i - 1] else -1 for i in range(1, len(outputs))
    ]
    tar_dir = [
        1 if targets[i] >= targets[i - 1] else -1 for i in range(1, len(targets))
    ]

    res = np.multiply(out_dir, tar_dir)
    return np.sum(res[res > 0]) / len(res)


def evauate_strategy(
    prices, pred, start=300, end=700, tax=0.9985, verbose=0, plot=False
):
    S = prices[start:end]
    pred_S = pred[start:end]

    time = np.arange(len(S))

    buy = []
    buy_t = []
    sell = []
    sell_t = []

    CHF = 5
    portfolio_list = []
    USD = 0

    trade = False

    for i in range(len(S) - 1):

        if pred_S[i+1] > S[i]:
            if trade == False:

                CHF = CHF - USD * S[i] - (1 - tax) * USD * S[i]
                USD = 0
                if verbose == 1:
                    print(
                        "gain = ", ((USD * tax / S[i]) - 1) * 100, "% [ TIME: ", i, " ]"
                    )
                    print("Portfolio value = ", CHF + USD * S[i])
                    print("-- -- -- -- -- -- -- -- -- -- -- -- -- -- ")

                USD += 1 * tax / S[i]
                CHF -= 1
                trade = True
                if verbose == 1:
                    print("LONG 1 [price = ", S[i], "] [ time: ", i, " ]")

            buy.append(S[i])
            buy_t.append(i)
            portfolio_list.append(CHF + USD * S[i])

        else:
            if trade == True:

                CHF += USD * S[i] * tax
                USD = 0
                if verbose == 1:
                    print(
                        "gain = ",
                        ((USD * S[i] * tax) - 1) * 100,
                        "%",
                        "[ TIME: ",
                        i,
                        " ]",
                    )
                    print("Portfolio value = ", CHF + USD * S[i])
                    print("-- -- -- -- -- -- -- -- -- -- -- -- -- -- ")

                CHF += 1 * tax
                USD += 1 / S[i]

                trade = False
                if verbose == 1:
                    print("SHORT 1 [price = ", S[i], "] [ time: ", i, " ]")

            sell.append(S[i])
            sell_t.append(i)

            portfolio_list.append(CHF - USD * S[i])

    portfolio_list.append(portfolio_list[-1])
    if plot:

        plt.figure(figsize=(25, 15))

        plt.subplot(2, 1, 1)
        plt.plot(
            buy_t,
            buy,
            marker="X",
            markersize=10,
            linestyle="None",
            color="green",
            label="buy",
        )
        plt.plot(
            sell_t,
            sell,
            marker="X",
            markersize=10,
            linestyle="None",
            color="red",
            label="sell",
        )

        plt.plot(time, S, ".-", label="open price")
        plt.plot(time, pred_S, ".-", label="open price predicted")
        plt.grid(True)
        plt.legend(fontsize=20)

        plt.subplot(2, 1, 2)
        plt.plot(time, portfolio_list, "o-", label="portfolio")
        plt.grid(True)
        plt.legend(fontsize=20)

        plt.xlabel("Time", fontsize=20)
        plt.savefig("result.svg")
        plt.show()

    p = np.array(portfolio_list)
    print(
        f"gain = {(portfolio_list[-1] / portfolio_list[0] - 1) * 100:0.02f}%",
    )

    average_ret = np.mean((p[1:] / p[:-1] - 1)) * 100
    return average_ret


def evaluate_trend(model, x_, y_, device="cpu", verbose=0):

    model.eval()

    inp = torch.from_numpy(x_)
    targets = torch.from_numpy(y_)

    outputs = model(inp.float().to(device))

    # _, outputs = torch.max(outputs, 1)
    outputs = nn.Sigmoid()(outputs).cpu().detach().numpy().reshape(-1).round()
    # _, targets = torch.max(targets, 1)

    targets = (
        targets.cpu()
        .detach()
        .numpy()
        .reshape(
            -1,
        )
    )

    ACC = np.mean((outputs == targets))

    if verbose == 1:
        print(f"ACC: {ACC: 0.05f}")
        print(
            f"nb 0 = {len(outputs[outputs == 0.])}; nb 1 = {len(outputs[outputs == 1.])};"
        )
    return outputs, targets, ACC


def evauate_strategy_trend(
    prices, pred, start=300, end=700, tax=0.9985, verbose=0, plot=False
):
    S = prices[start:end]
    pred_S = pred[start:end]

    time = np.arange(len(S))

    buy = []
    buy_t = []
    sell = []
    sell_t = []

    CHF = 1
    portfolio_list = []
    USD = 0
    tax = 0.9985
    trade = False

    for i in range(len(S) - 1):

        if pred_S[i] == 1:
            if trade == False:

                CHF = CHF - USD * S[i] - (1 - tax) * USD * S[i]
                USD = 0
                if verbose == 1:
                    print(
                        "gain = ", ((USD * tax / S[i]) - 1) * 100, "% [ TIME: ", i, " ]"
                    )
                    print("Portfolio value = ", CHF + USD * S[i])
                    print("-- -- -- -- -- -- -- -- -- -- -- -- -- -- ")

                USD += 1 * tax / S[i]
                CHF -= 1
                trade = True
                if verbose == 1:
                    print("LONG 1 [price = ", S[i], "] [ time: ", i, " ]")

            buy.append(S[i])
            buy_t.append(i)
            portfolio_list.append(CHF + USD * S[i])

        else:
            if trade == True:

                CHF += USD * S[i] * tax
                USD = 0
                if verbose == 1:
                    print(
                        "gain = ",
                        ((USD * S[i] * tax) - 1) * 100,
                        "%",
                        "[ TIME: ",
                        i,
                        " ]",
                    )
                    print("Portfolio value = ", CHF + USD * S[i])
                    print("-- -- -- -- -- -- -- -- -- -- -- -- -- -- ")

                CHF += 1 * tax
                USD += 1 / S[i]

                trade = False
                if verbose == 1:
                    print("SHORT 1 [price = ", S[i], "] [ time: ", i, " ]")

            sell.append(S[i])
            sell_t.append(i)

            portfolio_list.append(CHF - USD * S[i])

    portfolio_list.append(portfolio_list[-1])
    if plot:

        plt.figure(figsize=(25, 15))

        plt.subplot(2, 1, 1)
        plt.plot(
            buy_t,
            buy,
            marker="X",
            markersize=10,
            linestyle="None",
            color="green",
            label="buy",
        )
        plt.plot(
            sell_t,
            sell,
            marker="X",
            markersize=10,
            linestyle="None",
            color="red",
            label="sell",
        )

        plt.plot(time, S, ".-", label="open price")
        plt.grid(True)
        plt.legend(fontsize=20)

        plt.subplot(2, 1, 2)
        plt.plot(time, portfolio_list, "o-", label="portfolio")
        plt.grid(True)
        plt.legend(fontsize=20)

        plt.xlabel("Time", fontsize=20)
        plt.show()

    p = np.array(portfolio_list)
    average_ret = np.mean((p[1:] / p[:-1] - 1)) * 100
    return average_ret


def train_aux(
    model,
    train_x_,
    train_y_,
    val_x_,
    val_y_,
    batch_size_=256,
    num_epochs_=1000,
    lr_=0.0001,
    device_="cpu",
):

    train_data = TensorDataset(torch.from_numpy(train_x_), torch.from_numpy(train_y_))
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=batch_size_, drop_last=True
    )

    device = device_
    print_nb = int(num_epochs_ / 5)

    criterion = [
        nn.MSELoss(),
        nn.MSELoss(),
        nn.MSELoss(),
        nn.MSELoss(),
        nn.MSELoss(),
        nn.MSELoss(),
        nn.MSELoss(),
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_)

    train_loss = []
    val_loss = []

    aux_loss = [[], [], [], [], [], []]
    aux_loss_val = [[], [], [], [], [], []]

    for epoch in tqdm.tqdm(range(1, num_epochs_ + 1)):

        avg_loss = 0.0
        avg_loss_aux = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        model.train()
        for x, label in train_loader:
            outputs = model(x.to(device).float())

            optimizer.zero_grad()
            for i in range(len(aux_loss) - 1):
                loss = criterion[i](
                    outputs[i], label.to(device).float()[:, i].unsqueeze(1)
                )
                avg_loss_aux[i] += loss.item()
                loss.backward(retain_graph=True)

            loss = criterion[-2](
                outputs[-2], label.to(device).float()[:, 0].unsqueeze(1)
            )
            avg_loss_aux[-1] += loss.item()
            loss.backward(retain_graph=True)

            loss = criterion[-1](
                outputs[-1], label.to(device).float()[:, 0].unsqueeze(1)
            )
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss.append(avg_loss / (len(train_loader)))
        for i in range(len(aux_loss)):
            aux_loss[i].append(avg_loss_aux[i] / len(train_loader))

        model.eval()
        inp = torch.from_numpy(np.array(val_x_))
        labs = torch.from_numpy(np.array(val_y_[:, 0]))
        out = model(inp.to(device).float())
        outputs = out[-1].cpu().detach().numpy().reshape(-1)
        targets = labs.numpy().reshape(-1)
        MSE = np.mean((outputs - targets) ** 2)
        val_loss.append(MSE)

        for i in range(len(aux_loss) - 1):
            labs = torch.from_numpy(np.array(val_y_[:, i]))
            outputs = out[i].cpu().detach().numpy().reshape(-1)
            targets = labs.numpy().reshape(-1)
            aux_loss_val[i].append(np.mean((outputs - targets) ** 2))

        labs = torch.from_numpy(np.array(val_y_[:, 0]))
        outputs = out[-2].cpu().detach().numpy().reshape(-1)
        targets = labs.numpy().reshape(-1)
        aux_loss_val[-1].append(np.mean((outputs - targets) ** 2))

        if epoch % print_nb == 0:
            print(
                f"Epoch: {epoch}/{num_epochs_}\nMSE = [train loss mean : {np.mean(train_loss[-print_nb:]): .08f}] , [val loss mean: {np.mean(val_loss[-print_nb:]): .08f}, MSE (last){MSE*100: .05f}%]"
            )
            print(
                f"Aux loss train: [close: {np.mean(aux_loss[0][-print_nb:]): 0.05f} ], [SMI: {np.mean(aux_loss[1][-print_nb:]): 0.05f} ], [SP500: {np.mean(aux_loss[2][-print_nb:]): 0.05f} ], [bondCH: {np.mean(aux_loss[3][-print_nb:]): 0.05f} ] [bondUS: {np.mean(aux_loss[4][-print_nb:]): 0.05f} ] [Gru base: {np.mean(aux_loss[5][-print_nb:]): 0.05f}]"
            )
            print(
                f"Aux loss val:   [close: {np.mean(aux_loss_val[0][-print_nb:]): 0.05f} ], [SMI: {np.mean(aux_loss_val[1][-print_nb:]): 0.05f} ], [SP500: {np.mean(aux_loss_val[2][-print_nb:]): 0.05f} ], [bondCH: {np.mean(aux_loss_val[3][-print_nb:]): 0.05f} ] [bondUS: {np.mean(aux_loss_val[4][-print_nb:]): 0.05f} ] [Gru base: {np.mean(aux_loss_val[5][-print_nb:]): 0.05f}]"
            )

    return train_loss, val_loss, aux_loss, aux_loss_val


def smooth_loss(val, chunksize=100):
    mean_list = []
    for i in range(chunksize, len(val), chunksize):
        mean_list.append(np.mean(val[i - chunksize : i]))
    return mean_list


def evaluate_aux_trend(model_trend, model, x_, y_, device="cpu"):
    model.eval()
    model_trend.eval()

    outputs = []
    targets = []

    inp = torch.from_numpy(np.array(x_))
    labs = torch.from_numpy(np.array(y_[:, 0]))

    out1 = model_trend(inp.to(device).float())
    out = model(inp.to(device).float(), nn.Sigmoid()(out1))
    outputs = out[-1].cpu().detach().numpy().reshape(-1)
    targets = labs.numpy().reshape(-1)

    MSE = np.mean((outputs - targets) ** 2)
    print("MSE: {}%".format(MSE * 100))

    return outputs, targets, MSE


def evaluate_aux(model, x_, y_, device="cpu"):
    model.eval()

    outputs = []
    targets = []

    inp = torch.from_numpy(np.array(x_))
    labs = torch.from_numpy(np.array(y_[:, 0]))

    out = model(inp.to(device).float())
    outputs = out[-1].cpu().detach().numpy().reshape(-1)
    targets = labs.numpy().reshape(-1)

    MSE = np.mean((outputs - targets) ** 2)
    print("MSE: {}%".format(MSE * 100))

    return outputs, targets, MSE


def regression_result(targets, outputs):
    Y = targets
    X = outputs
    X = sm.add_constant(X, has_constant="add")

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
