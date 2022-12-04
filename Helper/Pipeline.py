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
    device_="cpu",
):

    train_data = TensorDataset(torch.from_numpy(train_x_), torch.from_numpy(train_y_))
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=batch_size_, drop_last=True
    )

    device = device_
    print_nb = int(num_epochs_ / 5)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_)

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

    criterion = [nn.MSELoss(), nn.MSELoss(),nn.MSELoss(),nn.MSELoss(),nn.MSELoss(),nn.MSELoss(), nn.MSELoss()]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_)

    train_loss = []
    val_loss = []
    
    aux_loss= [[], [], [], [], [], []]
    aux_loss_val= [[], [], [], [], [], []]

    for epoch in tqdm.tqdm(range(1, num_epochs_ + 1)):

        avg_loss = 0.0
        avg_loss_aux = [0.,0.,0.,0.,0.,0.]
        
        model.train()
        for x, label in train_loader:
            outputs = model(x.to(device).float())
            
            optimizer.zero_grad()
            for i in range(len(aux_loss)-1):
                loss = criterion[i](outputs[i], label.to(device).float()[:, i].unsqueeze(1))
                avg_loss_aux[i] += loss.item()
                loss.backward(retain_graph = True)

            loss = criterion[-2](outputs[-2], label.to(device).float()[:, 0].unsqueeze(1))
            avg_loss_aux[-1] += loss.item()
            loss.backward(retain_graph = True)
            
            loss = criterion[-1](outputs[-1], label.to(device).float()[:, 0].unsqueeze(1))
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        train_loss.append(avg_loss / (len(train_loader)))
        for i in range(len(aux_loss)):
            aux_loss[i].append(avg_loss_aux[i]/len(train_loader))

        model.eval()
        inp = torch.from_numpy(np.array(val_x_))
        labs = torch.from_numpy(np.array(val_y_[:, 0]))
        out = model(inp.to(device).float())
        outputs = out[-1].cpu().detach().numpy().reshape(-1)
        targets = labs.numpy().reshape(-1)
        MSE = np.mean((outputs - targets) ** 2)
        val_loss.append(MSE)
        
        for i in range(len(aux_loss)-1):
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
            print(f"Aux loss train: [close: {np.mean(aux_loss[0][-print_nb:]): 0.05f} ], [SMI: {np.mean(aux_loss[1][-print_nb:]): 0.05f} ], [SP500: {np.mean(aux_loss[2][-print_nb:]): 0.05f} ], [bondCH: {np.mean(aux_loss[3][-print_nb:]): 0.05f} ] [bondUS: {np.mean(aux_loss[4][-print_nb:]): 0.05f} ] [Gru base: {np.mean(aux_loss[5][-print_nb:]): 0.05f}]")
            print(f"Aux loss val:   [close: {np.mean(aux_loss_val[0][-print_nb:]): 0.05f} ], [SMI: {np.mean(aux_loss_val[1][-print_nb:]): 0.05f} ], [SP500: {np.mean(aux_loss_val[2][-print_nb:]): 0.05f} ], [bondCH: {np.mean(aux_loss_val[3][-print_nb:]): 0.05f} ] [bondUS: {np.mean(aux_loss_val[4][-print_nb:]): 0.05f} ] [Gru base: {np.mean(aux_loss_val[5][-print_nb:]): 0.05f}]")


        
    return train_loss, val_loss, aux_loss, aux_loss_val
    
    
def train_trend(
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
    buff = train_y_[:, 0]

    positive_weight = torch.tensor(len(buff[buff == 0.])/len(buff[buff == 1.])).float()#.cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_)

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
        outputs = nn.Sigmoid()(out).cpu().detach().numpy().reshape(-1).round()
        targets = labs.numpy().reshape(-1)
        ACC = np.mean((outputs == targets))
        val_loss.append(ACC)

        if epoch % print_nb == 0:
            print(
                f"Epoch: {epoch}/{num_epochs_}\nMSE = [train loss mean : {np.mean(train_loss[-print_nb:]): .08f}] , [val loss mean: {np.mean(val_loss[-print_nb:]): .08f}, MSE (last){ACC*100: .05f}%]"
            )
            print(f'nb 0: {len(outputs[outputs == 0])}, nb 1: {len(outputs[outputs == 1])}')


        
    return train_loss, val_loss
    
    
def train_aux_trend(
    model_trend,
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

    criterion = [nn.MSELoss(), nn.MSELoss(),nn.MSELoss(),nn.MSELoss(),nn.MSELoss(),nn.MSELoss(), nn.MSELoss()]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_)

    train_loss = []
    val_loss = []
    
    aux_loss= [[], [], [], [], [], []]
    aux_loss_val= [[], [], [], [], [], []]
    
    model_trend.eval()
    
    for epoch in tqdm.tqdm(range(1, num_epochs_ + 1)):

        avg_loss = 0.0
        avg_loss_aux = [0.,0.,0.,0.,0.,0.]
        
        model.train()
        for x, label in train_loader:
            out1 = model_trend(x.to(device).float())

            outputs = model(x.to(device).float(),  nn.Sigmoid()(out1))
            
            optimizer.zero_grad()
            for i in range(len(aux_loss)-1):
                loss = criterion[i](outputs[i], label.to(device).float()[:, i].unsqueeze(1))
                avg_loss_aux[i] += loss.item()
                loss.backward(retain_graph = True)

            loss = criterion[-2](outputs[-2], label.to(device).float()[:, 0].unsqueeze(1))
            avg_loss_aux[-1] += loss.item()
            loss.backward(retain_graph = True)
            
            loss = criterion[-1](outputs[-1], label.to(device).float()[:, 0].unsqueeze(1))
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        train_loss.append(avg_loss / (len(train_loader)))
        for i in range(len(aux_loss)):
            aux_loss[i].append(avg_loss_aux[i]/len(train_loader))

        model.eval()
        inp = torch.from_numpy(np.array(val_x_))
        labs = torch.from_numpy(np.array(val_y_[:, 0]))
        out1 = model_trend(inp.to(device).float())
        out = model(inp.to(device).float(), nn.Sigmoid()(out1))
        outputs = out[-1].cpu().detach().numpy().reshape(-1)
        targets = labs.numpy().reshape(-1)
        MSE = np.mean((outputs - targets) ** 2)
        val_loss.append(MSE)
        
        for i in range(len(aux_loss)-1):
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
            print(f"Aux loss train: [close: {np.mean(aux_loss[0][-print_nb:]): 0.05f} ], [SMI: {np.mean(aux_loss[1][-print_nb:]): 0.05f} ], [SP500: {np.mean(aux_loss[2][-print_nb:]): 0.05f} ], [bondCH: {np.mean(aux_loss[3][-print_nb:]): 0.05f} ] [bondUS: {np.mean(aux_loss[4][-print_nb:]): 0.05f} ] [Gru base: {np.mean(aux_loss[5][-print_nb:]): 0.05f}]")
            print(f"Aux loss val:   [close: {np.mean(aux_loss_val[0][-print_nb:]): 0.05f} ], [SMI: {np.mean(aux_loss_val[1][-print_nb:]): 0.05f} ], [SP500: {np.mean(aux_loss_val[2][-print_nb:]): 0.05f} ], [bondCH: {np.mean(aux_loss_val[3][-print_nb:]): 0.05f} ] [bondUS: {np.mean(aux_loss_val[4][-print_nb:]): 0.05f} ] [Gru base: {np.mean(aux_loss_val[5][-print_nb:]): 0.05f}]")


        
    return train_loss, val_loss, aux_loss, aux_loss_val
    

def train_AE(
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

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_)

    train_loss = []
    val_loss = []
    
    for epoch in tqdm.tqdm(range(1, num_epochs_ + 1)):

        avg_loss = 0.0
        model.train()
        for x, label in train_loader:
                 
     
            
            outputs = model(x.to(device).float())
            optimizer.zero_grad()
            
            loss = criterion(outputs, torch.cat((x.to(device).float(), label.to(device).float().reshape(-1,5,1)), axis=2)) #torch.cat((x.to(device).float(), label.to(device).float().reshape(-1,5,1)), axis=2)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            
        train_loss.append(avg_loss / (len(train_loader)))

        
        model.eval()
        inp = torch.from_numpy(np.array(val_x_))
        labs = torch.from_numpy(np.array(val_y_))

        outputs= model(inp.to(device).float())
        
        outputs = outputs.cpu().detach().numpy()
        
        MSE = np.mean((outputs -torch.cat((inp.float(), labs.float().reshape(-1,5,1)), axis=2).cpu().detach().numpy()) ** 2) #torch.cat((inp.float(), labs.float().reshape(-1,5,1)), axis=2)
   
    
        val_loss.append(MSE)

        

        if epoch % print_nb == 0:
            print(
                f"Epoch: {epoch}/{num_epochs_}\nMSE = [train loss mean : {np.mean(train_loss[-print_nb:]): .08f}] , [val loss mean: {np.mean(val_loss[-print_nb:]): .08f}, MSE (last){MSE*100: .05f}"
            )


    return train_loss, val_loss
    
def train_AEandGRU(
    model_AE,
    model,
    train_x_,
    train_y_,
    val_x_,
    val_y_,
    seq_len=10,
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

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_)

    train_loss = []
    val_loss = []
    

    model_AE.eval()
    
    for epoch in tqdm.tqdm(range(1, num_epochs_ + 1)):

        avg_loss = 0.0
        
        model.train()
        for x, label in train_loader:
       
            x =  model_AE(x.to(device).float())
            x = np.swapaxes(x.cpu().detach().numpy(),2,1) # x.reshape(-1, seq_len,1)
            
            optimizer.zero_grad()
            outputs = model(torch.from_numpy(x).to(device).float())

            loss = criterion(outputs, label.to(device).float()[:, 0].unsqueeze(1))
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        train_loss.append(avg_loss / (len(train_loader)))

        model.eval()
        inp = torch.from_numpy(np.array(val_x_))
        labs = torch.from_numpy(np.array(val_y_[:, 0]))
        
        out1 = model_AE(inp.to(device).float())
        out1 = np.swapaxes(out1.cpu().detach().numpy(),2,1)
        out = model(torch.from_numpy(out1).to(device).float())
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


def evaluate_AE(model, x_, y_, device="cpu"):
    model.eval()
    
    inp = torch.from_numpy(np.array(x_))
    labs = torch.from_numpy(np.array(y_))

    dec_outputs = model(inp.to(device).float())
   #outputs = outputs.cpu().detach().numpy().reshape(-1)
    dec_outputs = dec_outputs.cpu().detach().numpy()
    #targets = labs.numpy().reshape(-1)
    

    #MSE = np.mean((outputs - targets) ** 2)
    MSE_dec = np.mean((dec_outputs - torch.cat((inp.float(), labs.float().reshape(-1,5,1)), axis=2).cpu().detach().numpy()) ** 2) #torch.cat((inp.float(), labs.float().reshape(-1,5,1)), axis=2)
    
    #print("MSE: {}%".format(MSE * 100))
    print("MSE AE: {}%".format(MSE_dec * 100))
    
    return dec_outputs  #outputs, targets,dec_outputs, MSE, MSE_dec
    
    
def evaluate_AEandGRU(model_AE, model, x_, y_,seq_len=10, device="cpu"):
    inp = torch.from_numpy(np.array(x_))
    labs = torch.from_numpy(np.array(y_[:, 0]))

    dec_outputs = model_AE(inp.to(device).float())
    dec_outputs = np.swapaxes(dec_outputs.cpu().detach().numpy(),2,1)
    outputs = model(torch.from_numpy(dec_outputs).to(device).float())
    outputs = outputs.cpu().detach().numpy().reshape(-1)
   
    targets = labs.numpy().reshape(-1)
    

    MSE = np.mean((outputs - targets) ** 2)

    
    print("MSE: {}%".format(MSE * 100))
    
    
    return outputs, targets, MSE

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
