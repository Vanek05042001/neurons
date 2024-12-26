# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:20:06 2024

@author: Vanya
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class MLPptorch(nn.Module):
    # как и раньше для инициализации на вход нужно подать размеры входного, скрытого и выходного слоев
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        # nn.Sequential - контейнер модулей
        # он последовательно объединяет и позволяет запускать их одновременно
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size), # слой линейных сумматоров
                                    nn.Sigmoid(),                    # функция активации
                                    nn.Linear(hidden_size, out_size),
                                    nn.Sigmoid(),
        )

    
    # прямой проход    
    def forward(self,x):
        return self.layers(x)

class MLPptorch_NEW(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size):
        nn.Module.__init__(self)
        
        layers = []
        last_size = in_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, out_size))
        layers.append(nn.Sigmoid())  # Выходной слой
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

# функция обучения
def train(net, optimizer, x, y, num_iter):
    for i in range(0,num_iter):
        optimizer.zero_grad() # Обнуление градиентов
        pred = net.forward(x)
        loss = lossFn(pred, y)
        loss.backward()
        optimizer.step()
        if i%(num_iter/10)==0:
           print('Ошибка на ' + str(i) + ' итерации: ', loss.item())
    return loss.item()


df = pd.read_csv('data.csv')
df = df.iloc[np.random.permutation(len(df))]

X = df.iloc[0:100, 0:3].values
y = df.iloc[0:100, 4]
y = y.map({'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor':3}).values.reshape(-1,1)
Y = np.zeros((y.shape[0], np.unique(y).shape[0]))
for i in np.unique(y):
    Y[:,i-1] = np.where(y == i, 1, 0).reshape(1,-1)

X_test = df.iloc[100:150, 0:3].values
y = df.iloc[100:150, 4]
y = y.map({'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor':3}).values.reshape(-1,1)
Y_test = np.zeros((y.shape[0], np.unique(y).shape[0]))
for i in np.unique(y):
    Y_test[:,i-1] = np.where(y == i, 1, 0).reshape(1,-1)


inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = [50] # задаем число нейронов скрытого слоя 
outputSize = Y.shape[1] if len(Y.shape) else 1 # количество выходных сигналов равно количеству классов задачи

#lossFn = nn.MSELoss()
lossFn = nn.BCELoss()

net_relu = MLPptorch_NEW(inputSize,hiddenSizes,outputSize)
optimizer_relu = torch.optim.SGD(net_relu.parameters(), lr=0.05)
loss_relu = train(net_relu, optimizer_relu, torch.from_numpy(X.astype(np.float32)), 
              torch.from_numpy(Y.astype(np.float32)), 5000)

pred = net_relu.forward(torch.from_numpy(X.astype(np.float32))).detach().numpy()
err = sum(abs((pred>0.5)-Y))
print("Ошибки на обучающей выборке ReLU =", err)   

pred = net_relu.forward(torch.from_numpy(X_test.astype(np.float32))).detach().numpy()
err = sum(abs((pred>0.5)-Y_test))
print("Ошибки на тестовой выборке ReLU =", err)

hiddenSizes = 50

net_sigmoid = MLPptorch(inputSize,hiddenSizes,outputSize)
optimizer_sigmoid = torch.optim.SGD(net_sigmoid.parameters(), lr=0.05)
loss_sigmoid = train(net_sigmoid, optimizer_sigmoid, torch.from_numpy(X.astype(np.float32)), 
              torch.from_numpy(Y.astype(np.float32)), 5000)

pred = net_sigmoid.forward(torch.from_numpy(X.astype(np.float32))).detach().numpy()
err = sum(abs((pred>0.5)-Y))
print("Ошибки на обучающей выборке Sigmoid =", err)   

pred = net_sigmoid.forward(torch.from_numpy(X_test.astype(np.float32))).detach().numpy()
err = sum(abs((pred>0.5)-Y_test))
print("Ошибки на тестовой выборке Sigmoid =", err)