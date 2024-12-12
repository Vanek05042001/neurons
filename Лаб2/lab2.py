# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:48:13 2024

@author: Vanya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import random

class MLP:
    
    def __init__(self, inputSize, outputSize, learning_rate=0.1, hiddenSizes = 5):

        # инициализируем нейронную сеть 
        # веса инициализируем случайными числами, но теперь будем хранить их списком
        self.weights = [
            np.random.uniform(-2, 2, size=(inputSize,hiddenSizes)),  # веса скрытого слоя
            np.random.uniform(-2, 2, size=(hiddenSizes,outputSize))  # веса выходного слоя
        ]
        self.learning_rate = learning_rate
        self.layers = None

    # сигмоида
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # производная от сигмоиды
    def derivative_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
     
    # прямой проход 
    def feed_forward(self, x):
        input_ = x # входные сигналы
        hidden_ = self.sigmoid(np.dot(input_, self.weights[0])) # выход скрытого слоя = сигмоида(входные сигналы*веса скрытого слоя)
        output_ = self.sigmoid(np.dot(hidden_, self.weights[1]))# выход сети (последнего слоя) = сигмоида(выход скрытого слоя*веса выходного слоя)
        
        self.layers = [input_, hidden_, output_]
        return self.layers[-1]
    
   
    # на вход принимает скорость обучения, реальные ответы, предсказанные сетью ответы и выходы всех слоев после прямого прохода
    def backward(self, target):
    
        # считаем производную ошибки сети
        err = (target - self.layers[-1])
    
        # прогоняем производную ошибки обратно ко входу, считая градиенты и корректируя веса
        # цикл перебирает слои от последнего к первому
        for i in range(len(self.layers)-1, 0, -1):
            # градиент слоя = ошибка слоя * производную функции активации * на входные сигналы слоя
            
            # ошибка слоя * производную функции активации
            err_delta = err * self.derivative_sigmoid(self.layers[i])
    
            # пробрасываем ошибку на предыдущий слой
            err = np.dot(err_delta, self.weights[i - 1].T)
    
            # ошибка слоя * производную функции активации * на входные сигналы слоя
            dw = np.dot(self.layers[i - 1].T, err_delta)
    
            # обновляем веса слоя
            self.weights[i - 1] += self.learning_rate * dw
            
            
    
    # обучаем модель на одном образце
    def train_single_sample(self, x_value, target):
        self.feed_forward(x_value)
        self.backward(target)

    # стохастическое обучение
    def train_stochastic(self, x_values, targets):
        for x_value, target in zip(x_values, targets):
            # random.shuffle(x_values)
            x_value = random.choice(x_values)
            # x_values = np.delete(x_values, x_value)
            self.train_single_sample(x_value.reshape(1, -1), target.reshape(1, -1))
    
    # функция предсказания возвращает только выход последнего слоя
    def predict(self, x_values):
        return self.feed_forward(x_values)


df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]

y = df.iloc[0:100, 4].values
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

X = df.iloc[0:100, [0, 1, 2, 3]].values

inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 10 # задаем число нейронов скрытого (А) слоя 
outputSize = y.shape[1] # количество выходных сигналов равно количеству классов задачи

iterations = 50
learning_rate = 0.1

net = MLP(inputSize, outputSize, learning_rate, hiddenSizes)

# обучаем сеть
for i in range(iterations):  
    net.train_stochastic(X, y)

    # i = random.randint(0, iterations)
    if i % 10 == 0:
        print("На итерации: " + str(i) + ' || ' + "Средняя ошибка: " + str(np.mean(np.square(y - net.predict(X)))))

# считаем ошибку на обучающей выборке
pr = net.predict(X)
predicted_classes = np.argmax(pr, axis=1)
true_classes = np.argmax(y, axis=1)
num_incorrect = np.sum(predicted_classes != true_classes)
print("Количество неверных ответов:", num_incorrect)