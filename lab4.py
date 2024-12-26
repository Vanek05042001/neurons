# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:26:25 2024

@author: Vanya
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from torch.utils.data import Subset
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Устройство
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Параметры загрузки данных
train_dir = './animals/train'
val_dir = './animals/val'
batch_size = 100
num_epochs = 10

# Преобразования данных
data_transforms = transforms.Compose([
    transforms.Resize((28, 28)),  # Изменяем размер как для MNIST
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Загрузка данных
train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=data_transforms)
val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=data_transforms)

# Отбор случайных 10% данных
train_indices = random.sample(range(len(train_dataset)), len(train_dataset) // 10)
val_indices = random.sample(range(len(val_dataset)), len(val_dataset) // 10)

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)

# DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)

# Проверка классов
class_names = train_dataset.classes
print("Классы:", class_names)
num_classes = len(class_names)

# Визуализация данных
def imshow(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.pause(5)

inputs, classes = next(iter(train_loader))
out = torchvision.utils.make_grid(inputs)
imshow(out)

# Архитектура сети
class CnNet(nn.Module):
    def __init__(self, num_classes):
        super(CnNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # 3 входных канала (rgb), 32 выходных канала (фильтра)
            nn.BatchNorm2d(32), # батч-нормализация (нормализация выхода)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # уменьшает размеры пространственных измерений вдвое
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(256 * 1 * 1, 512) # преобразует плоский вектор из предыдущих слоёв в вектор размером 512
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # вероятность отключения каждого нейрона 0.5 для регуляризации и предотвращения переобучения
        self.fc2 = nn.Linear(512, num_classes) # выводит вектор размером num_classes
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Модель, функция потерь и оптимизатор
net = CnNet(num_classes).to(device)
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Обучение модели
def train_model(model, criterion, optimizer, num_epochs=2):
    t = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0) # Общая потеря на мини-батче
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_subset)
        epoch_acc = running_corrects.double() / len(train_subset)
        
        print(f'Эпоха {epoch}/{num_epochs - 1} Ошибка: {epoch_loss:.4f} Точность: {epoch_acc:.4f}')
        
    print(f'Обучение завершилось за {time.time() - t:.0f} с')

train_model(net, lossFn, optimizer, num_epochs)

# Оценка точности
def evaluate_model(model, dataloader):
    model.eval()
    running_corrects = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        
    accuracy = running_corrects.double() / len(val_subset)
    print(f'Тестовая точность: {accuracy:.4f}')

evaluate_model(net, val_loader)

# и отдельно загрузим тестовый набор
test_dataset = torchvision.datasets.ImageFolder(root='./animals/val',
                                             transform=data_transforms)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                    shuffle=True, num_workers=2) 
# Реализуем отображение картинок и их класса, предсказанного сетью
inputs, classes = next(iter(test_loader))
pred = net(inputs.to(device))
_, pred_class = torch.max(pred.data, 1)
for i,j in zip(inputs, pred_class):
    img = i.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(class_names[j])
    plt.pause(2)
    
    # построим на картинке
img = torchvision.utils.make_grid(inputs, nrow = 5) # метод делает сетку из картинок
img = img.numpy().transpose((1, 2, 0)) # транспонируем для отображения в картинке
plt.imshow(img)

# Сохранение модели
torch.save(net.state_dict(), 'CnNet_animals_2.ckpt')