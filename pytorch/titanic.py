  
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:18:20 2021
@author: Jasmmine Moreira


1) Preparar dados
2) Criar o modelo (input, output size, forward pass)
3) Criar a função de erro (loss) e o otimizador 
4) Criar o loop de treinamento
   - forward pass: calcular a predição e o erro
   - backward pass: calcular os gradientes
   - update weights: ajuste dos pesos do modelo
"""
import pandas as pd
import torch as t
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt


df = pd.read_csv (r'C:\Users\jasmi\Downloads\titanic\train.csv')
df = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked", "PassengerId","Survived"]]

df['Embarked'] = pd.Categorical(df['Embarked']).codes
df['Sex'] = pd.Categorical(df['Sex']).codes

train_set = df.sample(frac=0.7)
test_set = df[df.PassengerId.isin(train_set.PassengerId)==False]

num_classes = 2
train_labels = (train_set.Survived == torch.arange(num_classes).reshape(1, num_classes)).float()

num_classes = 2
test_labels = (test_set.Survived == torch.arange(num_classes).reshape(1, num_classes)).float()

train_set =  train_set.iloc[: , :-2]
test_set =  test_set.iloc[: , :-2]

train_set = torch.from_numpy(train_set) 
test_set = torch.from_numpy(test_set) 



use_cuda = t.cuda.is_available()
device = t.device("cuda:0" if use_cuda else "cpu") 

# Criar Modelo
net = nn.Sequential(nn.Linear(, 50),
                      nn.ReLU(),
                      nn.Linear(50, 10)
                      )
if use_cuda:
    net.cuda()

# Erro e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.AdamW(net.parameters(), lr=0.01) #e-1


# loop de treinamento
epoch = 10

loss_values = []
for epoch in range(epoch):
    epoch_loss = 0
    for data in train_loader:
        x, y = data
        optimizer.zero_grad()
        output = net(x.view(-1, 28*28).to(device)).to(device)
        loss = criterion(output, y.to(device))
        loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss+loss.item()       
    loss_values.append(epoch_loss/len(train_loader))
    print(epoch_loss/len(train_loader))
    
plt.plot(range(1,epoch+2), loss_values, 'bo', label='Training Loss')
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

      
with t.no_grad():
    correct = 0
    total = 0
    for data in test_loader:      
        x, y = data
        output = net(x.view(-1, 784).to(device))
        for idx, i in enumerate(output):
            if t.argmax(i) == y[idx]:
                correct +=1
            total +=1
print(f'accuracy: {round(correct/total, 3)}')
