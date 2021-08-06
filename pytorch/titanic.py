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
import torch.nn as nn
import matplotlib.pyplot as plt

df = pd.read_csv (r'C:\Users\jasmi\OneDrive\Área de Trabalho\RNNP_DS\PyTorch\titanic.csv')
df = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked", "PassengerId","Survived"]]
df = df.dropna()

df['Embarked'] = pd.Categorical(df['Embarked']).codes
df['Sex'] = pd.Categorical(df['Sex']).codes

train_set = df.sample(frac=0.7)
test_set = df[df.PassengerId.isin(train_set.PassengerId)==False]

train_labels = t.as_tensor(train_set.Survived.values)
test_labels = t.as_tensor(test_set.Survived.values)

train_set =  train_set.iloc[: , :-2]
test_set =  test_set.iloc[: , :-2]

n_feat = len(train_set.columns)


train_set = t.tensor(train_set.values, dtype=t.float32)
test_set = t.tensor(test_set.values, dtype=t.float32)

use_cuda = t.cuda.is_available()
device = t.device("cuda:0" if use_cuda else "cpu") 

# Criar Modelo
net = nn.Sequential(nn.Linear(n_feat, 20),
                    nn.ReLU(),
                    nn.Linear(20, 2)
                   )
if use_cuda:
    net.cuda()

# Erro e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.AdamW(net.parameters(), lr=0.01) #e-1

# loop de treinamento
epoch = 20
loss_values = []
for epoch in range(epoch):
    epoch_loss = 0
    for x,y in zip(train_set, train_labels):    
        optimizer.zero_grad()
        output = net(x.view(1,n_feat).to(device)).to(device)
        loss = criterion(output, y.view(1).to(device))
        loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss+loss.item()       
    loss_values.append(epoch_loss/len(train_set))
    print(epoch_loss/len(train_set))
    
plt.plot(range(1,epoch+2), loss_values, 'bo', label='Training Loss')
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
      
with t.no_grad():
    correct = 0
    total = 0
    for x,y in zip(test_set, test_labels):      
        output = net(x.view(1,n_feat).to(device)).to(device)
        if t.argmax(output) == y:
            correct +=1
        total +=1
print(f'accuracy: {round(correct/total, 3)}')
