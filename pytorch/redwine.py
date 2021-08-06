# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 13:30:15 2021
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

# Verificar se o CUDA está disponível
use_cuda = t.cuda.is_available()
device = t.device("cuda:0" if use_cuda else "cpu") 

# Carregar dataset
df = pd.read_csv (r'C:\Users\jasmi\OneDrive\Área de Trabalho\RNNP_DS\PyTorch\redwine.csv')
df = df.dropna()

# Separar dados de treinamento e dados de teste
train_set = df.sample(frac=0.7)
test_set = df[df.index.isin(train_set.index)==False]

# Extrair labels para treinamento e teste
train_labels = t.as_tensor(train_set.quality.values).to(device)
test_labels = t.as_tensor(test_set.quality.values).to(device)

# Eliminar features que não serão utilizadas como inputs
train_set =  train_set.iloc[: , :-1]
test_set =  test_set.iloc[: , :-1]

# Transformar as features em tensores
train_set = t.tensor(train_set.values, dtype=t.float32).to(device)
test_set = t.tensor(test_set.values, dtype=t.float32).to(device)

# Identificar número de features para parametrização da rede
n_feat = len(list(train_set[1]))

# Criar Modelo
net = nn.Sequential(nn.Linear(n_feat, 20),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(20, 10),
                    nn.ReLU(),
                    nn.Linear(10, 9)
                   )
if use_cuda: net.cuda()


# Criar funções de cálculo de erro e otimização
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.AdamW(net.parameters(), lr=0.001) #e-1

# Executar loop de treinamento
epoch = 30
loss_values = []
for epoch in range(epoch):
    epoch_loss = 0
    for x,y in zip(train_set, train_labels):    
        optimizer.zero_grad()
        output = net(x.view(1,n_feat))
        loss = criterion(output, y.view(1))
        loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss+loss.item()       
    loss_values.append(epoch_loss/len(train_set))
    print(epoch_loss/len(train_set))
    
# Calcular acurácia
with t.no_grad():
    correct = 0
    total = 0
    for x,y in zip(test_set, test_labels):      
        output = net(x.view(1,n_feat))
        if t.argmax(output) == y:
            correct +=1
        total +=1
print(f'accuracy: {round(correct/total, 3)}')

# Plotar gráfico de covergência do erro
plt.plot(range(1,epoch+2), loss_values, 'bo', label='Training Loss')
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
