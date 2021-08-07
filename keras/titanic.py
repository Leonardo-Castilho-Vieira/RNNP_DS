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
   
   
requisitos:
conda install tensorflow
conda install keras
"""
import pandas as pd
import tensorflow as tf
from keras import models, layers
from tensorflow.keras.utils import to_categorical

# Carregamento dos dados
df = pd.read_csv (r'C:\Users\jasmi\OneDrive\Área de Trabalho\RNNP_DS\PyTorch\titanic.csv')
df = df[["Pclass","Sex","SibSp","Parch","Fare","Embarked", "PassengerId","Survived"]]
df = df.dropna()

# Tranformar variáveis categóricas
df['Embarked'] = pd.Categorical(df['Embarked']).codes
df['Sex'] = pd.Categorical(df['Sex']).codes

# Seleção da base de treinamento e de teste
train_set = df.sample(frac=0.7)
test_set = df[df.PassengerId.isin(train_set.PassengerId)==False]

# Extração dos labels
train_labels = to_categorical(train_set.Survived)
test_labels = to_categorical(test_set.Survived)

# Remover features que não serão utilizadas no treinamento
train_set =  train_set.iloc[: , :-2]
test_set =  test_set.iloc[: , :-2]

# Normalizar dados
#train_set=(train_set-train_set.min())/(train_set.max()-train_set.min())
#test_set=(test_set-test_set.min())/(test_set.max()-test_set.min())

# Conversão do dataframe para tensores
train_set = tf.convert_to_tensor(train_set, dtype=tf.int64) 
test_set = tf.convert_to_tensor(test_set, dtype=tf.int64) 

# Criação do modelo
network = models.Sequential()
network.add(layers.Dense(50, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(6,)))
network.add(layers.Dense(20, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
network.add(layers.Dense(2, activation='softmax'))

network.summary()

# Compilação do modelo para definir função de erro e otimização
network.compile(optimizer=optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy', metrics=['accuracy'])

# Executar treinamento
history = network.fit(train_set, 
                      train_labels, 
                      epochs=60, 
                      batch_size=64, 
                      validation_data=(test_set,test_labels))


# Imprimir perda e acurácia do modelo treinado
test_loss, test_acc = network.evaluate(test_set, test_labels)
print('test_acc: ', test_acc)


# Plotar gráficos de perda e acurácia
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len( history_dict['loss']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training Acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation Acc')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
