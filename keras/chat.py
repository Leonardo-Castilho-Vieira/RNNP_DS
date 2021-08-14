# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 19:49:00 2021

@author: Jasmine Moreira

python -m pip install pyspellchecker
pip install SpeechRecognition
pip install PyAudio
"""
import pandas as pd
import numpy as np
import string
from spellchecker import SpellChecker
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#############################################################################################
#
# PREPARAÇÃO E TREINAMENTO DO MODELO
#
#############################################################################################

sp = SpellChecker(language="pt")

#Carregar perguntas e resspostas
qdf = pd.read_csv (r'C:\Users\jasmi\OneDrive\Área de Trabalho\RNNP\Keras\ChatBot\questions.csv',sep=";")
adf = pd.read_csv (r'C:\Users\jasmi\OneDrive\Área de Trabalho\RNNP\Keras\ChatBot\answers.csv',sep=";")

#Verificar ortografia e colocar em minúsculas
def spck(sentences):
    checked_q = []
    for sentence in sentences:
        q = ""
        sentence = sentence.translate(str.maketrans('','',string.punctuation))
        for word in sentence.lower().split():
          q = q+" "+sp.correction(word)
        checked_q.append(q)
    return checked_q

#Converter perguntas para minúsculas e corrigir a sintaxe    
qdf.question = spck(qdf.question)

max_len = 50        # tamanho máximo da frase
max_words = 5000    # tamanho do dicionário

#Tokenizar e codificar as perguntas
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(qdf.question)
sequences = tokenizer.texts_to_sequences(qdf.question)

#Criar inputs e labels
x_train = pad_sequences(sequences, maxlen=max_len)
y_train = to_categorical(qdf.answer_id)

#Criar o modelo
model = Sequential()
model.add(Embedding(5000, 32, input_length= max_len, embeddings_regularizer=regularizers.L2(0.0001)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(adf.answer_id.max()+1, activation = 'softmax'))

opt = optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=2000, batch_size=len(x_train))


# Plotar gráficos de perda e acurácia

import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
#val_loss_values = history_dict['val_loss']

epochs = range(1, len( history_dict['loss']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training Loss')
#plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#plt.clf()
acc_values = history_dict['accuracy']
#val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training Acc')
#plt.plot(epochs, val_acc_values, 'b', label='Validation Acc')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


###################### Teste do modelo

while True:
    sentence = input("você: ")
    if sentence == 'quit':
        break
    sentence = tokenizer.texts_to_sequences(spck([sentence]))
    sentence = pad_sequences(sentence, maxlen=max_len)
    prediction = model(sentence)
    category = np.argmax(prediction, axis=1)[0]
    answer = adf.query('answer_id=='+str(category)).to_numpy()
    print("Ana: "+answer[0][1])






#############################################################################################
#
# CAPTURA E RECONHECIMENTO DE VOZ E RESPOSTA
#
#############################################################################################

import speech_recognition as sr
import win32com.client as wincl
speak = wincl.Dispatch("SAPI.SpVoice")
speak.Rate=3

#Para MacOS e Linux
# import os
# os.system('say "'+answer+'"')

def recognize_speech_from_mic(recognizer, microphone):
    transcription = ""
    with microphone as source:
        audio = recognizer.listen(source)
        try:
            transcription = recognizer.recognize_google(audio, language="pt-BR")
        except sr.RequestError:
            print("API unavailable")
        except sr.UnknownValueError:
            pass
    return transcription  

recognizer = sr.Recognizer()
microphone = sr.Microphone()

with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

speak.Speak("Olá, sou a Ana, vamos começar?")
while(True):
    sentence = recognize_speech_from_mic(recognizer, microphone)
    if(sentence==""): continue
    print("Você: "+sentence)
    sentence = tokenizer.texts_to_sequences(spck([sentence]))
    sentence = pad_sequences(sentence, maxlen=max_len)
    prediction = model(sentence)
    category = np.argmax(prediction, axis=1)[0]
    answer = adf.query('answer_id=='+str(category)).to_numpy()
    print("Ana: "+answer[0][1])
    speak.Speak(answer[0][1])

#microphone.device_index = 4
#import sounddevice as sd
#print(sd.query_devices()) 

#prediction = model(x_train)
#print(prediction)
#np.argmax(prediction, axis=1) 
