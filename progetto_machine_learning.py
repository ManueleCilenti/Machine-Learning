# -*- coding: utf-8 -*-

# 
import os
import numpy as np
from numpy import array
from numpy import hstack
import tensorflow
import pandas as pd
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
#from keras.optimizers import sgd
from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense


os.chdir(r'C:\Users\Cilen\OneDrive\Documenti\Progetto machine learning')
#####################

########## Genero sequenze di dati

# splitto una sequenza UNIvariata
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        ### arrivo a fine sequenza
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        ### vedo se vado oltre la sequenza
        if out_end_ix > len(sequence):
            break
#  genero gli input e gli output
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

##################

#splitto una sequenza MULTIvariata
# def split_sequences(sequences, n_steps_in, n_steps_out):
#     X, y = list(), list()
#     for i in range(len(sequences)):
# # ### arrivo a fine sequenza
#         end_ix = i + n_steps_in
#         out_end_ix = end_ix + n_steps_out-1
#         ### vedo se vado oltre la sequenza
#         if out_end_ix > len(sequences): 
#             break
#         # genero gli input e gli output
#         seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix-1:out_end_ix, -1]
#         X.append(seq_x)
#         y.append(seq_y)
#     return array(X), array(y)

#data1=pd.read_csv('pfizer.csv')
#data1=pd.read_csv('moderna.csv')
data1=pd.read_csv('ASTRZ.csv')
#data1=pd.read_csv('JNJ.csv')



## creo colonne temporali per legare il dato al tempo
data1['date']=pd.to_datetime(data1['date'])
data1['week']=data1['date'].dt.week
data1['day']=data1['date'].dt.dayofweek
data1['month']=data1['date'].dt.month
data1['quarter']=data1['date'].dt.quarter

data1=data1.set_index('date')

### dati usati per fare la prevsione
#giorni precedenti
n_steps_in=20
#giorni di previsione
### dati usati su cui fare la previsione
n_steps_out=3

##### caso multivariato
# data2=np.array(data1[['day','week','month','quarter','soglia']])
#
# X, y = split_sequences(data2, n_steps_in,n_steps_out)
#print(X.shape, y.shape)
#for i in range(len(X)):
#    print(X[i], y[i])
#
## flatten input
# n_input = X.shape[1] * X.shape[2]
# X = X.reshape((X.shape[0], n_input))
# n_output = y.shape[1]
# y=y.reshape((y.shape[0], n_output))


### caso univariato
data2=list(np.array(data1[['soglia']]))
X, y = split_sequence(data2, n_steps_in,n_steps_out)
print(X.shape, y.shape)
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
n_output = y.shape[1]
y=y.reshape((y.shape[0], n_output))
for i in range(len(X)):
    print(X[i], y[i])
data2=hstack(data2)



    
##### MLP 
#opt = SGD(lr=0.01, momentum=0.9)
model = Sequential()
model.add(Dense(16, activation='sigmoid', input_dim=n_input))
#model.add(Dense(8, activation='sigmoid'))
model.add(Dense(n_steps_out))
model.compile(optimizer='sgd', loss='binary_crossentropy',  metrics=['binary_accuracy'])
##### validation_split=0.2
model.fit(X, y, epochs=200, verbose=1, shuffle='False', validation_split=0.2)
model.summary()

### generiamo previsione 
yhat = model.predict(X, verbose=0)


# Posso anche testare su dati test
#test_results = model.evaluate(X_test, Y_test, verbose=1)
#print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
#### converto output in 0 1 secondo soglia (0.5)
####

##### valori sopra a 0.5 diventano uno, sotto vanno a 0
y_conf=np.where(yhat>=0.5,1,0)


#### confusion matrix
#### genero vettori da matrici allo scopo di creare confusion matrix
yy=y.reshape(y.shape[0]*y.shape[1])
yy_conf=y_conf.reshape(y_conf.shape[0]*y_conf.shape[1])


confusion_matrix(yy, yy_conf)

### risultati classificazione
print(classification_report(yy, yy_conf, target_names=['0', '1']))