# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:54:12 2021

@author: Mutunga
"""
#DATA PREPROCESSING

#Importing Libraries
import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

#Importing the dataset
dataset=pd.read_csv("Churn_Modelling.csv")
X=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values
print(X)
print(y)

#Label encoding of the Categorocal Variable 'Gender'
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder() 
X[:,2]=le.fit_transform(X[:,2])
print(X)

#OneHotEncoding of the variable 'Geography'
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])], remainder='passthrough')  
X=np.array(ct.fit_transform(X))
print(X)

#Splitting the dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

#Feature Scaling  
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train) 
X_test=sc.transform(X_test) 
print(X_train)
print(X_test)

#BUILDING THE ANN

#Initializing the ANN as a sequence of layers
ann=tf.keras.models.Sequential() #Creating ann as an instance of the Sequential class from the keras library in tensorflow module

#Adding the input and hidden layers
ann.add(tf.keras.layers.Dense(units=6,activation='relu')) # The hyperparemeter 6 is the number of hidden neurons, input neurons= # of features

#Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid')) #For multiclass use activation='softmax' see tf.keras.activations. for documentations

#TRAINING THE ANN

#Compiling the ANN with an optimiser, loss function and accuracy metric
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) #see tf.keras.optimizers(loss or metric).

#Training the ANN
ann.fit(X_train, y_train, batch_size=32, epochs=100) #Hyperparameter batch_size; we are comparing our predictions to a bacth of 32 real values as opposed to one by one

#Making predictions for a new client
prediction=ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))
print(prediction) #Returns a probability owing to our output layer activation function


#Prediction inform of a classification as to whether or not this particular client left the bank
Exited=ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))
print(Exited>5)
 
#Predicting the test results
y_pred = ann.predict(X_test)
y_pred=(y_pred>0.5) #Update y_pred
Predictions=np.concatenate((y_test.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1)),1)
print(Predictions)

#Constructing the confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score #or from sklearn.metrics import confusion_matrix, accuracy_score
As = accuracy_score(y_test, y_pred)
print(As*100, '%')
