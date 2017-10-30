# -*- coding: utf-8 -*-
import sys
reload(sys)
#sys.setdefaultencoding("ISO-8859-1")
sys.setdefaultencoding("utf-8")
from flask import Flask,url_for,render_template,request,url_for,redirect,send_from_directory,abort
from werkzeug.utils import secure_filename
import math
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras import optimizers
from keras import metrics
from celery import Celery

broker = 'redis://localhost:6379/0'
backend = 'redis://localhost:6379/1'
flask_celery = Celery('tasks', broker=broker, backend=backend)

def classifier_model(input_dim, hidden_layers, hidden_unit, learning_rate, output_dim):
    model = Sequential()
    model.add(Dense(units=hidden_unit, input_dim=input_dim, kernel_initializer='uniform', activation='relu'))
    for layer in range(hidden_layers):
    	model.add(Dense(units=hidden_unit, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=output_dim, kernel_initializer='uniform', activation='softmax'))
    rmsprop = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-06)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def regressor_model(input_dim, hidden_layers, hidden_unit, learning_rate):
	model = Sequential()
	model.add(Dense(units=hidden_unit, input_dim=input_dim, kernel_initializer='uniform', activation='relu'))
	for layer in range(hidden_layers):
		model.add(Dense(units=hidden_unit, kernel_initializer='uniform', activation='relu'))
	model.add(Dense(1, kernel_initializer='uniform'))
	rmsprop = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-06)
	model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
	return model

def one_hot_encode_object_array(arr):
	'''One hot encode a numpy array of objects (e.g. strings)'''
	uniques, ids = np.unique(arr, return_inverse=True)
	return np_utils.to_categorical(ids, len(uniques))

def loadfile(file_path):
	lines = np.loadtxt(file_path, delimiter=',', skiprows=1, dtype='str')
	Y = lines[:, :1].astype('float')
	X = lines[:, 1:].astype('float')
	return Y, X

@flask_celery.task
def long_time_def(task_id, task, params, train_filename, test_filename, uploadfolder, epochs, batch_size):
	print "start============================================================"
	input_data_train_path = os.path.join(uploadfolder, train_filename)
	input_data_test_path = os.path.join(uploadfolder, test_filename)
	Y_train, X_train = loadfile(input_data_train_path)
	Y_test, X_test = loadfile(input_data_test_path)
	uniques, ids = np.unique(Y_train, return_inverse=True)
	input_dim = X_train.shape[1]
	output_dim = len(uniques)
	scaler = StandardScaler()  
	#  Don't cheat - fit only on training data
	scaler.fit(X_train)  
	X_train = scaler.transform(X_train)  
	#  apply same transformation to test data
	X_test = scaler.transform(X_test)
	
	params['input_dim']=input_dim
	if task == "0":
		Y_train = one_hot_encode_object_array(Y_train)
		Y_test = one_hot_encode_object_array(Y_test)
		params['output_dim']=output_dim
		model = classifier_model(**params)
		model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
		y_pred = model.predict(X_test, batch_size=128)
		acc = model.evaluate(X_test, Y_test, batch_size=128)[-1]
		#acc = metrics.categorical_accuracy(Y_test, y_pred)
		#print Y_test
		#print y_pred
		#acc = float((Y_test == y_pred).sum()) / output_dim / Y_test.shape[0]
		mse = ''
	elif task == "1":
		model = regressor_model(**params)
		model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
		y_pred = model.predict(X_test, batch_size=128)
		mse = mean_squared_error(Y_test, y_pred)
		acc = ''		
	# with open('models/model.pkl', 'w') as store:
	# 	pickle.dump(model, store)
	with open('result/'+task_id+'.txt', 'w') as res:
		for i in y_pred:
			res.write(str(i)+'\n')
	return mse, acc
