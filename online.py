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
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import uuid
from celery import Celery
from tasks import long_time_def
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import np_utils
from keras.optimizers import RMSprop



result_dic = {}
UPLOAD_FOLDER='upload_files'
RESULT_FOLDER='result'
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER


@app.route('/',methods=['GET','POST'])
def index():
	return "hello"

@app.route('/online',methods=['GET','POST'])
def train():
	if request.method=='POST':
		if request.form["action"] == "Upload":
			task_id = str(uuid.uuid1())
			task = request.form['task']
			hidden_layers = request.form['hidden_layers']
			if hidden_layers == '':
				hidden_layers = 2
			else:
				hidden_layers = int(hidden_layers)

			hidden_unit = request.form['hidden_unit']
			if hidden_unit == '':
				hidden_unit = 5
			else:
				hidden_unit = int(hidden_unit)

			epochs = request.form['epochs']
			if epochs == '':
				epochs = 2
			else:
				epochs = int(epochs)

			batch_size = request.form['batch_size']
			if batch_size == '':
				batch_size = 128
			else: 
				batch_size = int(batch_size)

			learning_rate = request.form['learning_rate']
			if learning_rate == '':
				learning_rate = 0.001
			else: 
				learning_rate = float(learning_rate)

			train_file = request.files["train_file"]
			train_filename = secure_filename(train_file.filename)
			train_file.save(os.path.join(app.config['UPLOAD_FOLDER'], train_filename))
			test_file = request.files["test_file"]
			test_filename = secure_filename(test_file.filename)
			test_file.save(os.path.join(app.config['UPLOAD_FOLDER'], test_filename))
			
			params=dict(hidden_layers=hidden_layers, hidden_unit=hidden_unit, learning_rate=learning_rate)
			#return render_template('online.html', TASK_ID=task_id, HL=hidden_layers, HU=hidden_unit, MI=max_iter, BS=batch_size, LR=learning_rate)
			result=long_time_def.apply_async(args=[task_id, task, params, train_filename, test_filename, app.config['UPLOAD_FOLDER'], epochs, batch_size])
			result_dic[task_id]=result
			#kwarg1='task_id', kwarg2='task', kwarg3='params', kwarg4='train_filename', kwarg5='test_filename')
			return render_template('online.html', TASK_ID=task_id, HL=hidden_layers, HU=hidden_unit, EP=epochs, BS=batch_size, LR=learning_rate)

		if request.form["action"] == "Check":
			task_id = request.form["task_id"]
			result = result_dic.get(task_id)
			if result == None:
				return render_template('online.html', STATUS="not valid", TASK_ID=task_id)
			elif result.ready():
				mse, acc = result.get()
				return render_template('online.html', STATUS="ready", TASK_ID=task_id, MSE=mse, ACC=acc)
			else:
				return render_template('online.html', STATUS="not ready", TASK_ID=task_id)

		if request.form["action"] == "Download":
			task_id = request.form["task_id"]
			result = result_dic.get(task_id)
			if result == None:
				return render_template('online.html', STATUS="not valid", TASK_ID=task_id)
			elif result.ready():
				print "download"
				mse, acc = result.get()
				filename = task_id+".txt"
				#render_template('online.html', STATUS="ready", TASK_ID=task_id, MSE=mse, ACC=acc)
				if os.path.isfile(os.path.join(app.config['RESULT_FOLDER'], filename)):
					return send_from_directory(app.config['RESULT_FOLDER'],filename, as_attachment=True)
			else:
				return render_template('online.html', STATUS="not ready", TASK_ID=task_id)

	return render_template('online.html')

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=2333, threaded=True)  # debug model only for debuging, delete it for production use.
