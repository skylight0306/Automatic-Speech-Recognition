from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # warning level
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # use GPU 0
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from utilities import *
from sklearn.metrics import confusion_matrix


def runModel():
	global sess
	global predY
	global testAcc
	  
	# Set configs
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:
	
		# Restore model weights from previously saved model
		saver = tf.train.import_meta_graph(model_path +'model.ckpt.meta')
		saver.restore(sess,tf.train.latest_checkpoint(model_path))
		print("Model is restored" )

		# Restore model variables from previously saved model
		graph = tf.get_default_graph()
		accuracy = graph.get_tensor_by_name("Accuracy:0")
		x = graph.get_tensor_by_name("x:0")
		y = graph.get_tensor_by_name("y:0")
		pred = graph.get_tensor_by_name("pred:0")

		# run the model with test data
		testAcc, predY = sess.run([accuracy, pred], feed_dict={x: testX, y: testYcat})
		
		# save the prediction
		directory = os.path.dirname(predsPath)
		try:
			os.stat(directory)
		except:
			os.mkdir(directory)
		np.save( predsPath + 'predY.npy', predY)
	
	return


def personalResults(): # this function calculates the confusion matrix for each ID
	predClass = np.argmax(predY,1)
	testY.ravel()
	'''for i in range(len(predClass)):
		if predClass[i] == testY[i]: 
			print(i)
			print(predY[i])
			print(predClass[i],testY[i])'''
	C = confusion_matrix(testY.ravel(), predClass.ravel(), np.arange(n_classes))
	np.set_printoptions(suppress=True)
	print(C)
	return C



def loadTestVars():
	global testX
	global testY
	global testYcat
	global n_input
	
	# load test variables
	testX = np.load(vars_path + 'testX.npy')
	testX = testX.reshape(len(testX),n_steps,n_hidden)
	n_input = (testX.shape[2])
	testX = np.reshape(testX,(-1,n_steps,n_input))
	testY = np.load(vars_path +'testY.npy')
	testYcat = np.load(vars_path +'testYcat.npy')
	testYcat = testYcat.reshape(len(testYcat),n_classes)
	return

# --------******** main *********----------

# Hyper-parameters
n_classes = 3
n_hidden = 11
n_steps = 20

# data paths
vars_path = './data/'
models_path = '/MFCC_CNN/model/'
#res_path = models_path + 'resA.txt'
#os.system('rm ' + res_path)
outArr = np.zeros((1,3)) # array of 8 elements for the output

# ***load models***

# path of saving models and their results
model_path = models_path 
predsPath = '../preds/'
		
# loop over all patients
loadTestVars() #load variables
tf.reset_default_graph() #reset the graph
runModel() #run the full model with new test data
C = personalResults() #see the results
 #store confusion matrices

outArr += calc_tables(C, n_classes)

print("final results:\n", outArr)
