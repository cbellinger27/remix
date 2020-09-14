from sklearn.model_selection import RepeatedStratifiedKFold
import tensorflow as tf
import pandas as pd
import numpy as np 
import remix as ReMix
import Resampler as Resampler
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit, StratifiedKFold

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#DATA GENERATOR TO CREATE MINI-BATCHES FOR TRAINING. THIS FUCTION PERFORMS REMIX ACTIONS BALANCE AND MIX
class DataGenerator(tf.keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, data, labels, batch_size=64, shuffle=True, remixFunction=None, balanceType=None):
    self.y = labels                         # array of labels
    self.X = data                           # array of data
    self.batch_size   = batch_size          # batch size
    self.shuffle      = shuffle             # shuffle bool
    self.remixFunction = remixFunction
    self.balanceType = balanceType
    self.on_epoch_end()
  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(self.X.shape[0] / self.batch_size))
  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(self.X.shape[0])
    if self.shuffle:
      np.random.shuffle(self.indexes)
  def __getitem__(self, index):
    'Generate one batch of data'
    # selects indices of data for next batch
    indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
    batchY = np.array([self.y[k,:] for k in indexes])
    batchX =  np.array([self.X[k,:] for k in indexes])
    if 'none' in self.balanceType:     # PLAIN BATCH
    	return batchX, batchY
    elif 'SMOTE' in self.balanceType:     # APPLY SMOTE TO THE BATCH
    	return Resampler.Resampler.smote(batchX, batchY)
    else:     # MIXUP OR REMIX
    	return self.remixFunction.sample(batchX, batchY, self.balanceType)
    return batchX, batchY


METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]

def get_model(inputDim, outputDim, modelName='best_model'):
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
    # mc = tf.keras.callbacks.ModelCheckpoint(modelName+".h5", monitor='val_loss', mode='min', save_best_only=True)
    # tb = TensorBoard(log_dir="log_"+modelName+".log")
    inp = tf.keras.Input(inputDim)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inp)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    out = tf.keras.layers.Dense(outputDim, activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=METRICS)
    return model


def brierPos(pred, posLab=1):
	return (np.sum((posLab - pred)**2)/pred.shape[1])


def brierNeg(pred, negLab=0):
	return (np.sum((negLab - pred)**2)/pred.shape[1])

def balancedBrier(predPos, predNeg, posLab=1, negLab=0):
	return (brierNeg(predNeg, negLab) + brierPos(predPos, posLab))/2


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

minClsId = 5 # dogo\
majClsId = 0

X = train_images.copy()
y = train_labels[:,-1].copy()

#comment out after verifying code
idx = np.random.choice(len(y), 5000,replace=False)
X = X[idx, :, :, :]
y = y[idx]
#############

noMinIdx = np.where(y!=minClsId)[0]
minIdx = np.where(y==minClsId)[0]
y[noMinIdx] = 0
y[minIdx] = 1

btchSz = 64


allResults = np.ndarray(shape=(0,12))
brierPosResultsAll = np.ndarray(shape=(0,11))
brierNegResultsAll = np.ndarray(shape=(0,11))
brierBalResultsAll = np.ndarray(shape=(0,11))
fmResultsAll       = np.ndarray(shape=(0,11))
gmResultsAll       = np.ndarray(shape=(0,11))
fmResults = np.ndarray(shape=(0,2))
gmResults = np.ndarray(shape=(0,2))
brierPosResults = np.ndarray(shape=(0,2))
brierNegResults = np.ndarray(shape=(0,2))
brierBalResults = np.ndarray(shape=(0,2))
datasetsNum = np.array([])
minSizes = np.array([])
dsNum = 0


for ir in [0.05, 0.1, 0.025]:
	print(ir)
	clsLabs, clsSizes = np.unique(y, return_counts=True)
	outDim    = len(np.bincount(y))
	maxClsIdx = np.argmax(clsSizes)
	minClsIdx = np.argmin(clsSizes)
	maxSize = np.max(clsSizes)
	minSize = np.min(clsSizes)
	if minSize > maxSize * ir:
		minIdx = np.where(y==minClsIdx)[0]
		sbsMinIdx = np.random.choice(minIdx, int(maxSize * ir),replace=False)
		minDel = np.setdiff1d(minIdx, sbsMinIdx)
		imbY = np.delete(y, minDel)
		imbX = np.delete(X, minDel, axis=0)
	for techType, a in [['none',0],['WeightedBase',0],['SMOTE', 0],['mixup',0.2],['remix',0.2]]:
		rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=10, random_state=36851234)
		tmpGm = np.array([])
		tmpFm = np.array([])
		tmpBp = np.array([])
		tmpBn = np.array([])
		tmpBb = np.array([])
		for train_index, test_index in rskf.split(imbX, imbY):
			X_train, X_test = imbX[train_index, :], imbX[test_index, :]
			y_train, y_test = imbY[train_index], imbY[test_index]
			ss = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=568453)
			val_index, _ = ss.split(X_train, y_train)
			X_train2, X_val = X_train[val_index[0]], X_train[val_index[1]]
			y_train2, y_val = y_train[val_index[0]], y_train[val_index[1]]
			y_trainEncoded = tf.keras.utils.to_categorical(y_train2)
			y_valEncoded = tf.keras.utils.to_categorical(y_val)
			y_testEncoded = tf.keras.utils.to_categorical(y_test)
			y_trainEncoded = tf.keras.utils.to_categorical(y_train2)
			model = get_model(X_train2[0].shape, outDim)
			if techType == 'SMOTE':
				#Batch SMOTE
				model = get_model(X_train2[0].shape, outDim)
				mu = ReMix.ReMix(alpha=None)
				train_data = DataGenerator(X_train2, y_trainEncoded, batch_size=btchSz, remixFunction=mu, balanceType="SMOTE")
				reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
				model.fit(train_data, epochs=500, shuffle=True,verbose=0,validation_data=(X_val, y_valEncoded), callbacks=[reduce_lr])
			elif techType == 'WeightedBase':
				class_weights = compute_class_weight('balanced', np.unique(y_train2), y_train2)
				model = get_model(X_train2[0].shape, outDim)
				reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
				model.fit(X_train2, y_trainEncoded, batch_size=btchSz, epochs=500, class_weight=class_weights, shuffle=True,verbose=0,validation_data=(X_val, y_valEncoded),callbacks=[reduce_lr])
			elif techType == 'mixup':
				model = get_model(X_train2[0].shape, outDim)
				mu = ReMix.ReMix(alpha=a)
				train_data = DataGenerator(X_train2, y_trainEncoded, batch_size=btchSz, remixFunction=mu, balanceType="mixup")
				reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
				model.fit(train_data, epochs=500, shuffle=True,verbose=0,validation_data=(X_val, y_valEncoded), callbacks=[reduce_lr])
			elif techType == 'remix':
				model = get_model(X_train2[0].shape, outDim)
				mu = ReMix.ReMix(alpha=a)
				train_data = DataGenerator(X_train2, y_trainEncoded, batch_size=btchSz, remixFunction=mu, balanceType="remix")
				reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
				model.fit(train_data, epochs=500, shuffle=True,verbose=0,validation_data=(X_val, y_valEncoded), callbacks=[reduce_lr])
			else:
				model = get_model(X_train2[0].shape, outDim)
				reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
				model.fit(X_train2, y_trainEncoded, batch_size=btchSz, epochs=500, shuffle=True, validation_data=(X_val, y_valEncoded),verbose=0,callbacks=[reduce_lr])
			y_prob = model.predict(X_test)
			y_pred = np.argmax(y_prob,axis=1)
			tmpFm = np.append(tmpFm, f1_score(y_test, y_pred, average='macro'))
			tmpGm = np.append(tmpGm, geometric_mean_score(y_test, y_pred))
			tmpBp = np.append(tmpBb, brierPos(y_prob[np.where(y_test==1),1]))
			tmpBn = np.append(tmpBn, brierNeg(y_prob[np.where(y_test==0),0], negLab=1))
			tmpBb = np.append(tmpBb, balancedBrier(y_prob[np.where(y_test==1),1], y_prob[np.where(y_test==0),1], posLab=1, negLab=1))
		fmResults = np.concatenate((fmResults, np.array([np.mean(tmpFm), np.std(tmpFm)]).reshape(1,2)),axis=0)
		gmResults = np.concatenate((gmResults, np.array([np.mean(tmpGm), np.std(tmpGm)]).reshape(1,2)),axis=0)
		brierPosResults = np.concatenate((brierPosResults, np.array([np.mean(tmpBp), np.std(tmpBp)]).reshape(1,2)),axis=0)
		brierNegResults = np.concatenate((brierNegResults, np.array([np.mean(tmpBn), np.std(tmpBn)]).reshape(1,2)),axis=0)
		brierBalResults = np.concatenate((brierBalResults, np.array([np.mean(tmpBb), np.std(tmpBb)]).reshape(1,2)),axis=0)
		datasetsNum = np.append(datasetsNum, dsNum)
	print(ir)
	print(fmResults)
	print(gmResults)
	print(brierBalResults)
	print(brierPosResults)
	print(brierNegResults)


	# 	f.write(mixNames)
	# 	f.write(str(aveRep))
	# 	f.write('\n')
	# 	allFmResults = np.concatenate((allFmResults, fmResults.reshape(1,11)))
	# 	allGmResults = np.concatenate((allGmResults, gmResults.reshape(1,11)))
	# print(allFmResults)
	# print(allGmResults)
	# f.close()
	# btchSzIdx += 1

