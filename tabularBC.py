import tensorflow as tf
import pandas as pd
import numpy as np 
import remix as ReMix
from functools import reduce
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score, balanced_accuracy_score,classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit,StratifiedKFold


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


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
    if 'remix' in self.balanceType:     # IF WE WANT TO BALANCE THE BATCH
      augmentedX = np.ndarray(shape=(0,self.X.shape[1]))
      augmentedY = np.array([])
      rsmplFunction = RandomOverSampler()
      tmpY = np.argmax(batchY,axis=1)
      clsLabs, clsSizes = np.unique(tmpY, return_counts=True)
      if np.argmin(clsSizes) > 0:
        batchX, batchY = rsmplFunction.fit_resample(batchX, tmpY)     #PERFORM RANDOM OVERSAMPLING TO BALANCE THE CLASSES IN THE BATCH
        cBatchSz = int(np.round(self.batch_size/len(clsLabs)))        #DETERMIN HOW MANY EXAMPLES OF EACH CLASS SHOULD BE PRESENT
        for c in clsLabs:                                             #SELECT THE REQUIRED NUMBER OF SAMPLES FOR EACH CLASS
          tmpIdx = np.random.choice(np.where(batchY==c)[0], cBatchSz, replace=np.sum(batchY==c)<cBatchSz)
          augmentedX = np.concatenate((augmentedX, batchX[tmpIdx,:]))
          augmentedY = np.append(augmentedY, batchY[tmpIdx])
        if len(augmentedY) < self.batch_size:
          idx = np.random.choice(len(batchY), self.batch_size-len(augmentedY))
          augmentedX = np.concatenate((augmentedX, batchX[idx,:]))
          augmentedY = np.append(augmentedY, batchY[idx])
        batchX = augmentedX
        batchY = tf.keras.utils.to_categorical(augmentedY).astype(int)
        idx = np.random.choice(batchX.shape[0], np.min([batchX.shape[0],self.batch_size]), replace=False)   # SELECT SUBSET EQUAL IN SIZE TO ORGINAL BATCH SIZE
        batchX = batchX[idx,:]
        batchY = batchY[idx,:]
    x_out, y_out = self.remixFunction.sample(batchX, batchY, self.balanceType)
    return x_out, y_out


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

def get_model(inputDim, outputDim, hiddenSize, modelName='best_model'):
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
    # mc = tf.keras.callbacks.ModelCheckpoint(modelName+".h5", monitor='val_loss', mode='min', save_best_only=True)
    # tb = TensorBoard(log_dir="log_"+modelName+".log")
    inp = tf.keras.Input((inputDim,))
    x = tf.keras.layers.Dense(hiddenSize, activation='relu')(inp)
    x = tf.keras.layers.Dropout(rate=0.1)(x)
    x = tf.keras.layers.Dense(hiddenSize, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.1)(x)
    x = tf.keras.layers.Dense(hiddenSize, activation='relu')(x)
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

scaler = MinMaxScaler()
results = []


path = "data/"

btchSzs = [64,64,64,64,32,64,64,64,64] 
files = ["mustVersion2.csv","segment.csv","landsatSatellite.csv","epilepticSeizure.csv","coil2000.csv", "ozoneOnehr.csv","aps_failure_all.csv"]
minClseslst = [[0],[1],[1],[1],[0],[0],[-1]]
minClsSizes = [[0.01, 0.025, 0.05],[0.01, 0.025, 0.05],[0.01, 0.025, 0.05],[0.01, 0.025, 0.05],[0.01, 0.025, 0.05],[0.025, 0.01],[0.01]]
# files = ["ionosphere-python","sonar-python"]
fileIdx = 0



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


for file in fileNames:
	print(file)
	f = open("tabularBinaryResults"+file+".txt", "a")
	techNames = []
	imbSizesTxt = []
	btchSz = btchSzs[fileIdx]
	data = pd.read_csv(path+file)  
	data = data.fillna(data.mean())
	X = data.to_numpy()
	y = X[:,X.shape[1]-1]
	y = y.astype(int)
	X = X[:,0:X.shape[1]-1].astype(float)
	clsLabs, clsSizes = np.unique(y, return_counts=True)
	outDim    = len(np.bincount(y))
	hiddenSize = int(2 * ((X.shape[1]+outDim)/3))
	maxClsId = np.argmax(clsSizes)
	minClsId = np.argmin(clsSizes)
	maxSize = np.max(clsSizes)
	minSize = np.min(clsSizes)
	for imbRatio in minClsSizes[fileIdx]:
		minIdx = np.where(y==minClsId)[0]
		sbsMinIdx = np.random.choice(minIdx, int(maxSize * imbRatio),replace=False)
		minDel = np.setdiff1d(minIdx, sbsMinIdx)
		imbY = np.delete(y, minDel)
		imbX = np.delete(X, minDel, axis=0)
		for techType, a in [['none',0],['WeightedBase',0],['SMOTE', 0],['mixup',0.1],['remix',0.1]]:	
			techNames = np.append(techNames, techType+str(a))
			imbSizesTxt = np.append(imbSizesTxt, str(imbRatio))
			rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=36851234)
			# rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=10, random_state=36851234)
			tmpGm = np.array([])
			tmpFm = np.array([])
			tmpBp = np.array([])
			tmpBn = np.array([])
			tmpBb = np.array([])
			for train_index, test_index in rskf.split(imbX, imbY):
				X_train, X_test = imbX[train_index, :], imbX[test_index, :]
				y_train, y_test = imbY[train_index], imbY[test_index]
				scaler = StandardScaler()
				X_train = scaler.fit_transform(X_train)
				X_test = scaler.transform(X_test)
				X_train = np.clip(X_train, -5, 5)
				X_test = np.clip(X_test, -5, 5)
				ss = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=568453)
				val_index, _ = ss.split(X_train, y_train)
				X_train2, X_val = X_train[val_index[0]], X_train[val_index[1]]
				y_train2, y_val = y_train[val_index[0]], y_train[val_index[1]]
				y_trainEncoded = tf.keras.utils.to_categorical(y_train2)
				y_valEncoded = tf.keras.utils.to_categorical(y_val)
				y_testEncoded = tf.keras.utils.to_categorical(y_test)
				y_trainEncoded = tf.keras.utils.to_categorical(y_train2)
				print(np.unique(y_train2, return_counts=True))
				model = get_model(X_train2.shape[1], outDim, hiddenSize)
				if techType == 'SMOTE':
					#Batch SMOTE
					model = get_model(X_train.shape[1], 2, hiddenSize)
					mu = ReMix.ReMix(alpha=None)
					train_data = DataGenerator(X_train2, y_trainEncoded, batch_size=btchSz, remixFunction=mu, balanceType="SMOTE")
					reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
					model.fit(train_data, epochs=500, shuffle=True,verbose=0,validation_data=(X_val, y_valEncoded), callbacks=[reduce_lr])
				elif techType == 'WeightedBase':
					class_weights = compute_class_weight('balanced', np.unique(y_train2), y_train2)
					model = get_model(X_train.shape[1], 2, hiddenSize)
					reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
					model.fit(X_train2, y_trainEncoded, batch_size=btchSz, epochs=500, class_weight=class_weights, shuffle=True,verbose=0,validation_data=(X_val, y_valEncoded),callbacks=[reduce_lr])
				elif techType == 'mixup':
					model = get_model(X_train.shape[1], 2, hiddenSize)
					mu = ReMix.ReMix(alpha=a)
					train_data = DataGenerator(X_train2, y_trainEncoded, batch_size=btchSz, remixFunction=mu, balanceType="mixup")
					reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
					model.fit(train_data, epochs=500, shuffle=True,verbose=0,validation_data=(X_val, y_valEncoded), callbacks=[reduce_lr])
				elif techType == 'remix':
					model = get_model(X_train.shape[1], 2, hiddenSize)
					mu = ReMix.ReMix(alpha=a)
					train_data = DataGenerator(X_train2, y_trainEncoded, batch_size=btchSz, remixFunction=mu, balanceType="remix")
					reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
					model.fit(train_data, epochs=500, shuffle=True,verbose=0,validation_data=(X_val, y_valEncoded), callbacks=[reduce_lr])
				else:
					reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
					model.fit(X_train2, y_trainEncoded, batch_size=btchSz, epochs=500, shuffle=True, validation_data=(X_val, y_valEncoded),verbose=0,callbacks=[reduce_lr])
				y_prob = model.predict(X_test)
				y_pred = np.argmax(y_prob,axis=1)
				tmpFm = np.append(tmpFm, f1_score(y_test, y_pred, average='macro'))
				tmpGm = np.append(tmpGm, geometric_mean_score(y_test, y_pred))
				tmpBp = np.append(tmpBb, brierPos(y_prob[np.where(y_test==1),1]))
				tmpBn = np.append(tmpBn, brierNeg(y_prob[np.where(y_test==0),0], negLab=1))
				tmpBb = np.append(tmpBb, balancedBrier(y_prob[np.where(y_test==minClsId),1], y_prob[np.where(y_test==maxClsId),1], posLab=1, negLab=1))
			fmResults = np.concatenate((fmResults, np.array([np.mean(tmpFm), np.std(tmpFm)]).reshape(1,2)),axis=0)
			gmResults = np.concatenate((gmResults, np.array([np.mean(tmpGm), np.std(tmpGm)]).reshape(1,2)),axis=0)
			brierPosResults = np.concatenate((brierPosResults, np.array([np.mean(tmpBp), np.std(tmpBp)]).reshape(1,2)),axis=0)
			brierNegResults = np.concatenate((brierNegResults, np.array([np.mean(tmpBn), np.std(tmpBn)]).reshape(1,2)),axis=0)
			brierBalResults = np.concatenate((brierBalResults, np.array([np.mean(tmpBb), np.std(tmpBb)]).reshape(1,2)),axis=0)
			datasetsNum = np.append(datasetsNum, fileIdx)
	fileIdx += 1
		
print("fm")
print(fmResults)
print("gm")
print(gmResults)
print("Brier Balanced")
print(brierBalResults)
print("Brier Positive")
print(brierPosResults)
print("Brier Negative")
print(np.round(brierNegResults, 3))
f.write(techNames)
f.write(imbSizesTxt)
f.write(str(aveRep))
f.write('\n')
details = np.concatenate((techNames, imbSizesTxt)).reshape(2,18)
allFmResults = np.concatenate((allFmResults, fmResults.reshape(1,18)))
allGmResults = np.concatenate((allGmResults, gmResults.reshape(1,18)))
print(allFmResults)
print(allGmResults)
f.close()

