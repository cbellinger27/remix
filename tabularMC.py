from sklearn.model_selection import RepeatedStratifiedKFold
import tensorflow as tf
import pandas as pd
import numpy as np 
from functools import reduce
import imbalancedMixUp as mixUp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
import imbalancedMixUp as mixUp
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from imblearn.metrics import geometric_mean_score

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, labels, batch_size=64, shuffle=True, mixUp=None, mixUpType=None):
        self.y = labels      # array of labels
        self.X = data        # array of data
        self.batch_size   = batch_size          # batch size
        self.shuffle      = shuffle             # shuffle bool
        self.mixUp = mixUp
        self.mixUpType = mixUpType
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
        # select data and load images
        batchY = np.array([self.y[k,:] for k in indexes])
        batchX =  np.array([self.X[k,:] for k in indexes])
        if 'balance' in self.mixUpType:         # CONSIDER SWICH FROM UP-SAMPLING THE MINORITY CLASSES TO PRIORITY SAMPLING THE EACH BATCH
            augmentedX = np.ndarray(shape=(0,self.X.shape[1]))
            augmentedY = np.array([])
            rsmplFunction = RandomOverSampler()
            tmpY = np.argmax(batchY,axis=1)
            clsLabs, clsSizes = np.unique(tmpY, return_counts=True)
            # if np.argmin(clsSizes) > 0:
            batchX, batchY = rsmplFunction.fit_resample(batchX, tmpY)
            cBatchSz = int(np.round(self.batch_size/len(clsLabs)))
            for c in clsLabs:
                tmpIdx = np.random.choice(np.where(batchY==c)[0], cBatchSz, replace=np.sum(batchY==c)<cBatchSz)
                augmentedX = np.concatenate((augmentedX, batchX[tmpIdx,:]))
                augmentedY = np.append(augmentedY, batchY[tmpIdx])
            if len(augmentedY) < self.batch_size:
                idx = np.random.choice(len(batchY), self.batch_size-len(augmentedY))
                augmentedX = np.concatenate((augmentedX, batchX[idx,:]))
                augmentedY = np.append(augmentedY, batchY[idx])
            # print(np.unique(augmentedY, return_counts=True))
            batchX = augmentedX
            batchY = tf.keras.utils.to_categorical(augmentedY).astype(int)
            idx = np.random.choice(batchX.shape[0], np.min([batchX.shape[0],self.batch_size]), replace=False)   # SELECT SUBSET EQUAL IN SIZE TO ORGINAL BATCH SIZE
            batchX = batchX[idx,:]
            batchY = batchY[idx,:]
        x_out, y_out = self.mixUp.sample(batchX, batchY, self.mixUpType)
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
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=METRICS)
    return model


def brierPos(pred, posLab=1):
	return (np.sum((posLab - pred)**2)/pred.shape[1])


def brierNeg(pred, negLab=0):
	return (np.sum((negLab - pred)**2)/pred.shape[1])

def balancedBrier(predPos, predNeg, posLab=1, negLab=0):
	return (brierNeg(predNeg, negLab) + brierPos(predPos, posLab))/2


def mcBrier(preds, labs):
	mcbs = 0
	n = preds.shape[0]
	k = labs.shape[1]
	for p in range(n):
		for c in range(k):
			mcbs += (labs[p,c] - preds[p,c])**2
	return mcbs/(2*n)


def multiClassBalancedBrier2(preds, labs):
	mcbbs = 0
	k = labs.shape[1]
	clsCnt=0
	for c in range(k):
		idx = np.where(labs[:,c]==1)[0]
		if len(idx) > 0:
			clsCnt += 1
			mcbbs += mcBrier(preds[idx,:], labs[idx, :])
	return mcbbs/k

def multiClassBalancedBrier(preds, labs):
	bbs = 0
	clsCnt = 0
	for c in range(labs.shape[1]):
		clsInstCnt = np.sum(labs[:,c])
		if clsInstCnt > 0:
			clsCnt += 1
			bbs += (np.sum((1 - preds[np.where(labs[:,c]==1),c])**2)/clsInstCnt)
	return bbs / clsCnt


scaler = MinMaxScaler()
results = []

# file = "segment"
# file = "abalone"
# file = "sonar-python"
# file = "ionosphere-python"
# file = "cifar_5-python"

# "sonar-pythonBine1vAll.csv",
fileNames = ["optdigits.csv","landsatSatellite.csv","epilepticSeizure.csv","letter.csv"]
minClseslst = [[1,2,3],[2,3,4],[1,2],[1,2,3,5,7]]


fileNames = ["optdigits.csv"]
minClseslst = [[1,2,3]]

path = "data/"

btchSz = 128


allResults = np.ndarray(shape=(0,12))
brierPosResultsAll = np.ndarray(shape=(0,11))
brierNegResultsAll = np.ndarray(shape=(0,11))
brierBalResultsAll = np.ndarray(shape=(0,11))
fmResultsAll       = np.ndarray(shape=(0,11))
gmResultsAll       = np.ndarray(shape=(0,11))
fmResults = np.ndarray(shape=(0,2))
gmResults = np.ndarray(shape=(0,2))
baResults = np.ndarray(shape=(0,2))
brierPosResults = np.ndarray(shape=(0,2))
brierMcResults = np.ndarray(shape=(0,2))
brierBalResults = np.ndarray(shape=(0,2))
datasetsNum = np.array([])
minSizes = np.array([])
dsNum = 0

fileIdx = 0
for file in fileNames:
	print(file)
	minClses = minClseslst[fileIdx]
	data = pd.read_csv(path+file)  
	data = data.fillna(data.mean())
	X = data.to_numpy()
	y = X[:,X.shape[1]-1]
	y = y.astype(int)
	X = X[:,0:X.shape[1]-1].astype(float)
	clsLabs, clsSizes = np.unique(y, return_counts=True)
	outDim    = len(np.bincount(y))
	hiddenSize = int(2 * ((X.shape[1]+outDim)/0.75))
	maxClsIdx = np.argmax(clsSizes)
	minClsIdx = np.argmin(clsSizes)
	majSize = np.max(clsSizes)
	minSize = np.min(clsSizes)
	maxSize = np.max(clsSizes)
	for imbRatio in [0.05, 0.025, 0.01]:
	# for imbRatio in [0.1, 0.05, 0.25]:
	# for imbRatio in [1]:
		# f.write('\n\n')
		print("imbalance ration " + str(imbRatio))
		imbX = X.copy()
		imbY = y.copy()
		for c in minClses:
			print("min class adjusted from size: " + str(np.sum(imbY==c)) + " with maj class size: " + str(majSize) + " and IR: " + str(imbRatio))
			minIdx = np.where(imbY==c)[0]
			if len(minIdx) < int(majSize * imbRatio):
				sbsMinIdx = np.random.choice(minIdx, int(majSize * imbRatio), replace=True)
			else:
				sbsMinIdx = np.random.choice(minIdx, int(majSize * imbRatio), replace=False)
			minDel = np.setdiff1d(minIdx, sbsMinIdx)
			if len(minDel) > 0:
				imbY = np.delete(imbY, minDel)
				imbX = np.delete(imbX, minDel, axis=0)
				print("to size: " + str(np.sum(imbY==c)))
		mixNames = []
		for mixUpType, a, b in [['none',0,0],['WeightedBase',0,0],['SMOTE', 0, 0],['basicMix',0.1,0.1],['balanceMix',0.1,0.1],['balanceMix',0.75,5]]:
		# for mixUpType, a, b in [['basicMix',0.05,0.05],['balanceMix',0.05,0.05]]:	
		# for mixUpType, a, b in [['SMOTE', 0, 0]]:	
		# mixNames = np.append(mixNames, mixUpType+str(a)+"_"+str(b))
			rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=10, random_state=36851234)
			tmpGm = np.array([])
			tmpFm = np.array([])
			tmpBa  = np.array([])
			tmpBp = np.array([])
			tmpBmc = np.array([])
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
				# print(np.unique(y_train2, return_counts=True))
				model = get_model(X_train2.shape[1], outDim, hiddenSize)
				if mixUpType == 'SMOTE':
					#Batch SMOTE
					model = get_model(X_train.shape[1], y_trainEncoded.shape[1], hiddenSize)
					mu = mixUp.ImbalanceMixUp(alpha=0.1, beta=0.1)
					train_data = DataGenerator(X_train2, y_trainEncoded, batch_size=btchSz, mixUp=mu, mixUpType="SMOTE")
					reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
					model.fit(train_data, epochs=500, shuffle=True,verbose=0,validation_data=(X_val, y_valEncoded), callbacks=[reduce_lr])
				elif mixUpType == 'WeightedBase':
					class_weights = compute_class_weight('balanced', np.unique(y_train2), y_train2)
					model = get_model(X_train.shape[1], y_trainEncoded.shape[1], hiddenSize)
					reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
					model.fit(X_train2, y_trainEncoded, batch_size=btchSz, epochs=500, class_weight=class_weights, shuffle=True,verbose=0,validation_data=(X_val, y_valEncoded),callbacks=[reduce_lr])
				elif mixUpType == 'basicMix':
					model = get_model(X_train.shape[1], y_trainEncoded.shape[1], hiddenSize)
					mu = mixUp.ImbalanceMixUp(alpha=a, beta=b)
					train_data = DataGenerator(X_train2, y_trainEncoded, batch_size=btchSz, mixUp=mu, mixUpType="basicMix")
					reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
					model.fit(train_data, epochs=500, shuffle=True,verbose=0,validation_data=(X_val, y_valEncoded), callbacks=[reduce_lr])
				elif mixUpType == 'balanceMix':
					model = get_model(X_train.shape[1], y_trainEncoded.shape[1], hiddenSize)
					mu = mixUp.ImbalanceMixUp(alpha=a, beta=b)
					train_data = DataGenerator(X_train2, y_trainEncoded, batch_size=btchSz, mixUp=mu, mixUpType="balanceMix")
					reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
					model.fit(train_data, epochs=500, shuffle=True,verbose=0,validation_data=(X_val, y_valEncoded), callbacks=[reduce_lr])
				else:
					reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
					model.fit(X_train2, y_trainEncoded, batch_size=btchSz, epochs=500, shuffle=True, validation_data=(X_val, y_valEncoded),verbose=0,callbacks=[reduce_lr])
				y_prob = model.predict(X_test)
				y_pred = np.argmax(y_prob,axis=1)
				tmpFm = np.append(tmpFm, f1_score(y_test, y_pred, average='macro'))
				tmpGm = np.append(tmpGm, geometric_mean_score(y_test, y_pred))
				tmpBa = np.append(tmpFm, balanced_accuracy_score(y_test, y_pred))
				tmpBmc = np.append(tmpBmc, mcBrier(y_prob, y_testEncoded))
				tmpBb = np.append(tmpBb, multiClassBalancedBrier2(y_prob, y_testEncoded))
				tmpBp = np.append(tmpBp, multiClassBalancedBrier2(y_prob[:,minClseslst[0]], y_testEncoded[:,minClseslst[0]]))
			fmResults = np.concatenate((fmResults, np.array([np.mean(tmpFm), np.std(tmpFm)]).reshape(1,2)),axis=0)
			gmResults = np.concatenate((gmResults, np.array([np.mean(tmpGm), np.std(tmpGm)]).reshape(1,2)),axis=0)
			baResults = np.concatenate((baResults, np.array([np.mean(tmpBa), np.std(tmpBa)]).reshape(1,2)),axis=0)
			brierBalResults = np.concatenate((brierBalResults, np.array([np.mean(tmpBb), np.std(tmpBb)]).reshape(1,2)),axis=0)
			brierPosResults = np.concatenate((brierPosResults, np.array([np.mean(tmpBp), np.std(tmpBp)]).reshape(1,2)),axis=0)
			brierMcResults = np.concatenate((brierMcResults, np.array([np.mean(tmpBmc), np.std(tmpBmc)]).reshape(1,2)),axis=0)
			datasetsNum = np.append(datasetsNum, dsNum)
	print("fm")
	print(fmResults)
	print("gm")
	print(gmResults)
	print("Ba")
	print(baResults)
	print("Brier Balanced")
	print(brierBalResults)
	print("Brier Positive")
	print(brierPosResults)
	print("Brier Negative")
	print(np.round(brierNegResults, 3))
	fileIdx += 1
	# 	f.write(mixNames)
	# 	f.write(str(aveRep))
	# 	f.write('\n')
	# 	allFmResults = np.concatenate((allFmResults, fmResults.reshape(1,11)))
	# 	allGmResults = np.concatenate((allGmResults, gmResults.reshape(1,11)))
	# print(allFmResults)
	# print(allGmResults)
	# f.close()
	# btchSzIdx += 1
