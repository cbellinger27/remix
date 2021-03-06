import tensorflow as tf
import pandas as pd
import numpy as np 
import remix2 as ReMix
import Resampler as Resampler
from functools import reduce
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score, balanced_accuracy_score,classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit,StratifiedKFold


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
      tmpY = np.argmax(batchY,axis=1)
      #print(np.unique(tmpY, return_counts=True))
      return Resampler.Resampler.smote(batchX, batchY)
    else:     # MIXUP OR REMIX
      tmpY = np.argmax(batchY,axis=1)
      #print(np.unique(tmpY, return_counts=True))
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


fileNames = ["optdigits.csv","landsatSatellite.csv","epilepticSeizure.csv","letter.csv"]
minClseslst = [[1,2,3],[2,3,4],[1,2],[1,2,3,5,7]]


path = "data/"

btchSz = 64


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

file = fileNames[0]

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

imbRatio = 0.05

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


rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=10, random_state=36851234)

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



#test none ############################################################
model = get_model(X_train.shape[1], y_trainEncoded.shape[1], hiddenSize)
techType = 'none'
a = 0.2

train_data = DataGenerator(X_train2, y_trainEncoded, batch_size=btchSz, remixFunction=mu, balanceType=techType)
bX, bY = train_data.__getitem__(0)

tmpY = np.argmax(bY,axis=1)
print(np.unique(tmpY, return_counts=True))


#test smote ############################################################
model = get_model(X_train.shape[1], y_trainEncoded.shape[1], hiddenSize)
techType = 'SMOTE'
a = 0.2

train_data = DataGenerator(X_train2, y_trainEncoded, batch_size=btchSz, remixFunction=mu, balanceType=techType)
bX, bY = train_data.__getitem__(0)

tmpY = np.argmax(bY,axis=1)
print(np.unique(tmpY, return_counts=True))

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
model.fit(train_data, epochs=500, shuffle=True,verbose=0,validation_data=(X_val, y_valEncoded), callbacks=[reduce_lr])

y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob,axis=1)
f1_score(y_test, y_pred, average='macro')
geometric_mean_score(y_test, y_pred)
balanced_accuracy_score(y_test, y_pred)
multiClassBalancedBrier2(y_prob, y_testEncoded)

#test mixup ############################################################
model = get_model(X_train.shape[1], y_trainEncoded.shape[1], hiddenSize)
techType = 'mixup'
a = 0.2
mu = ReMix.ReMix(alpha=a)

train_data = DataGenerator(X_train2, y_trainEncoded, batch_size=btchSz, remixFunction=mu, balanceType=techType)
bX, bY = train_data.__getitem__(0)

tmpY = np.argmax(bY,axis=1)
print(np.unique(tmpY, return_counts=True))

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
model.fit(train_data, epochs=500, shuffle=True,verbose=0,validation_data=(X_val, y_valEncoded), callbacks=[reduce_lr])

y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob,axis=1)
f1_score(y_test, y_pred, average='macro')
geometric_mean_score(y_test, y_pred)
balanced_accuracy_score(y_test, y_pred)
multiClassBalancedBrier2(y_prob, y_testEncoded)

#test remix ############################################################
model = get_model(X_train.shape[1], y_trainEncoded.shape[1], hiddenSize)
techType = 'remix'
a = 0.2
mu = ReMix.ReMix(alpha=a)

train_data = DataGenerator(X_train2, y_trainEncoded, batch_size=btchSz, remixFunction=mu, balanceType=techType)
bX, bY = train_data.__getitem__(0)

tmpY = np.argmax(bY,axis=1)
print(np.unique(tmpY, return_counts=True))

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
model.fit(train_data, epochs=500, shuffle=True,verbose=0,validation_data=(X_val, y_valEncoded), callbacks=[reduce_lr])

y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob,axis=1)
f1_score(y_test, y_pred, average='macro')
geometric_mean_score(y_test, y_pred)
balanced_accuracy_score(y_test, y_pred)
multiClassBalancedBrier2(y_prob, y_testEncoded)

