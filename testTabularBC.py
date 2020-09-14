import tensorflow as tf
import pandas as pd
import numpy as np 
import remix as ReMix
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
    tmpY = np.argmax(batchY,axis=1)
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
fileNames = ["mustVersion2.csv","segment.csv","landsatSatellite.csv","epilepticSeizure.csv","coil2000.csv", "ozoneOnehr.csv","aps_failure_all.csv"]
minClseslst = [[0],[1],[1],[1],[0],[0],[-1]]
minClsSizes = [[0.01, 0.025, 0.05],[0.01, 0.025, 0.05],[0.01, 0.025, 0.05],[0.01, 0.025, 0.05],[0.01, 0.025, 0.05],[0.025, 0.01],[0.01]]
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


file = fileNames[0]
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

imbRatio = minClsSizes[fileIdx][0]
minIdx = np.where(y==minClsId)[0]
sbsMinIdx = np.random.choice(minIdx, int(maxSize * imbRatio),replace=False)
minDel = np.setdiff1d(minIdx, sbsMinIdx)
imbY = np.delete(y, minDel)
imbX = np.delete(X, minDel, axis=0)

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=36851234)

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
balancedBrier(y_prob[np.where(y_test==minClsId),1], y_prob[np.where(y_test==maxClsId),1], posLab=1, negLab=1)

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
balancedBrier(y_prob[np.where(y_test==minClsId),1], y_prob[np.where(y_test==maxClsId),1], posLab=1, negLab=1)

#test remix ############################################################
model = get_model(X_train.shape[1], y_trainEncoded.shape[1], hiddenSize)
techType = 'remix'
a = 0.4
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
balancedBrier(y_prob[np.where(y_test==minClsId),1], y_prob[np.where(y_test==maxClsId),1], posLab=1, negLab=1)



