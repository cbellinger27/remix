import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import remix as ReMix
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from imblearn.over_sampling import SMOTE, RandomOverSampler
from functools import reduce
from sklearn.calibration import calibration_curve

h = .02
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])


#CREATE CIRCLE DATASET
ds = make_circles(n_samples=1000,noise=0.1, factor=0.1, random_state=1)

X, y = ds
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

#CONVERT THE DATA SET TO AN IMBALNACED BINARY DATASET
imbX = X_train.copy()
imbY = y_train.copy()
minIdx = np.where(y_train==1)[0]
sbsMinIdx = np.random.choice(minIdx, len(minIdx)-25, replace=False)
imbY = np.delete(imbY, sbsMinIdx)
imbX = np.delete(imbX, sbsMinIdx, axis=0)
imby_trainEncoded = tf.keras.utils.to_categorical(imbY)



#PLOT THE BALANCED CIRCLE DATASET
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='k')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title("Balanced Dataset")
plt.show()

#PLOT THE IMBALANCED CIRCLE DATASET
plt.scatter(imbX[:, 0], imbX[:, 1], c=imbY, cmap=cm_bright,edgecolors='k')
plt.scatter(imbX[:, 0], imbX[:, 1], c=imbY, cmap=cm_bright, alpha=0.6,edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title("Imbalanced Dataset")
plt.show()



#DEMONSTRATION CLASSIFICATION MODEL
def get_model(inputDim, outputDim, hiddenSize):
    inp = tf.keras.Input((inputDim,))
    x = tf.keras.layers.Dense(hiddenSize, activation='relu')(inp)
    x = tf.keras.layers.Dropout(rate=0.1)(x)
    x = tf.keras.layers.Dense(hiddenSize, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.1)(x)
    x = tf.keras.layers.Dense(hiddenSize, activation='relu')(x)
    out = tf.keras.layers.Dense(outputDim, activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam())
    return model

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


#TRAIN BASELINE MODEL
model = get_model(X_train.shape[1], 2, 10)
model.fit(imbX, imby_trainEncoded, batch_size=32, epochs=150, shuffle=True,verbose=0)
score = model.predict(X_test)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
zImbPlain = Z

plt.contourf(xx, yy, Z, levels=20, cmap="Blues", alpha=.8)
plt.colorbar()
plt.scatter(imbX[:, 0], imbX[:, 1], c=imbY, cmap=cm_bright,edgecolors='k')
plt.clim(0, 1);
plt.title("Baseline Model")
plt.show()


#PLOT CALIBRATION CURVES OF MODEL TRAINED WITH REMIX
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, score[:,1], n_bins=10)
fig = plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))
ax1.plot(mean_predicted_value, fraction_of_positives, "s-",label="Baseline Model")
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
ax1.legend()
ax2.hist(score[:,1], range=(0, 1), bins=10, histtype="step", lw=2)
plt.show()

#####################################


#TRAIN MODEL WITH MIXUP
model = get_model(X_train.shape[1], 2, 10)
mu = ReMix.ReMix(alpha=0.3)
train_data = DataGenerator(imbX, imby_trainEncoded, batch_size=32, remixFunction=mu, balanceType="mixup")
model.fit(train_data, epochs=150, shuffle=True,verbose=0)
score = model.predict(X_test)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

#PLOT RESULTS OF MODEL TRAINED WITH MIXUP
plt.contourf(xx, yy, Z, levels=20, cmap="Blues", alpha=.8)
plt.colorbar()
plt.clim(0, 1);
plt.scatter(imbX[:, 0], imbX[:, 1], c=imbY, cmap=cm_bright,edgecolors='k')
plt.title("Trained with MixUp")
plt.show()

#PLOT CALIBRATION CURVES OF MODEL TRAINED WITH REMIX
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, score[:,1], n_bins=10)
fig = plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))
ax1.plot(mean_predicted_value, fraction_of_positives, "s-",label="Trained with MixUp")
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
ax1.legend()
ax2.hist(score[:,1], range=(0, 1), bins=10, histtype="step", lw=2)
plt.show()

#####################################


#TRAIN MODEL WITH REMIX
model = get_model(X_train.shape[1], 2, 10)
mu = ReMix.ReMix(alpha=0.5)
train_data = DataGenerator(imbX, imby_trainEncoded, batch_size=32, remixFunction=mu, balanceType="remix")
model.fit(train_data, epochs=150, shuffle=True,verbose=0)
score = model.predict(X_test)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

#PLOT RESULTS OF MODEL TRAINED WITH REMIX
plt.contourf(xx, yy, Z, levels=20, cmap="Blues", alpha=.8)
plt.clim(0, 1);
plt.colorbar()
plt.scatter(imbX[:, 0], imbX[:, 1], c=imbY, cmap=cm_bright,edgecolors='k')
plt.title("Trained with ReMix")
plt.show()


#PLOT CALIBRATION CURVES OF MODEL TRAINED WITH REMIX
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, score[:,1], n_bins=10)
fig = plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))
ax1.plot(mean_predicted_value, fraction_of_positives, "s-",label="Trained with ReMix")
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
ax1.legend()

ax2.hist(score[:,1], range=(0, 1), bins=10, histtype="step", lw=2)
plt.show()

#####################################


#TRAIN MODEL WITH SMOTE IN MINI-BATCH
model = get_model(X_train.shape[1], 2, 10)
mu = ReMix.ReMix(alpha=0.1)
train_data = DataGenerator(imbX, imby_trainEncoded, batch_size=32, remixFunction=mu, balanceType="SMOTE")
model.fit(train_data, epochs=150, shuffle=True,verbose=0)
score = model.predict(X_test)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)


#PLOT RESULTS OF MODEL WITH SMOTE IN MINI-BATCH
plt.contourf(xx, yy, Z, levels=20, cmap="Blues", alpha=.8)
plt.clim(0, 1);
plt.colorbar()
plt.scatter(imbX[:, 0], imbX[:, 1], c=imbY, cmap=cm_bright,edgecolors='k')
plt.title("Trained with SMOTE in Mini-Batch")
plt.show()

#PLOT CALIBRATION CURVES OF MODEL WITH SMOTE IN MINI-BATCH
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, score[:,1], n_bins=10)
fig = plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))
ax1.plot(mean_predicted_value, fraction_of_positives, "s-",label="Trained with SMOTE in Mini-Batch")
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
ax1.legend()
ax2.hist(score[:,1], range=(0, 1), bins=10, histtype="step", lw=2)
plt.show()


