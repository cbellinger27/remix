# <one line to give the program's name and a brief idea of what it does.>
# Copyright (C) <2020>  <Colin Bellinger>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
# Please direct any questions / comments to myself, Colin Bellinger, at
# colin.bellinger@gmail.com. For additional software and publications
# please see https://web.cs.dal.ca/~bellinger/ and researchgate
# https://www.researchgate.net/profile/Colin_Bellinger
#
# Relevant publications include: 
#
#AAAI 2021 paper: ReMix Training for Calibrated Imbalanced Deep Learning
#Colin Bellinger, colin.bellinger@nrc-cnrc.gc.ca - 1
#Roberto Corizzo, rcorizzo@american.edu - 2 
#Nathalie Japkowicz, japkowic@american.edu - 2 
#
#1 - National Research Council of Canada \\ Ottawa, Canada\\
#2 - American University \\ Department of Computer Science \\ Washington, DC 20016, USA\\
#
#
####################################################################################


import numpy as np
import random
import tensorflow as tf
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler

class ReMix:
	def __init__(self, alpha=0.2,minPercentage=None):
		self.alpha   = alpha
		self.minPercentage = minPercentage

	def sample(self, X, y, mixStyle):
		clsSizes   = np.sum(y, axis=0)
		minClses   = np.where(clsSizes < np.max(clsSizes))[0]
		numClses   = np.sum(clsSizes>0)
		synClsSize = int(X.shape[0] / numClses) # ASSUME WE WANT BALANCED CLASSES
		augmentedX = np.ndarray(shape=(0,X.shape[1]))
		augmentedY = np.ndarray(shape=(0,y.shape[1]))
		batchSize  = X.shape[0]
		if mixStyle == "balance":  # CALL BALANCE AND DOWN SAMPLE TO BATCH SIZE
			augmentedX, augmentedY = self.__balance__(X, y)
			augmentedX, augmentedY = self.__downsample__(augmentedX, augmentedY, batchSize)
		elif mixStyle == "mixup":  # basicMix # Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017). 
			augmentedX, augmentedY, lam = self.__mix__(X, y)
		elif mixStyle == "remix": # balanceMix # CALL BALANCE, MIX THE BALANCED DATA, AND THEN DOWN SAMPLE TO BATCH SIZE
			augmentedX, augmentedY = self.__balance__(X, y)
			augmentedX, augmentedY = self.__downsample__(augmentedX, augmentedY, batchSize)
			augmentedX, augmentedY, lam = self.__mix__(augmentedX, augmentedY)
		return augmentedX, augmentedY

	def __balance__(self, data, labels):
		balancedX = np.ndarray(shape=(0,data.shape[1]))
		balancedY = np.array([])
		rsmplFunction = RandomOverSampler()
		tmpY = np.argmax(labels,axis=1)
		clsLabs, clsSizes = np.unique(tmpY, return_counts=True)
		cBatchSz = np.max(clsSizes)      		  						#DETERMIN THE LARGEST CLASS
		for c in clsLabs:                                             	#SELECT THE REQUIRED NUMBER OF SAMPLES FOR EACH CLASS
			tmpIdx = np.random.choice(np.where(tmpY==c)[0], cBatchSz, replace=np.sum(tmpY==c)<cBatchSz)
			balancedX = np.concatenate((balancedX, data[tmpIdx,:]))
			balancedY = np.append(balancedY, tmpY[tmpIdx])
		balancedY = tf.keras.utils.to_categorical(balancedY).astype(int)
		return balancedX, balancedY

	def __downsample__(self, data, labels, batchSize):
		idx = np.random.choice(data.shape[0], batchSize)
		data = data[idx,:]
		labels = labels[idx, :]
		return data, labels

	def __mix__(self, data, labels):
		'''Returns mixed inputs, pairs of targets, and lambda'''
		if self.alpha > 0:
			lam = np.random.beta(self.alpha, self.alpha)
		else:
			lam = 1
		index = np.random.permutation(data.shape[0])
		mixed_x = lam * data + (1 - lam) * data[index, :]
		mixed_y = lam * labels + (1 - lam) * labels[index, :]
		return mixed_x, mixed_y, lam
