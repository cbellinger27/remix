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

class Resampler:
	def __init__(self):
		print("resampler class")

	@staticmethod
	def smote(X, y): # Chawla, Nitesh V., et al. "SMOTE: synthetic minority over-sampling technique." Journal of artificial intelligence research 16 (2002): 321-357.
		augmentedX = np.ndarray(shape=(0,X.shape[1]))
		augmentedY = np.ndarray(shape=(0,y.shape[1]))
		allSizesY = np.zeros(y.shape[1])
		tmpY = np.argmax(y,axis=1)
		clsLabs, clsSizes = np.unique(tmpY, return_counts=True)
		allSizesY[clsLabs] = clsSizes
		if len(clsSizes) > 1:
			if np.min(clsSizes) < 3:
				rsmplFunction = RandomOverSampler()
			else: 
				rsmplFunction = SMOTE(k_neighbors=np.min([np.min(clsSizes)-1,5]))
			rsmpX, rsmpY = rsmplFunction.fit_resample(X.reshape((X.shape[0], -1)), tmpY)
			rsmpX = rsmpX.reshape(np.append(rsmpX.shape[0], X.shape[1:]))
			tmpIdx = np.random.choice(rsmpX.shape[0], X.shape[0], replace=False)
			augmentedX = rsmpX[tmpIdx,:]
			augmentedY = tf.keras.utils.to_categorical(rsmpY[tmpIdx],num_classes=y.shape[1]).astype(int)
		else:
			augmentedX = X
			augmentedY = y
		return augmentedX, augmentedY

	@staticmethod
	def randomOverSample(X, y):
		augmentedX = np.ndarray(shape=(0,X.shape[1]))
		augmentedY = np.ndarray(shape=(0,y.shape[1]))
		tmpY = np.argmax(y,axis=1)
		clsLabs, clsSizes = np.unique(tmpY, return_counts=True)
		if np.min(clsSizes) > 0:
			rsmplFunction = RandomOverSampler()
			rsmpX, rsmpY = rsmplFunction.fit_resample(X.reshape((X.shape[0], -1)), tmpY)
			rsmpX = rsmpX.reshape(np.append(rsmpX.shape[0], X.shape[1:]))
			tmpIdx = np.random.choice(rsmpX.shape[0], X.shape[0], replace=False)
			augmentedX = rsmpX[tmpIdx,:]
			augmentedY = tf.keras.utils.to_categorical(rsmpY[tmpIdx],num_classes=y.shape[1]).astype(int)
		else:
			augmentedX = X
			augmentedY = y
		return augmentedX, augmentedY
