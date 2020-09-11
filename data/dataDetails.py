import pandas as pd
import numpy as np

fileNames = ["talendHC6.csv", "ozoneOnehr.csv","optdigits.csv","mustVersion2.csv","landsatSatellite.csv","epilepticSeizure.csv","aps_failure_all.csv","coil2000.csv","letter.csv"]
path = "mixUpData/"


dataDetails = [["File Name", "Number of Samples", "Number of Features", "Class Labels", "Instances per Class"]]

for fileName in fileNames:
	print(fileName)
	data = pd.read_csv(path+fileName)  
	data = data.fillna(data.mean())
	X = data.to_numpy()
	y = X[:,X.shape[1]-1]
	X = X[:,0:X.shape[1]-1]
	clsLabs = np.unique(y)
	tmp = [fileName, str(X.shape[0]), str(X.shape[1]), str(clsLabs)]
	clsSizes = ""
	for c in clsLabs:
		clsSizes = clsSizes + " " + str(np.sum(y==c))
	tmp = tmp + [clsSizes]
	print(tmp)
	dataDetails.append(tmp)

