# %%
# Combine individual participants' features into single large matrix, then export file

import numpy as np
import scipy as sp
import pandas as pd


from utilityFunctions import pairLoader

subNames=list(['fBlock00','fBlock01','fBlock02','fBlock03','fBlock04','fBlock05','fBlock06','fBlock07','fBlock08','fBlock09','fBlock10','fBlock11','fBlock12','fBlock13','fBlock14','fBlock15','fBlock16','fBlock17','fBlock18','fBlock19'])

[X0,y0]=pairLoader(subNames[0])
[xw,xh]=np.shape(X0)

#[X1,y1]=pairLoader(subName)

#X2=np.squeeze(np.concatenate((X0,X1),axis=0))

#y2=np.squeeze(np.concatenate((y0,y1),axis=0))

#print(np.shape(X2))
#print(np.shape(y2))

for subName in subNames:
	print(subName)
	[X1,y1]=pairLoader(subName)
	print(np.shape(X1))
	print(np.shape(y1))
	X1=X1[:,0:xh]
	X0=np.squeeze(np.concatenate((X0,X1),axis=0))
	y0=np.squeeze(np.concatenate((y0,y1),axis=0))
	print(np.shape(X0))
	print(np.shape(y0))
df = pd.DataFrame(X0)
outNamData='allSubs_Data.csv'
outNamLabels='allSubs_Labels.csv'

df.to_csv(outNamData,sep=',')

np.savetxt(outNamLabels, y0, delimiter=",", fmt='%s')
