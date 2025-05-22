# %%
# Load raw files and perform feature extraction

import numpy as np
import scipy as sp
import pandas as pd

from scipy import signal, arange, fft, fromstring, roll
from scipy.signal import butter, lfilter, ricker
import os
import glob
import re

from scipy.stats import stats
from utilityFunctions import featureExtraction, ghostFeatures, ghostHeap, ghostVector



print('Setting parameters...')
N = 4  # value for N-fold cross-validation
SAMPLE_RATE = int(250)  # Hz (subject to change)
fs=SAMPLE_RATE
NUM_WINDOWS = 1  # Dependent on number of samples of phonemes
lowcut=1
highcut=(np.floor(SAMPLE_RATE/2))
pcti = 99.95
phoLimit=int(12)
triLimit=int(10)
triMin=int(3)
# initialize script

featureList=list()
phonemeList=list()


subNames=list(['GTP000','GTP003','GTP004','GTP007','GTP044','GTP045','GTP047','GTP104','GTP223','GTP303','GTP308','GTP455','GTP545','GTP556','GTP762','GTP765'])
subNames=list(['Nos000','Nos001','Nos002','Nos003','Nos004','Nos005','Nos006','Nos007','Nos008','Nos009','Nos010','Nos101','Nos109','Nos111','Nos112','Nos125','Nos133','Nos222'])
subName='Nos000'
# nos101
subName=subNames[11]
#nos222
subName=subNames[17]

dirSub='C:/nostalgiaAlpha/eegData/'+subName +'/'
subName = subName.upper()
print(subName)

triNum=triLimit
for phoNum in range(0,phoLimit):
	nameFile=dirSub+subName + '_' + str(int(phoNum)) + '_' + str(int(triMin)) + '.txt'
	print(nameFile)
	try:
        	df = pd.read_csv(nameFile, sep='\t', header=None)
        	rawData=df.to_numpy()
        	lastRow=np.squeeze(rawData[:,31])
        	lastRow = np.where(lastRow != 0, 1, lastRow)
        	h0=np.unique(lastRow)
        	totalL=np.shape(h0)[0]
        	totalL=int(totalL)        	
        	indVal=np.where(lastRow==int(1))
        	indVal1=indVal[0]
        	indVal2=indVal1.tolist()

        	i1=indVal2[0]

        	for aa in range(0,len(indVal2)):
        		i1=indVal2[aa]
        		featureVector=ghostHeap(rawData, i1, fs, lowcut, highcut, pcti)
        		print(aa)
        		featureList.append(featureVector)
        		featureVector=list()
        		phonemeList.append(str(phoNum))
	except:
        	print('No epochs.')






labels_vector = np.array(phonemeList)
df = pd.DataFrame(featureList)
outNamData=subName+'_Data.csv'
outNamLabels=subName+'_Labels.csv'

df.to_csv(outNamData,sep=',')

np.savetxt(outNamLabels, labels_vector, delimiter=",", fmt='%s')
# %%
