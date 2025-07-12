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
SAMPLE_RATE = 250  # Hz (subject to change)
fs=SAMPLE_RATE
NUM_WINDOWS = 2  # Dependent on number of samples of phonemes
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
subNames=list(['Nos000','Nos001','Nos002','Nos003','Nos004','Nos005','Nos006','Nos007'])
subName='Nos000'

dirSub='./nostalgiaAlpha/eegData/'+subName +'/'

phoNum=int(2)
subName = subName.upper() 
nameFile=dirSub+subName + '_' + str(int(phoNum)) + '_' + str(int(triMin)) + '.txt'
print(nameFile)

nameFile='C:/nostalgiaAlpha/eegData/Nos000/NOS000_2_3.txt'
print(nameFile)
#rawData = np.genfromtxt(nameFile, delimiter='/t')



df = pd.read_csv(nameFile, sep='\t', header=None)
rawData=df.to_numpy()
lastRow=np.squeeze(rawData[:,31])
lastRow = np.where(lastRow != 0, 1, lastRow)

h0=np.unique(lastRow)
print(np.unique(lastRow))
totalL=np.shape(h0)[0]
totalL=int(totalL)

h0=np.unique(lastRow)

totalL=np.shape(h0)[0]
totalL=int(totalL)    	
indVal=np.where(lastRow==int(1))
indVal1=indVal[0]
indVal2=indVal1.tolist()
print((indVal2))
i1=indVal2[0]
print(i1)
featureVector=ghostHeap(rawData, i1, fs, lowcut, highcut, pcti)


for aa in range(0,len(indVal2)):

	i1=indVal2[aa]
	featureVector=ghostHeap(rawData, i1, fs, lowcut, highcut, pcti)
	print(aa)
	featureList.append(featureVector)
	featureVector=list()
	phonemeList.append(str(phoNum))

print(np.shape(featureList))
print(np.shape(phonemeList))