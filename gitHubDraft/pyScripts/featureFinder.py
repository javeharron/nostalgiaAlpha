#running intra-subject tests

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal, arange, fft, fromstring, roll
from scipy.signal import butter, lfilter, ricker
import os
import glob
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn import svm

from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, cross_validate, train_test_split
from sklearn import metrics, linear_model, preprocessing
from sklearn.cluster import DBSCAN
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, make_scorer, classification_report
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from scipy.stats import stats
from utilityFunctions import pairLoader, eegFeatureReducer, balancedMatrix, featureSelect, speedClass, dirClass, dualClass, fsClass, classOutputs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn import svm


#find top features

featureNumber=int(3)

stringName='allSub_Data.csv'
df = pd.read_csv(stringName)
s_array = df.to_numpy()
X=s_array
y= np.genfromtxt('allSub_Labels.csv', delimiter=',')
fList=list()

for aa in range(0,11):
	aFeatures,totalLength,X1,X2=featureSelect(X, y, featureNumber, aa)
	print(aFeatures)
	aaFeatures=str(aFeatures)
	fList.append(aaFeatures)


print(fList)
ab=[s.replace('[', '') for s in fList]
ac=[s.replace(']', '') for s in ab]
fAr=np.asarray(ac)
np.savetxt("listArray.csv", fAr, delimiter=", ", fmt="% s")
#np.histogram(fAr)
print(ac)


#63 64 65
#30 63 64 65
yar= np.genfromtxt('listArray.csv', delimiter='\t')
#print(yar)
#print(np.unique(yar))
# top feature: 168
#yar=np.array([339 340 341 342 160 161 181 182 183 329 336 337 338 718 719 720 184 185 186 203 204 205 190 191 192 193 194 168 169 170 310 164 165 166 217 218 219 168 169 170 404 405 406 #189 190 191 203 204 205 155 156 157 409 410 411 201 202 203 343 344 345 165 166 167 303 304 305 188 189 190 195 196 197 161 162 163 342 343 344 100 181 182 239 240 870 165 166 167 293 ##294 295 325 326 327 328 329 330 168 169 170 287 288 289 168 169 170 133 134 135 168 169 658 659 660 153 154 155 378 379 380 63 64 65 203 204 205 413 414 415 145 146 147 168 169 170 168 #169 170 332 333 334 190 191 192 298 299 300 186 202 203 204 344 345 191 192 193 203 204 205 238 239 240 288 289 290 236 237 238 239 240 168 169 170 205 153 154 155 293 294 295 162 163 ##164 165 166 131 132 133 134 164 165 166 168 169 170 168 169 170 709 778 779 168 169 170 232 233 234 378 379 380 605 606 607 98  99 100 181 182 183  9  10  11 308 309 310 98  99 100])
#print(np.unique(yar))
#np.histogram(yar)



