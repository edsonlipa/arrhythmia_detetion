import numpy as np
import sys, os
import wfdb
import pywt
import matplotlib.pyplot as plt
import pickle as pk
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score 
import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.models import load_model

print(sys.argv)

mode='NLRAV'
data_names = ['118e00']

wid = 100
n_samples = 650000 #<65000
labels = ['N', 'L', 'R', 'A', 'V']

if mode=='NLRAV':
    X = []
    Y = []
    for d in data_names:
        r=wfdb.rdrecord('./data/mit-bih-noise/'+d,sampto=n_samples)
        ann=wfdb.rdann('./data/mit-bih-noise/'+d, 'atr', return_label_elements=['label_store', 'symbol'],sampto=n_samples)
        if d!='114': # since 114 is reversed
            sig = np.array(r.p_signal[:,0])
        else:
            sig = np.array(r.p_signal[:,1])

        sig_len = len(sig)
        sym = ann.symbol
        pos = ann.sample
        beat_len = len(sym)
        
        for i in range(beat_len):
            if sym[i] in labels and pos[i]-wid>=0 and pos[i]+wid+1<=sig_len:
                a = sig[pos[i]-wid:pos[i]+wid+1]
                if len(a) != 2*wid+1:
                    print("Length error")
                    continue
                X.append(a)
                Y.append(labels.index(sym[i]))
        
X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)
print(Counter(Y))

data_len = len(X)
np.random.seed(1200)
model_trained=load_model('cnn_trained')

while True:
        
    select_to_test = np.random.random_integers(data_len)
    test=np.expand_dims(X[select_to_test],axis=-1)
    test=np.expand_dims(test,axis=0)
    #print(X[select_to_test])
    print(test)
    """
    select_to_test = np.expand_dims(select_to_test, axis=-1)
    print(select_to_test)
    select_to_test = np.expand_dims(select_to_test, axis=-1)
    print(select_to_test)
    """
    y_pred = model_trained.predict(test)
    y_pred = np.argmax(y_pred, axis=1)
    
    y_pred = y_pred[0]
    
    print(y_pred)
    print('prediccion ',labels[y_pred])
    print('real label',labels[Y[select_to_test]])
    
    #y_pred = np.argmax(y_pred, axis=1)
    """acc = np.sum(y_pred==Y_test)/len(Y_test)
    y_true = list(map(bin_label, Y_test))
    y_pred = list(map(bin_label, y_pred))
    auc = roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    SE = tp/(tp+fn)
    SP = tn/(fp+tn)
    print(mode+" Test | SE: %.4f | ACC: %.4f | AUC: %.4f | SP: %.4f | valid SE: %.4f | valid ACC: %.4f" %(SE, acc, auc, SP, best_SE, best_ACC))
       """ 

    plt.plot(X[select_to_test])
    print(Y[select_to_test])
    titulo = str(select_to_test)+' : '+labels[Y[select_to_test]]
    plt.title(titulo)
    plt.show()


#X_train = np.expand_dims(X_train, axis=-1)

