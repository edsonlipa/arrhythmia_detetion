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



#if len(sys.argv)!=(5+1):
#    print("Usage: python preprocess.py <NLRAV/NSVFQ> <denoise/no> <normalize/no> <augment/no> <random_seed>")
#    exit(-1)

#mode = sys.argv[1] # NLRAV / NSVFQ
mode ='NLRAV'
#r_seed = int(sys.argv[2])
print(sys.argv)


data_names = ['100', '101', '102', '103', '104', '105', '106', '107', 
              '108', '109', '111', '112', '113', '114', '115', '116', 
              '117', '118', '119', '121', '122', '123', '124', '200', 
              '201', '202', '203', '205', '207', '208', '209', '210', 
              '212', '213', '214', '215', '217', '219', '220', '221', 
              '222', '223', '228', '230', '231', '232', '233', '234']

wid = 100
n_samples = 650000 #<65000

def denoise(sig, sigma, wn='bior1.3'):
    threshold = sigma * np.sqrt(2*np.log2(len(sig)))
    c = pywt.wavedec(sig, wn)
    thresh = lambda x: pywt.threshold(x,threshold,'soft')
    nc = list(map(thresh, c))
    return pywt.waverec(nc, wn)

if mode=='NLRAV':
    labels = ['N', 'L', 'R', 'A', 'V']
    X = []
    Y = []
    for d in data_names:
        r=wfdb.rdrecord('./data/'+d,sampto=n_samples)
        ann=wfdb.rdann('./data/'+d, 'atr', return_label_elements=['label_store', 'symbol'],sampto=n_samples)
        if d!='114': # since 114 is reversed
            sig = np.array(r.p_signal[:,0])
        else:
            sig = np.array(r.p_signal[:,1])

        #quitar ruido
        #sig = denoise(sig, 0.005, 'bior1.3')
        
        #normalizar
        #sig = (sig-min(sig)) / (max(sig)-min(sig))
        
        sig_len = len(sig)
        print("sig_len ",sig_len)
        sym = ann.symbol
        #print("ann.symbol ",sym)
        #print(sig)
        pos = ann.sample
        #print("ann.sample ",pos)
        beat_len = len(sym)
        print("beat_len ",beat_len)
        
        for i in range(beat_len):
            if sym[i] in labels and pos[i]-wid>=0 and pos[i]+wid+1<=sig_len:
                a = sig[pos[i]-wid:pos[i]+wid+1]
                #print(a)
                #print('rangue ',i,' :',pos[i]-wid," - ",pos[i]+wid+1)
                if len(a) != 2*wid+1:
                    print("Length error")
                    continue
                X.append(a)
                Y.append(labels.index(sym[i]))
                #print('sym[i] :',sym[i])
                #print('labels.index(sym[i])   ',labels.index(sym[i]))
        #wfdb.plot_wfdb(record=r,annotation=ann,title=d+" normal",time_units='seconds')
        
X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)
print(Counter(Y))

data_len = len(X)
np.random.seed(1200)
idx = list(range(data_len))
np.random.shuffle(idx)

train_len = int(data_len*0.6) # 60%
valid_len = int(data_len*0.2) # 20%
test_len = data_len-train_len-valid_len # 20%

X_train = X[idx][:train_len]
X_valid = X[idx][train_len:train_len+valid_len]
X_test = X[idx][train_len+valid_len:]
Y_train = Y[idx][:train_len]
Y_valid = Y[idx][train_len:train_len+valid_len]
Y_test = Y[idx][train_len+valid_len:]



print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
print(Counter(Y_train))
print(Counter(Y_valid))
print(Counter(Y_test))


X_train = np.expand_dims(X_train, axis=-1)
X_valid = np.expand_dims(X_valid, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)


f_size = X_train.shape[1]
class_num = 5

#============================================#

lr = 0.01
batch_size=32

Y_train = keras.utils.to_categorical(Y_train, num_classes=class_num)

def make_model():
    model = Sequential()
    model.add(Conv1D(18, 7, activation='relu', input_shape=(f_size,1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(18, 7, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(class_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))
    return model
model = make_model()

best_SE = 0
best_ACC = 0
best_model = make_model()
patience = 30
pcnt = 0

bin_label = lambda x: min(1,x)
save = 'best_train'
for e in range(1, 100+1):

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=10, verbose=0)

    y_pred = model.predict(X_valid)
    y_pred = np.argmax(y_pred, axis=1)
    acc = np.sum(y_pred==Y_valid)/len(Y_valid)

    y_true = list(map(bin_label, Y_valid))
    y_pred = list(map(bin_label, y_pred))
    auc = roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    SE = tp/(tp+fn)
    SP = tn/(fp+tn)

    if SE+acc > best_SE+best_ACC:
        best_SE, best_ACC = SE, acc
        best_model.set_weights(model.get_weights())
        pcnt = 0
    else:
        pcnt += 1
    
    print("Epoch: %d | SE: %.4f | Best SE: %.4f | ACC: %.4f | Best ACC: %.4f | AUC: %.4f | SP: %.4f" %(e, SE, best_SE, acc, best_ACC, auc, SP))
    if pcnt==patience:
        y_pred = best_model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        acc = np.sum(y_pred==Y_test)/len(Y_test)
        y_true = list(map(bin_label, Y_test))
        y_pred = list(map(bin_label, y_pred))
        auc = roc_auc_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        SE = tp/(tp+fn)
        SP = tn/(fp+tn)
        print(mode+" Test | SE: %.4f | ACC: %.4f | AUC: %.4f | SP: %.4f | valid SE: %.4f | valid ACC: %.4f" %(SE, acc, auc, SP, best_SE, best_ACC))
        with open("./result/"+save, "a") as fw:
            fw.write("SE: %.4f | ACC: %.4f | AUC: %.4f | SP: %.4f | valid SE: %.4f | valid ACC: %.4f\n" %(SE, acc, auc, SP, best_SE, best_ACC))
        break

model.save('cnn_trained')
