import numpy as np
import sys, os
import wfdb
import pywt
import matplotlib.pyplot as plt
#import pickle as pk
#from collections import Counter

#if len(sys.argv)!=(5+1):
#    print("Usage: python preprocess.py <NLRAV/NSVFQ>  <random_seed>")
#    exit(-1)

mode = 'NLRAV'#sys.argv[1] # NLRAV / NSVFQ
#r_seed = int(sys.argv[1])
r_seed = 1200
print(sys.argv)

'''data_names = ['100', '101', '102', '103', '104', '105', '106', '107', 
              '108', '109', '111', '112', '113', '114', '115', '116', 
              '117', '118', '119', '121', '122', '123', '124', '200', 
              '201', '202', '203', '205', '207', '208', '209', '210', 
              '212', '213', '214', '215', '217', '219', '220', '221', 
              '222', '223', '228', '230', '231', '232', '233', '234']'''
data_names = ['100']
widb = 99
wida = 160
#wid = 100
n_samples = 650000 #<650000
foto = np.zeros((650,1000),dtype=float)
print(foto.shape)
#np.reshape(foto,(650,1000))

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
        #normalizar 0- 255
        #sig = (sig-min(sig)) / (max(sig)-min(sig))
        #sig=sig*255
        for i in range(650):
           print('iteracion',i)
           foto[i]= np.array(sig[i*1000:(i+1)*1000])
           #for j in range(1000):
                #foto[i][j] = sig[i*1000+j] 
                
        foto = np.array(foto)
        print('dimenciones de foto ',foto.shape)
        plt.imshow(foto,cmap="plasma")
        cb = plt.colorbar()
        cb.set_label('Number of entries')
        plt.show()
        #foto = np.reshape(650,1000)
        #print(foto.shape)
        #print(np.shape(np.reshape(np.matrix(foto,(650,1000)))))
        
        x = input()
        
        sig_len = len(sig)
        print("sig_len ",sig_len)
        sym = ann.symbol
        print("ann.symbol ",sym)
        #print(sig)
        pos = ann.sample
        print("ann.sample ",pos)
        beat_len = len(sym)
        print("beat_len ",beat_len)
        #wfdb.plot_wfdb(record=r,annotation=ann,title=d,time_units='seconds')
        #input ()
        
X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)
#   print(Counter(Y))

