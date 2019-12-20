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
r_seed = int(sys.argv[1])
print(sys.argv)

data_names = ['100', '101', '102', '103', '104', '105', '106', '107', 
              '108', '109', '111', '112', '113', '114', '115', '116', 
              '117', '118', '119', '121', '122', '123', '124', '200', 
              '201', '202', '203', '205', '207', '208', '209', '210', 
              '212', '213', '214', '215', '217', '219', '220', '221', 
              '222', '223', '228', '230', '231', '232', '233', '234']

widb = 99
wida = 160
n_samples = 650000 #<650000
foto= []
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
        
        '''for i in range(649):
           foto= [] + sig[i:(i+1)*1000].tolist()
           #for j in range(1000):
                #foto[i][j] = sig[i*1000+j] 
                
        print(np.shape(np.reshape(np.matrix(foto,(650,1000)))))
        x = input()'''
            
        '''plt.plot(sig)
        plt.title=('normal signal')
        #quitar ruido
        sig = denoise(sig, 0.005, 'bior1.3')
        plt.plot(sig)
        
        #normalizar
        sig = (sig-min(sig)) / (max(sig)-min(sig))
        
        plt.plot(sig)
        plt.title=('denoised signal')
        plt.legend(['normal','denoised','normalized'])'''
        
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
        
        x_to_plot=[]
        y_to_plot=[]
        for i in range(beat_len):
            if sym[i] in labels and pos[i]-wid>=0 and pos[i]+wid+1<=sig_len:
                a = sig[pos[i]-wid:pos[i]+wid+1]
                print(a)
                print('rangue ',i,' :',pos[i]-wid," - ",pos[i]+wid+1)
                if len(a) != 2*wid+1:
                    print("Length error")
                    continue
                X.append(a)
                x_to_plot.append(a)
                Y.append(labels.index(sym[i]))
                y_to_plot.append(labels.index(sym[i]))
                print('sym[i] :',sym[i])
                print('labels.index(sym[i])   ',labels.index(sym[i]))
        #print('X : ',X)
        #print('Y : ',Y)
        '''for j in range(len(x_to_plot)):
            plt.subplot(len(y_to_plot),1,j+1)
            plt.ylabel(j)
            plt.plot(x_to_plot[j])    
        plt.show()
        wfdb.plot_wfdb(record=r,annotation=ann,title=d+" crudo",time_units='seconds')'''
        
X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)
#   print(Counter(Y))

