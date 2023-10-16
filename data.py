import numpy as np
import keras
import random
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def corruption(x_train, y_train, size=0.5):
    '''
        randomly assignement of labels (different from the true label)
    '''
    t50=int(len(x_train)//(1/size))

    y_train_adv = []   
    for y in y_train[0:t50]: 
      c = [0,1,2,3,4,5,6,7,8,9]
      c.pop(y)
      y_train_adv.append(np.random.choice(c))
    y_train_adv = np.concatenate((y_train_adv,y_train[t50:]))
    return y_train_adv

def data_ge(): 
    '''
        get corruped fascion MNIST data set
    '''
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.1)
    listindex = np.arange(0,len(x_train))
    random.shuffle(listindex)
    x_train = x_train[listindex]
    y_train = y_train[listindex]
    y_train_adv50 = corruption(x_train, y_train, size=0.5)
    return x_train, y_train_adv50, x_val, y_val, x_test, y_test

def results(ypred, ytrue):
    '''
        show the performance for test set
    '''
    print(classification_report(ytrue,ypred))
    cm = confusion_matrix(ytrue,ypred)
    ConfusionMatrixDisplay(cm).plot()