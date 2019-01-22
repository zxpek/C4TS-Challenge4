
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def cleanLongLat(l):
    split = l.str.split(',', expand=True)
    split = (split[0]+'.'+[i if len(i)>1 else i+'0' for i in split[1]]).astype(float)
    return(split)
    
class scaler:
    def __init__(self, x = None):
        if type(x) == pd.core.frame.DataFrame:
            self.fit(x)
        elif x == None:
            self.x = None
            self.mean = None
            self.var = None
        else:
            raise Exception('Require pandas.DF input')


    def fit(self, x):
        self.x = x
        self.mean = x.mean()
        self.var = x.var()

    def scale(self, new_x):
        result = (new_x - self.mean) / np.sqrt(self.var)
        return (result)
    
def prepare(X, fit = False, scaler_obj = None, pca_obj = None):
    
    if fit:
        s = scaler(X)
        pca = PCA()
        X_ = s.scale(X)
        X_ = pca.fit_transform(X_)
        X_ = X_[:,:10]
        
        return(X_, s, pca)
    else:
        if scaler_obj == None or pca_obj == None:
            raise Exception('Non fitting requires scaler/pca obj')
        X_ = pca_obj.transform(scaler_obj.scale(X))[:,:10]
        return X_
