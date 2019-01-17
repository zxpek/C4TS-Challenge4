
 
import pickle
import json
import numpy
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model
import pandas as pd
import numpy as np
from utils import scaler

def cleanLongLat(l):
    split = l.str.split(',', expand=True)
    split = (split[0]+'.'+[i if len(i)>1 else i+'0' for i in split[1]]).astype(float)
    return(split)


def init():
    from utils import scaler

    global model
    global pca_transform
    global s
        
    s = scaler(pd.DataFrame([1,2]))
    model_id = "AutoMLee562a66fbest"
    #model_path = Model.get_model_path(model_name = model_id)
    # deserialize the model file back into a sklearn model
    #model = joblib.load(model_path)
    pca_transform = pickle.load(open('pca_transform.pkl', 'rb'))
    s = pickle.load(open('scaler.pkl','rb'))

def run(rawdata):
    try:
        df = json.loads(j_test)['data']
        df = pd.DataFrame.from_records(df)
        df['Latitude'] = cleanLongLat(df['Latitude'])
        df['Longitude'] = cleanLongLat(df['Longitude'])
        df.drop(['Machine_ID', 'District'], axis=1, inplace=True)
        df.drop('Failure_NextHour', axis = 1, inplace=True)
        df = df.apply(pd.to_numeric)
        data = pca_transform.transform(s.scale(df))
        result = best_model.predict(data) #Changed to best_model here
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
    return json.dumps({"result":result.tolist()}) 
    

if __name__ == "__main__":
	init()
	print(s)
