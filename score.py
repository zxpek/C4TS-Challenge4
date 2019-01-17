
 
import pickle
import json
import numpy
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model
import pandas as pd
import numpy as np

def cleanLongLat(l):
    split = l.str.split(',', expand=True)
    split = (split[0]+'.'+[i if len(i)>1 else i+'0' for i in split[1]]).astype(float)
    return(split)

def scale(x):
    s_mean = pd.Series([1018.6536186748193, 1.7702220063430383, 859.1854052972942, 443661.2680362296, 269.6304465841881, 226.50495728449386, 233.054087259636, 515.351352895797, 1.361667476213606, 313.32132346638474, 0.18909111688905397, 13.01694334123832, 49.73222092059773, 5.895254150118575, 1.085973884968142, 0.7912226063601817, 63.65644732706649, 338.97779936569617, 78.55941598331381, 0.9553415811880339, 0.1976913626103603, 1.3385345295579019, 103.82176690763242],['Density_Overload', 'Abnormal_Flow_Rate', 'Heat_Flow', 'Asset_Integrity', 'Temperature_Differential', 'Volumetric_Flow_Rate', 'Tangential_Stress', 'Duct_Lengths_in_Units', 'Fault_in_last_Month', 'Avg_hours_in_Use', 'Pressure_Alarm', 'Inclination_Angle', 'Location_Fault_Code', 'Operating_Pressure_above_Normal', 'Vandalism_Reports', 'Compression_Ratio', 'Multiple_Connects', 'Water_Exposure_units', 'Humidity_Factor', 'Cathodic_Protection', 'Pressure_Class', 'Latitude', 'Longitude'])
    s_var = pd.Series([2513529.9777650624, 1113.1883000570303, 1959427.1616068266, 148713873848.40573, 768178.5282640787, 77121.39970541374, 80938.43409944583, 369590.581359321, 10.430767846333982, 159138.84074506673, 0.2866048056581768, 251.76847112977725, 6363.065443013388, 56.590319495798596, 112.62690166816, 0.7542563456388259, 18409.98951720073, 403728.7554076676, 3777.931544501477, 0.042665263483529926, 0.15861401971831993, 0.0018770140014010134, 0.006205724168020728],['Density_Overload', 'Abnormal_Flow_Rate', 'Heat_Flow', 'Asset_Integrity', 'Temperature_Differential', 'Volumetric_Flow_Rate', 'Tangential_Stress', 'Duct_Lengths_in_Units', 'Fault_in_last_Month', 'Avg_hours_in_Use', 'Pressure_Alarm', 'Inclination_Angle', 'Location_Fault_Code', 'Operating_Pressure_above_Normal', 'Vandalism_Reports', 'Compression_Ratio', 'Multiple_Connects', 'Water_Exposure_units', 'Humidity_Factor', 'Cathodic_Protection', 'Pressure_Class', 'Latitude', 'Longitude'])
    
    result = (x - s_mean) / np.sqrt(s_var)
    return (result)
    
def init():

    global model
    global pca_transform
    
    model_id = "AutoMLee562a66fbest"
    model_path = Model.get_model_path(model_name = model_id)
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    pca_transform = pickle.load(open('pca_transform.pkl', 'rb'))


def run(rawdata):
    try:
        df = json.loads(rawdata)['data']
        df = pd.DataFrame.from_records(df)
        df['Latitude'] = cleanLongLat(df['Latitude'])
        df['Longitude'] = cleanLongLat(df['Longitude'])
        df.drop(['Machine_ID', 'District'], axis=1, inplace=True)
        df.drop('Failure_NextHour', axis = 1, inplace=True)
        df = df.apply(pd.to_numeric)
        data = pca_transform.transform(scale(df))
        result = model.predict(data)
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
    return json.dumps({"result":result.tolist()}) 
    
