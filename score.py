
 
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
    s_mean = pd.Series([1018.3124861106702, 1.7504047747547542, 856.766437029747, 443319.28505666845, 267.7161497190387, 226.50373027715165, 233.13035334455063, 516.0932410552716, 1.3675354773167403, 313.3358201847678, 0.1875932569287914, 13.010889234578876, 49.57471665767167, 5.892504523953141, 1.088415505254135, 0.7893583923299152, 63.58341534651894, 338.8552969935553, 78.57674846820534, 0.9547604685863044, 0.19664116321153052, 1.3387101177814704, 103.82191688625682],['Density_Overload', 'Abnormal_Flow_Rate', 'Heat_Flow', 'Asset_Integrity', 'Temperature_Differential', 'Volumetric_Flow_Rate', 'Tangential_Stress', 'Duct_Lengths_in_Units', 'Fault_in_last_Month', 'Avg_hours_in_Use', 'Pressure_Alarm', 'Inclination_Angle', 'Location_Fault_Code', 'Operating_Pressure_above_Normal', 'Vandalism_Reports', 'Compression_Ratio', 'Multiple_Connects', 'Water_Exposure_units', 'Humidity_Factor', 'Cathodic_Protection', 'Pressure_Class', 'Latitude', 'Longitude'])
    s_var = pd.Series([2517027.6315063024, 1065.216575095123, 1954634.0917101507, 148756206721.0674, 764132.956667073, 76893.30307675204, 80870.7758832055, 370331.0477856238, 10.621120127599418, 158812.33683707696, 0.2865106179695057, 251.13585830824238, 6367.112289805666, 56.3985656333141, 115.93951243950882, 0.7537428704395421, 18527.598183252845, 404949.04503282363, 3777.5525179547, 0.04319428750190964, 0.15797843148983523, 0.0018802365891634565, 0.006215126624172454],['Density_Overload', 'Abnormal_Flow_Rate', 'Heat_Flow', 'Asset_Integrity', 'Temperature_Differential', 'Volumetric_Flow_Rate', 'Tangential_Stress', 'Duct_Lengths_in_Units', 'Fault_in_last_Month', 'Avg_hours_in_Use', 'Pressure_Alarm', 'Inclination_Angle', 'Location_Fault_Code', 'Operating_Pressure_above_Normal', 'Vandalism_Reports', 'Compression_Ratio', 'Multiple_Connects', 'Water_Exposure_units', 'Humidity_Factor', 'Cathodic_Protection', 'Pressure_Class', 'Latitude', 'Longitude'])
    
    result = (x - s_mean) / np.sqrt(s_var)
    return (result)
    
def init():

    global model
    global pca_transform
    
    model_id = "AutoML057e77cdebest"
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
        df = df.apply(pd.to_numeric)
        data = pca_transform.transform(scale(df))[:,:10]
        result = model.predict(data)
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
    return json.dumps({"result":result.tolist()}) 
    
