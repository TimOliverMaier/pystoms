"""This script is showing how modles can be serialized and read from pkl
format using cloudpickle"""
import numpy as np
from pystoms.aligned_feature_data import AlignedFeatureData
from pystoms.models_glm import ModelGLM3D
from proteolizarddata.data import PyTimsDataHandleDDA
import cloudpickle

# get data
data_path = "/home/tim/Master/MassSpecDaten/M210115_001_Slot1-1_1_850.d/"
data_handle = PyTimsDataHandleDDA(data_path)
total_features = 1
feature_ids = np.random.randint(1000,4001,size=total_features)
aligned_features = AlignedFeatureData(
                            data_handle,
                            ids=feature_ids,
                            is_parallel=True)
# create model
model = aligned_features.generate_model()
# store as pkl
cloudpickle.dump(model,open("data/model.pkl","wb"))
# read as pkl
model_file = open("data/model.pkl","rb")
model_read = cloudpickle.loads(model_file.read())
model_file.close()
print(model_read.idata)
