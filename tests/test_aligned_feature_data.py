import numpy as np
from pystoms.aligned_feature_data import AlignedFeatureData
from pystoms.models_3d.models_glm import ModelGLM3D
from proteolizarddata.data import PyTimsDataHandleDDA

import pytest

# PARAMETERS
data_path = "/home/tim/Master/MassSpecDaten/M210115_001_Slot1-1_1_850.d/"
data_handle = PyTimsDataHandleDDA(data_path)
total_features = 2
feature_ids = np.random.random_integers(1000,4000,size=total_features)


class TestAlignedFeatureData():

    def test_general_aligned_feature_data(self):

        aligned_features = AlignedFeatureData(
                                data_handle,
                                ids=feature_ids,
                                is_parallel=True)
        model = aligned_features.generate_model()
        assert(isinstance(model,ModelGLM3D))
