import numpy as np
import os
from pystoms.aligned_feature_data import AlignedFeatureData
from pystoms.models_3d.model_3d_m1 import ModelM1
from proteolizarddata.data import PyTimsDataHandleDDA
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

rs = RandomState(MT19937(SeedSequence(123456789)))
file_path = os.path.dirname(__file__)
rel_data_path = "../../MassSpecDaten/M210115_001_Slot1-1_1_850.d/"
data_path = os.path.join(file_path, rel_data_path)
data_handle = PyTimsDataHandleDDA(data_path)
total_features = 2
feature_ids = np.random.random_integers(1000, 4000, size=total_features)


class TestAlignedFeatureData:
    def test_general_aligned_feature_data(self):

        aligned_features = AlignedFeatureData(
            data_handle, ids=feature_ids, is_parallel=True
        )
        model = aligned_features.generate_model()
        assert isinstance(model, ModelM1)
