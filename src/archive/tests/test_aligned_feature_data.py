import numpy as np
import os
from pystoms.aligned_feature_data import AlignedFeatureData
from proteolizarddata.data import PyTimsDataHandleDDA
from xarray import Dataset

rng = np.random.default_rng(2022)
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
        assert isinstance(aligned_features.feature_data, Dataset)
