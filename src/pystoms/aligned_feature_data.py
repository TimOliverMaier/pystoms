""" Class holding data of one or more features
"""
from logging import warning
from typing import List, Tuple, Union
import xarray as xa
import pandas as pd
import numpy as np
import numpy.typing as npt
from proteolizarddata.data import PyTimsDataHandleDDA
from proteolizardalgo.feature_loader_dda import FeatureLoaderDDA

# typing
NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int64]


class AlignedFeatureData:
    """This class manages import, alignment and processing
        of feature data.

    Using FeatureLoaderDDA the features are stored in an aligned fashion,
    meaning from all features 'num_data_points' highest signals are chosen.
    Features that do not have enough signals are discarded.

    Attributes:
        feature_data (xa.Dataset): Feature data is stored in
            xarray format. If several features are aligned 'feature'
            dimensions stores the feature's ids as coordinates.
            Note: If only one feature is used, the dimension 'feature'
            still exists, however length of 'datapoint' dimension is
            depending on the feature's size and not on 'num_data_points'
        accepted_feature_ids (List[int]): List of feature ids, that are
            stored in feature_data.
    Args:
        data_handle (Union[str,PyTimsDataHandleDDA]): Path to experimental data
            or data_handle.
        ids (List[int]): List with feature ids.
        is_parallel (bool, optional): Wether several features are to
            be aligned. If True 'num_data_points' highest signals
            are loaded from features, if False all data points of the
            single feature are loaded. Defaults to True.
        num_data_points (int, optional): Number of data_points to load
            from features to align. Defaults to 10.
    """

    def __init__(
        self,
        data_handle: Union[str, PyTimsDataHandleDDA],
        ids: List[int],
        is_parallel: bool = True,
        num_data_points=10,
    ) -> None:
        self.accepted_feature_ids = []
        charges = []
        feature_data = []
        if isinstance(data_handle, str):
            data_path = data_handle
            data_handle = PyTimsDataHandleDDA(data_path)
        for feature_id in ids:
            feature = FeatureLoaderDDA(data_handle, feature_id)
            if np.isnan(feature.monoisotopic_mz):
                continue
            try:
                # estimate feature hull boundaries with
                # averagine model for isotopic pattern
                # and gaussian model for IMS
                data_tmp = feature.load_hull_data_3d(
                    intensity_min=0,
                    ims_model="gaussian",
                    plot_feature=False,
                )
            except RuntimeError:
                warning(f"RuntimeError with feature {feature_id}")
                continue
            if is_parallel:
                if len(data_tmp) < num_data_points:
                    continue
                feature_data.append(
                    data_tmp.nlargest(num_data_points, columns="Intensity")
                )
            else:
                feature_data.append(data_tmp)
            self.accepted_feature_ids.append(feature_id)
            charges.append(feature.charge)

        if len(self.accepted_feature_ids) == 0:
            raise ValueError("No accepted features. Check chosen features.")
        s, mz, i = self._set_data_parallel(feature_data)
        data_dict = {
            "Charge": ("feature", charges),
            "Scan": (("data_point", "feature"), s),
            "Mz": (("data_point", "feature"), mz),
            "Intensity": (("data_point", "feature"), i),
        }
        coord_dict = {"feature": self.accepted_feature_ids}
        self.feature_data = xa.Dataset(data_vars=data_dict, coords=coord_dict)

    def _set_data_parallel(
        self, feature_data_list: List[pd.DataFrame]
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Stacking features in axis 1

        Args:
            feature_data_list (List[pd.DataFrame]): Loaded feature dataframes.
        Returns:
            Tuple[NDArrayFloat,NDArrayFloat,NDArrayFloat]: stacked scan, mz and
                intensity data.
        """
        s = np.stack([fd["Scan"] for fd in feature_data_list], axis=1)
        mz = np.stack([fd["Mz"] for fd in feature_data_list], axis=1)
        i = np.stack([fd["Intensity"] for fd in feature_data_list], axis=1).astype(
            "float"
        )

        return (s, mz, i)
