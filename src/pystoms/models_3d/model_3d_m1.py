"""Probabilistic 3d model for LC-IMS-MS precursor features
model M1
"""

from typing import Dict, Optional
import pymc as pm
import pymc.math as pmath
import numpy as np
import numpy.typing as npt
from pystoms.aligned_feature_data import AlignedFeatureData
from scipy.special import factorial
from numpy.typing import ArrayLike
from pystoms.models_3d.abstract_3d import AbstractModel

# typing
NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int64]


class ModelM1(AbstractModel):
    """3D model of precursor feature

    Model fitting the 2D function

    .. math::

        f(mz,scan) = α*f_{NM}(mz)*f_{N}(scan)

    With :math:`f_{NM}(mz)` being the pdf of a normal
    mixture distribution and :math:`f_{N}(scan)` being
    the pdf of a normal distribution.

    Args:
        features (AlignedFeatureData): Feature's data wrapped in
            pystoms.AlignedFeatureData container.
        model_parameters (Dictionary, optional): parameters for
            priors and hyperpriors. Defaults to None.
        likelihood (str, optional): Likelihood distribution. Currently
            supported: 'Normal', 'StudentT'. Defaults to 'Normal'.
        random_number_generator(np.random.Generator, optional): random number
            generator for pymc sampling processes. Defaults to None.
        name (str,optional): Defaults to empty string.
        coords (Optional[dict[str,ArrayLike]],optional):
            Coordinates for dims of model.
    Raises:
        NotImplementedError if provided likelihood is not supported.

    """

    def __init__(
        self,
        features: AlignedFeatureData,
        model_parameters: Optional[Dict] = None,
        likelihood: str = "Normal",
        random_number_generator: Optional[np.random.Generator] = None,
        name: str = "",
        coords: Optional[dict[str, ArrayLike]] = None,
    ) -> None:

        feature_ids = features.feature_data.feature.values
        batch_size = len(feature_ids)
        if model_parameters is None:
            model_parameters = {}
        self.model_parameters = model_parameters.copy()
        super().__init__(
            feature_ids, batch_size, random_number_generator, name, coords=coords
        )
        self.setup_mutable_data(features)
        # TODO is there a nicer way to share this between init and setup_mutable_data()
        dims_2d = ["data_point", "feature"]
        dims_3d = ["data_point", "feature", "isotopic_peak"]
        # priors
        # IMS
        self.i_t = pm.Normal(
            "i_t", mu=self.ims_mu, sigma=self.ims_sigma_max / 2, dims=dims_2d
        )
        self.i_s = pm.Uniform("i_s", lower=0, upper=self.ims_sigma_max, dims=dims_2d)

        # mass spec
        self.ms_mz = pm.Normal(
            "ms_mz", mu=self.mz_mu, sigma=self.mz_sigma, dims=dims_3d
        )
        # TODO(Tim) separate mz_sigma
        self.ms_s = pm.Exponential("ms_s", lam=self.mz_sigma, dims=dims_3d)
        self.pos = self.peaks / (self.charge) + self.ms_mz
        self.lam = 0.000594 * (self.charge) * self.ms_mz - 0.03091
        self.ws_matrix = self.lam**self.peaks / self.factorials * pmath.exp(-self.lam)

        # scalar α
        self.alpha = pm.Exponential("alpha", lam=self.alpha_lam, dims=dims_2d)
        # α*f_IMS(t)
        self.pi1 = self.alpha * pmath.exp(
            -((self.i_t - self.scan) ** 2) / (2 * self.i_s**2)
        )
        # f_mass(mz)
        self.pi2 = pmath.sum(
            self.ws_matrix
            * pmath.exp(-((self.pos - self.mz) ** 2) / (2 * self.ms_s**2)),
            axis=2,
        )

        # f(t,mz) = α*f_IMS(t)*f_mass(MZ)
        self.pi = pm.Deterministic(
            "mu", var=self.pi1 * self.pi2, auto=True, dims=dims_2d
        )
        # debug deterministic:
        # self.pi = self.pi1*self.pi2
        # Model error
        self.me = pm.HalfNormal("me", sigma=self.me_sigma, dims=dims_2d)
        # Likelihood
        if likelihood == "Normal":
            self.obs = pm.Normal(
                "obs", mu=self.pi, sigma=self.me, observed=self.intensity, dims=dims_2d
            )
        elif likelihood == "StudentT":
            self.obs = pm.StudentT(
                "obs",
                nu=5,
                mu=self.pi,
                sigma=self.me,
                observed=self.intensity,
                dims=dims_2d,
            )
        else:
            raise NotImplementedError("This likelihood is not supported")

        self._initialize_idata()

    def setup_mutable_data(
        self,
        features: AlignedFeatureData,
        num_isotopic_peaks: int = 6,
        standardize: bool = False,
    ) -> None:

        # get dimensions
        dataset = features.feature_data
        num_features = dataset.dims["feature"]
        num_data_points = dataset.dims["data_point"]
        # get observed data
        scan = dataset.Scan.values
        intensity = dataset.Intensity.values
        mz = dataset.Mz.values.reshape((num_data_points, num_features, 1))
        charge = dataset.Charge.values

        # hyper priors
        if standardize:
            pass
        else:
            # reshape is necessary here, because average deletes first
            # dimension
            ims_mu = np.average(scan, axis=0, weights=intensity).reshape(
                (1, num_features)
            )
            ims_sigma_max = np.max(scan, axis=0) - np.min(scan, axis=0).reshape(
                (1, num_features)
            )
            mz_mu = np.average(
                mz.reshape((num_data_points, num_features)),
                axis=0,
                weights=intensity,
            ).reshape((1, num_features, 1))
            self.model_parameters.setdefault("mz_sigma", 10)
            mz_sigma = (
                np.ones((1, num_features, 1), dtype="float")
                * self.model_parameters["mz_sigma"]
            )
            self.model_parameters.setdefault("alpha_lam", 1 / intensity.max(axis=0))
            alpha_lam = (
                np.ones((1, num_features), dtype="float")
                * self.model_parameters["alpha_lam"]
            )
            self.model_parameters.setdefault("me_sigma", 10)
            me_sigma = (
                np.ones((1, num_features), dtype="float")
                * self.model_parameters["me_sigma"]
            )
            z = np.array(charge).reshape((1, num_features, 1))
            mz_tile = np.tile(mz, (1, 1, num_isotopic_peaks))
            peaks = np.arange(num_isotopic_peaks)
            peaks = peaks.reshape((1, 1, num_isotopic_peaks))
            peaks_tile = np.tile(peaks, (num_data_points, num_features, 1))

        dims_2d = ["data_point", "feature"]
        dims_3d = ["data_point", "feature", "isotopic_peak"]
        self.intensity = pm.MutableData(
            "intensity", intensity, broadcastable=(False, False), dims=dims_2d
        )

        self.ims_mu = pm.MutableData(
            "ims_mu", ims_mu, broadcastable=(True, False), dims=dims_2d
        )
        self.ims_sigma_max = pm.MutableData(
            "ims_sigma_max", ims_sigma_max, broadcastable=(True, False), dims=dims_2d
        )
        self.scan = pm.MutableData(
            "scan", scan, broadcastable=(False, False), dims=dims_2d
        )

        self.alpha_lam = pm.MutableData(
            "alpha_lam", alpha_lam, broadcastable=(True, False), dims=dims_2d
        )
        self.me_sigma = pm.MutableData(
            "me_sigma", me_sigma, broadcastable=(True, False), dims=dims_2d
        )

        self.charge = pm.MutableData(
            "charge", z, broadcastable=(True, False, True), dims=dims_3d
        )
        self.mz = pm.MutableData(
            "mz", mz_tile, broadcastable=(False, False, False), dims=dims_3d
        )
        self.peaks = pm.MutableData(
            "peaks", peaks_tile, broadcastable=(False, False, False), dims=dims_3d
        )
        self.factorials = pm.MutableData(
            "factorials",
            factorial(peaks_tile),
            broadcastable=(False, False, False),
            dims=dims_3d,
        )
        self.mz_mu = pm.MutableData(
            "mz_mu", mz_mu, broadcastable=(True, False, True), dims=dims_3d
        )
        self.mz_sigma = pm.MutableData(
            "mz_sigma", mz_sigma, broadcastable=(True, False, True), dims=dims_3d
        )

    def _set_grid_data(self) -> None:
        """Set model's pm.MutableData container to grid data

        Used for prediction on a grid.
        """
        data = self.idata.constant_data
        # calculate hull boundaries of feature
        # isotopic peak dimension of mz values is repetition
        # of same value to fit shapes of peaks tensor
        obs_mz = data.mz.isel(isotopic_peak=0)
        obs_scan = data.scan
        peak_num = data.dims["isotopic_peak"]
        feature_num = data.dims["feature"]

        # set axis limits accordingly
        mz_min = obs_mz.min(dim="data_point").values - 1
        mz_max = obs_mz.max(dim="data_point").values + 1
        scan_min = obs_scan.min(dim="data_point").values - 1
        scan_max = obs_scan.max(dim="data_point").values + 1

        # draw axis
        scan_grid_num = 10
        mz_grid_num = 100
        grid_num = scan_grid_num * mz_grid_num
        scan_axes = np.linspace(scan_min, scan_max, num=scan_grid_num)
        mz_axes = np.linspace(mz_min, mz_max, num=mz_grid_num)

        # calculate grids
        mz_grids = np.zeros((feature_num, grid_num))
        scan_grids = np.zeros((feature_num, grid_num))
        for i in range(feature_num):
            x, y = np.meshgrid(mz_axes[:, i], scan_axes[:, i])
            mz_grids[i] = x.flatten()
            scan_grids[i] = y.flatten()
        # reshape into shape of shared variables
        mz_grids = mz_grids.T.reshape((-1, feature_num, 1))
        mz_grids = np.tile(mz_grids, (1, 1, peak_num))
        peaks = np.arange(peak_num).reshape(1, 1, -1)
        peaks = np.tile(peaks, (grid_num, feature_num, 1)).astype("int")
        scan_grids = scan_grids.T
        # get rest of data, necessary to reset
        # these as well to run properly
        # parameters that are broadcasted for a certaui
        # dimension are stored in idata with a lot of nan
        # to fit arrays to other data with same dimensions

        # parameters below all only change for dimension "feature"
        # value of each parameter for a given feature is at
        # first position in "isotopic_peak" and "data_point" dim

        # slicers for 3d and 2d parameters
        slice_3d = {"isotopic_peak": 0, "data_point": 0}
        slice_2d = {"data_point": 0}
        # data put into model must have same number of dimensions
        # there are 2d and 3d parameters.
        s_3d = (1, feature_num, 1)
        s_2d = (1, feature_num)
        # extract parameters as 1d array and reshape it into 2d or 3d array
        charge = data.charge.isel(slice_3d).values.astype("int").reshape(s_3d)
        ims_mu = data.ims_mu.isel(slice_2d).values.reshape(s_2d)
        ims_sigma_max = data.ims_sigma_max.isel(slice_2d).values.reshape(s_2d)
        mz_mu = data.mz_mu.isel(slice_3d).values.reshape(s_3d)
        mz_sigma = data.mz_sigma.isel(slice_3d).values.reshape(s_3d)
        alpha_lam = data.alpha_lam.isel(slice_2d).values.reshape(s_2d)
        me_sigma = data.me_sigma.isel(slice_2d).values.reshape(s_2d)
        pm.set_data(
            {
                "scan": scan_grids,
                "mz": mz_grids,
                "intensity": np.zeros_like(scan_grids, dtype="float"),
                "charge": charge,
                "peaks": peaks,
                "ims_mu": ims_mu,
                "ims_sigma_max": ims_sigma_max,
                "mz_mu": mz_mu,
                "mz_sigma": mz_sigma,
                "alpha_lam": alpha_lam,
                "me_sigma": me_sigma,
                "factorials": factorial(peaks),
            },
            model=self,
        )

    def _set_observed_data(self) -> None:
        """Set model's pm.MutableData container (back) to observed data"""
        data = self.idata.constant_data
        # extract original observed values
        mz = data.mz.values
        scan = data.scan.values
        intensity = data.intensity.values
        peaks = data.peaks.values
        feature_num = data.dims["feature"]
        # get rest of data, necessary to reset
        # these as well to run properly
        # parameters that are broadcasted for a certaui
        # dimension are stored in idata with a lot of nan
        # to fit arrays to other data with same dimensions

        # parameters below all only change for dimension "feature"
        # value of each parameter for a given feature is at
        # first position in "isotopic_peak" and "data_point" dim

        # slicers for 3d and 2d parameters
        slice_3d = {"isotopic_peak": 0, "data_point": 0}
        slice_2d = {"data_point": 0}
        # data put into model must have same number of dimensions
        # there are 2d and 3d parameters.
        s_3d = (1, feature_num, 1)
        s_2d = (1, feature_num)
        # extract parameters as 1d array and reshape it into 2d or 3d array
        charge = data.charge.isel(slice_3d).values.astype("int").reshape(s_3d)
        ims_mu = data.ims_mu.isel(slice_2d).values.reshape(s_2d)
        ims_sigma_max = data.ims_sigma_max.isel(slice_2d).values.reshape(s_2d)
        mz_mu = data.mz_mu.isel(slice_3d).values.reshape(s_3d)
        mz_sigma = data.mz_sigma.isel(slice_3d).values.reshape(s_3d)
        alpha_lam = data.alpha_lam.isel(slice_2d).values.reshape(s_2d)
        me_sigma = data.me_sigma.isel(slice_2d).values.reshape(s_2d)
        pm.set_data(
            {
                "scan": scan,
                "mz": mz,
                "intensity": intensity,
                "charge": charge,
                "peaks": peaks,
                "ims_mu": ims_mu,
                "ims_sigma_max": ims_sigma_max,
                "mz_mu": mz_mu,
                "mz_sigma": mz_sigma,
                "alpha_lam": alpha_lam,
                "me_sigma": me_sigma,
                "factorials": factorial(peaks),
            },
            model=self,
        )
