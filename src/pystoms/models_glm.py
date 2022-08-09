"""Probabilistic General linear models for LC-IMS-MS precursor features



"""

from logging import warning
from typing import List,Optional
import pandas as pd
import pymc as pm
import pymc.math as pmath
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import plotly.express as px
import arviz as az
from aesara import tensor as at
from aesara.tensor.sharedvar import TensorSharedVariable
import matplotlib.pyplot as plt
import os
from numpy.typing import ArrayLike
#typing
NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt   = npt.NDArray[np.int64]


class AbstractModel(pm.Model):
    """Abstract model class for model evaluation.

    This class provides the 2D GLM model subclasses with
    evaluation methods.

    Args:
        feature_ids(List[int]): List of feature ids in batch.
        batch_size (int): Number of features
            in one batch.
        name (str): Name of model.
        coords (Optional[dict[str,ArrayLike],optional]):
            Coordinates for dims in model.
    Attributes:
        feature_ids(List[int]): List of feature ids in batch.
        batch_size (int): Number of features in
            one batch.
        idata (az.InferenceData):
          inferenceData of current model.

    """
    def __init__(self,
                 feature_ids:List[int],
                 batch_size:int,
                 name:str,
                 coords:Optional[dict[str,ArrayLike]] = None) -> None:
        # name and model must be passed to pm.Model
        coords = {}
        coords.setdefault("feature",feature_ids)
        super().__init__(name,coords=coords)
        # instantiate inference data
        self.feature_ids = feature_ids
        self.batch_size = batch_size
        self.idata = az.InferenceData()

    def _reset_idata(self) -> None:
        self.idata = az.InferenceData()

    def _initialize_idata(self):
        """Inits InferenceData with constant_data group
        """
        prior_idata = pm.sample_prior_predictive(5,model=self)
        self.idata.add_groups({"constant_data":prior_idata.constant_data})

    def _get_model_shared_data(self,
                               predictions_constant_data:bool = True
                              ) -> az.InferenceData:
        """Get model's input data from model as idata.

        This function is depreceated!

        Args:
            predictions_constant_data (bool, optional): If True data is put in
                `predictions_constant_data` group. Else into `constant_data`
                 group. Defaults to True.

        Returns:
            az.InferenceData: InferenceData with model's input
                data in `constant_data` or `predictions_constant_data`
        """
        data = {}
        for key,var in self.named_vars.items():
            if isinstance(var, TensorSharedVariable):
                data[key] = var.get_value()
        if predictions_constant_data:
            idata = az.from_dict(predictions_constant_data=data)
        else:
            idata = az.from_dict(constant_data = data)
        return idata

    def _sample_predictive(self,
                           is_prior:bool = False,
                           is_grid_predictive:bool = False,
                           **kwargs) -> None:
        """Calls PyMC's predictions sampler and extends inference data.

        Args:
            is_prior (bool, optional): If True calls
              `pymc3.sample_prior_predictive`, if False calls
              `pymc3.sample_posterior_predictive`. Defaults to False.
            is_grid_predictive (bool, optional): If True make
              out-of-sample predictions with feature's
              grid data and store it in `idata.predictions`
              or `idata.prior_predictions`, depending on `is_prior`.
              Defaults to False.
            **kwargs: Keyword arguments passed to predictive sampler.
        """

        # for overview on arviz inferenceData groups visit
        # https://arviz-devs.github.io/arviz/schema/schema.html

        # only prior can be sampled before posterior was sampled
        if "posterior" not in self.idata.groups() and not is_prior:
            self._sample()

        if is_prior and is_grid_predictive:
            # prior out-of-sample predictions with grid
            # as predictors

            if "prior_predictions" not in self.idata.groups():
                self._set_grid_data()
                prior_prediction = pm.sample_prior_predictive(model=self,
                                                          **kwargs)
                if "mu" in prior_prediction.prior.data_vars.keys():
                    prior_prediction.prior_predictive["mu"]\
                        = prior_prediction.prior["mu"]
                warning("Prior out-of-sample predictions are currently \
                    a work around.\n Arviz will warn about non-defined \
                    InferenceData group 'prior_predictions'.")
                self.idata.add_groups({
                "prior_predictions":prior_prediction.prior_predictive,
                "predictions_constant_data":prior_prediction.constant_data
                })

            else:
                warning("Inference data of model already\n\
                         has prior_predictions")

            # else: do nothing, posterior out-of-sample predictions
            # were sampled before, providing predictions_constant_data

            self._set_observed_data()

        if is_prior and not is_grid_predictive:
            # prior predictions on coordinates of observed data
            # used for posterior sampling (in-sample predictions)
            if "prior" not in self.idata.groups():
                prior_idata = pm.sample_prior_predictive(**kwargs,model=self)
                if "mu" in prior_idata.prior.data_vars.keys():
                    prior_idata.prior_predictive["mu"]\
                        = prior_idata.prior["mu"]
                self.idata.extend(prior_idata)
            else:
                warning("Inference data of model already\n\
                        has prior_predictive")

        if not is_prior and is_grid_predictive:
            # posterior out of-sample-predictions with grid
            # as predictors
            if "predictions" not in self.idata.groups():
                self._set_grid_data()
                prediction = pm.sample_posterior_predictive(
                                                    self.idata,
                                                    model=self,
                                                    #extend_inferencedata=True,
                                                    predictions=True,
                                                    **kwargs)
                # until fix inference data must be updated
                self.idata.extend(prediction)
                self._set_observed_data()
            else:
                warning("Inference data of model already\n\
                        has predictions")

        if not is_prior and not is_grid_predictive:
            # posterior predictions on coordinates of observed data
            # used for posterior sampling (in-sample predictions)
            if "posterior_predictive" not in self.idata.groups():
                pm.sample_posterior_predictive(self.idata,
                                                extend_inferencedata=True,
                                                model=self,
                                                **kwargs)
            else:
                warning("Inference data of model already\n\
                        has posterior_predictive")

    def visualize_predictions_scatter(self,
                                      size:int = 50,
                                      in_sample:bool = True,
                                      is_prior:bool = False,
                                      pred_name:str = "obs",
                                      plot_observed_data:bool = True,
                                      write_to_file:bool = False,
                                      folder_path:str = ".",
                                      file_name:Optional[str] = None,
                                      use_renderer:str ="notebook") -> None:
        """Plotting posterior/prior predictions.

        Plotting posterior/prior predictions as plotly 3D scatter
        plots.

        Args:
            size (int,optional): Number of predictions to plot
              for each position. Defaults to 50.
            in_sample (bool, optional): If True in-sample
              predictions are plotted. Else out-of-sample
              predictions are plotted. Defaults to True.
            is_prior (bool, optional): If True prior
              predictions are plotted, else posterior.
              Defaults to False.
            pred_name (str,optional): predicted variable to plot.
              Defaults to 'obs'. If variable is not found, first
              variable in `data_vars` of Dataset is used.
            plot_observed_data (bool,optional): Wether to plot observed
              data on top of scatter plot. Defaults to True.
            write_to_file (bool, optional): Wether to write plot to file.
                Defaults to False.
            folder_path (str, optional): Path to folder in which output files
                are stored. Defaults to ".".
            file_name (Optional[str], optional): Name of html output file.
                If is None, generic file name depending on `is_prior`,
                `pred_name` and `in_sample` is used. Defaults to None.
            use_renderer (str,optional): Which plotly renderer to use.
              Defaults to 'notebook'.
        """
        # we want to plot predictors -> predicted
        # First, we need to get correct data depending on arguments

        if not in_sample:
            # out-of-sample grid data as predictors
            predictors_data = self.idata.predictions_constant_data

            if is_prior:
                # prior predictions on grid data
                # extract {size} random draws to reduce load
                draw_sample = np.random.choice(self.idata\
                                                .prior_predictions\
                                                .draw\
                                                .values,
                                                size)
                predicted_data = self.idata.\
                                    prior_predictions[{"draw":draw_sample}]
            if not is_prior:
                # posterior predictions on grid data
                draw_sample = np.random.choice(self.idata\
                                                .predictions\
                                                .draw\
                                                .values,
                                                size)
                predicted_data = self.idata.predictions[{"draw":draw_sample}]

        if in_sample:
            # original observed data as predictors
            predictors_data = self.idata.constant_data

            if is_prior:
                # prior predictions on observed data
                draw_sample = np.random.choice(self.idata\
                                                .prior_predictive\
                                                .draw\
                                                .values,
                                                size)
                predicted_data = self.idata\
                                    .prior_predictive[{"draw":draw_sample}]
            if not is_prior:
                # posterior predictions on observed data
                draw_sample = np.random.choice(self.idata\
                                                .posterior_predictive\
                                                .draw\
                                                .values,
                                                size)
                predicted_data = self.idata\
                                    .posterior_predictive[{"draw":draw_sample}]


        # get mz and scan values (predictors)
        # first extract variable from xarray dataset
        # because direct to_dataframe crashed
        df_mz = predictors_data.mz\
                    .to_dataframe()\
                    .xs(0,level="isotopic_peak")

        df_scan = predictors_data.scan\
                    .to_dataframe()
        # then merge to one df
        df_predictors = pd.merge(df_mz,
                                df_scan,
                                left_index=True,
                                right_index=True)
        # get corresponding predicted values
        # test if desired pred_name was sampled
        if pred_name not in predicted_data.data_vars.keys():
            # use first data variable in Dataset otherwise
            pred_name_new = list(predicted_data.data_vars.keys())[0]
            warning(f"{pred_name} not in 'prior_predictive',\
                    using {pred_name_new}")
            pred_name = pred_name_new

        df_predicted = getattr( predicted_data,pred_name)\
                                .to_dataframe()
        # merge to dataframe with predictors and predicted vars
        df_plot = pd.merge(
                          df_predictors,
                          df_predicted,
                          left_index = True,
                          right_index = True
                          ).reset_index()\
                          .astype({"chain":"str"})
        # plotting
        for feature in self.feature_ids:
            fig = px.scatter_3d(data_frame=df_plot[df_plot.feature==feature],
                                x="scan",
                                y="mz",
                                z=pred_name,
                                color="chain",
                                opacity=0.1)
            if plot_observed_data:
                obs_data_trace = self.plot_feature_data(return_fig_trace=True,
                                                        feature_ids=[feature])[0]
                fig.add_trace(obs_data_trace)

            if write_to_file:
                feature_path = folder_path + f"/feature{feature}/"
                if file_name is None:
                    fn_1 = "prior_predictive" \
                            if is_prior \
                            else "posterior_predictive"
                    fn_2 = "in_sample" if in_sample else "out_of_sample"
                    file_name = fn_1 + "_" + fn_2 + "_" + pred_name + ".html"

                if not file_name.endswith(".html"):
                    fn_prefix = file_name.split(".")[0]
                    file_name = fn_prefix + ".html"

                path_to_file = feature_path+file_name

                if not os.path.exists(feature_path):
                    os.makedirs(feature_path)
                fig.write_html(path_to_file)
            else:
                fig.show(renderer=use_renderer)

    def plot_feature_data(self,
                          return_fig_trace:bool = False,
                          feature_ids:Optional[List[int]] = None,
                          ) -> Optional[List[go.Scatter3d]]:
        """plots model's input feature data.

        Args:
            return_fig_trace (bool, optional): Wether to only return
                plotly 3D scatter trace. Defaults to False.
            feature_ids (Optional[List[int]], optional): Which features
                to plot. If None all features are plotted. Defaults to None.
        Returns:
            Optional[List[go.Scatter3d]]: If `return_fig_trace` is True,
                then list of plotly Scatter3d traces with observed data is
                returned.
        """
        if feature_ids is None:
            feature_ids = self.feature_ids
        scatter_traces = []
        for feature_id in feature_ids:
            data = self.idata.constant_data.sel(feature=feature_id)
            x = data.scan.values
            y = data.mz.values[:,0]
            z = data.intensity.values
            fig_trace = go.Scatter3d(x=x,
                                    y=y,
                                    z=z,
                                    mode="markers",
                                    marker=dict(color="black",size=10))
            scatter_traces.append(fig_trace)
            if not return_fig_trace:
                fig = go.Figure(data=[fig_trace])
                fig.show()
        if return_fig_trace:
            return scatter_traces

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
        mz_min = obs_mz.min(dim="data_point").values-1
        mz_max = obs_mz.max(dim="data_point").values+1
        scan_min = obs_scan.min(dim="data_point").values-1
        scan_max = obs_scan.max(dim="data_point").values+1

        # draw axis
        scan_grid_num = 10
        mz_grid_num = 100
        grid_num = scan_grid_num*mz_grid_num
        scan_axes = np.linspace(scan_min,scan_max,num=scan_grid_num)
        mz_axes = np.linspace(mz_min,mz_max,num=mz_grid_num)

        # calculate grids
        mz_grids = np.zeros((feature_num,grid_num))
        scan_grids = np.zeros((feature_num,grid_num))
        for i in range(feature_num):
            x,y = np.meshgrid(mz_axes[:,i],scan_axes[:,i])
            mz_grids[i] = x.flatten()
            scan_grids[i] = y.flatten()
        # reshape into shape of shared variables
        mz_grids = mz_grids.T.reshape((-1,feature_num,1))
        mz_grids = np.tile(mz_grids,(1,1,peak_num))
        peaks = np.arange(peak_num).reshape(1,1,-1)
        peaks = np.tile(peaks,(grid_num,feature_num,1)).astype("int")
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
        slice_3d = {"isotopic_peak":0,"data_point":0}
        slice_2d = {"data_point":0}
        # data put into model must have same number of dimensions
        # there are 2d and 3d parameters.
        s_3d = (1,feature_num,1)
        s_2d = (1,feature_num)
        # extract parameters as 1d array and reshape it into 2d or 3d array
        charge = data.charge.isel(slice_3d).values.astype("int").reshape(s_3d)
        ims_mu = data.ims_mu.isel(slice_2d).values.reshape(s_2d)
        ims_sigma_max = data.ims_sigma_max.isel(slice_2d).values.reshape(s_2d)
        mz_mu = data.mz_mu.isel(slice_3d).values.reshape(s_3d)
        mz_sigma = data.mz_sigma.isel(slice_3d).values.reshape(s_3d)
        alpha_lam = data.alpha_lam.isel(slice_2d).values.reshape(s_2d)
        me_sigma = data.me_sigma.isel(slice_2d).values.reshape(s_2d)
        pm.set_data({"scan":scan_grids,
                      "mz":mz_grids,
                      "intensity":np.zeros_like(scan_grids,dtype="float"),
                      "charge":charge,
                      "peaks":peaks,
                      "ims_mu":ims_mu,
                      "ims_sigma_max":ims_sigma_max,
                      "mz_mu":mz_mu,
                      "mz_sigma":mz_sigma,
                      "alpha_lam":alpha_lam,
                      "me_sigma":me_sigma,
                    },
                    model=self)

    def _set_observed_data(self) -> None:
        """Set model's pm.MutableData container (back) to observed data
        """
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
        slice_3d = {"isotopic_peak":0,"data_point":0}
        slice_2d = {"data_point":0}
        # data put into model must have same number of dimensions
        # there are 2d and 3d parameters.
        s_3d = (1,feature_num,1)
        s_2d = (1,feature_num)
        # extract parameters as 1d array and reshape it into 2d or 3d array
        charge = data.charge.isel(slice_3d).values.astype("int").reshape(s_3d)
        ims_mu = data.ims_mu.isel(slice_2d).values.reshape(s_2d)
        ims_sigma_max = data.ims_sigma_max.isel(slice_2d).values.reshape(s_2d)
        mz_mu = data.mz_mu.isel(slice_3d).values.reshape(s_3d)
        mz_sigma = data.mz_sigma.isel(slice_3d).values.reshape(s_3d)
        alpha_lam = data.alpha_lam.isel(slice_2d).values.reshape(s_2d)
        me_sigma = data.me_sigma.isel(slice_2d).values.reshape(s_2d)
        pm.set_data({"scan":scan,
                      "mz":mz,
                      "intensity":intensity,
                      "charge":charge,
                      "peaks":peaks,
                      "ims_mu":ims_mu,
                      "ims_sigma_max":ims_sigma_max,
                      "mz_mu":mz_mu,
                      "mz_sigma":mz_sigma,
                      "alpha_lam":alpha_lam,
                      "me_sigma":me_sigma,
                    },
                    model=self)

    def _sample(self,**kwargs) -> az.InferenceData:
        """Method to call PyMC sampler.

        Args:
            **kwargs: Keyword arguments for
                PyMC.sample()
        Returns:
            az.InferenceData: Sample trace as InferenceData.
        """
        kwargs.setdefault("return_inferencedata", True)
        trace = pm.sample(**kwargs,model=self)
        self.idata.extend(trace)
        return trace

    def arviz_plots(self,
                    var_names:Optional[List[str]] = None,
                    save_fig:bool = True,
                    path:str = ".") -> None:
        """Generate various arviz plots

        Args:
            var_names (Optional[List[str]], optional): Which variables
                to consider. If None, then
                ["i_t","i_s","alpha","ms_mz","ms_s","me"] are considered.
                Defaults to None.
            save_fig (bool, optional): Wether to save plots to png.
                Defaults to True.
            path (str, optional): Path to folder in which plots shall
                be saved. Defaults to ".".
        Raises:
            ValueError: If one of the variables in var_names has to many dimensions.
        """

        if var_names is None:
            # because list as default values are dangerous
            var_names = ["i_t","i_s","alpha","ms_mz","ms_s","me"]
        is_data_point_assoc = list(filter(lambda v: v in ["obs","mu"] or getattr(self,v).eval().shape[0]>1, var_names))
        if len(is_data_point_assoc)>0:
            raise ValueError(f"Following variables can not be plotted, due to high dimensionality:{is_data_point_assoc}")

        # test if posterior and prior was sampled
        if "posterior" not in self.idata.groups():
            self._sample()
        if "prior" not in self.idata.groups():
            self._sample_predictive(is_prior=True)

        for idx,feature in enumerate(self.feature_ids):
            feature_path = path+f"/feature{feature}/"
            if not os.path.exists(feature_path):
                os.makedirs(feature_path)
            idata_sliced = self.idata.isel(feature=idx,
                                           data_point=0,
                                           isotopic_peak=0)
            az.plot_posterior(idata_sliced,var_names)
            if save_fig:
                plt.savefig(feature_path+"posterior.png")
                plt.close()
            else:
                plt.show()

            az.plot_trace(idata_sliced,var_names)
            if save_fig:
                plt.savefig(feature_path+"trace.png")
                plt.close()
            else:
                plt.show()

            az.plot_pair(idata_sliced,var_names=var_names)
            if save_fig:
                plt.savefig(feature_path+"pairs.png")
                plt.close()
            else:
                plt.show()

            az.plot_energy(idata_sliced)
            if save_fig:
                plt.savefig(feature_path+"energy.png")
                plt.close()
            else:
                plt.show()

            az.plot_density(idata_sliced,group="prior",var_names=var_names)
            if save_fig:
                plt.savefig(feature_path+"prior.png")
                plt.close()
            else:
                plt.show()

    def evaluation(self,
                   prior_pred_in:bool = False,
                   posterior_pred_in:bool = True,
                   prior_pred_out:bool = False,
                   posterior_pred_out:bool = False,
                   plots:Optional[List[str]] = None,
                   reset_idata:bool = True,
                   progressbar:bool = True,
                   pred_name_list:Optional[List[str]] = None,
                   **plot_kwargs
                   ) -> az.InferenceData:
        """Evaluate precursor feature model.

        This function is wrapping several steps
        such as sampling of the model ,predictive
        analyses and plotting.

        Args:
            prior_pred_in (bool, optional): Wether to perform prior
                predictive check (in-sample). Defaults to False.
            posterior_pred_in (bool, optional): Wether to perform posterior
                predictive check (in-sample). Defaults to True.
            prior_pred_out (bool, optional): Wether to perform prior
                predictive check (out-of-sample). Defaults to False.
            posterior_pred_out (bool, optional): Wether to perform posterior
                predictive check (out-of-sample). Defaults to False.
            plots (Optional[List[str]],optional): List of plots to generate.
                Possible entries :  'prior_pred_in','prior_pred_out',
                'posterior_pred_in','posterior_pred_out'.
                Defaults to None.
            reset_idata (bool, optional): Wether to reset
                inferenceData. Defaults to True.
            progressbar (bool,optional): Wether to plot progressbar.
                Defaults to True.
            pred_name_list (Optional[List[str]],optional): Which predicted
                variables to plot. If None, 'obs' is plotted. Defaults to None.
            **plot_kwargs: Keyword Arguments passed to
                `visualize_predictions_scatter` method.
        Returns:
            az.InferenceData: Inference data of model.
        """

        if reset_idata:
            self.idata = az.InferenceData()
        self._sample(progressbar=progressbar)

        if pred_name_list is None:
            pred_name_list = ["obs"]

        if plots is None:
            # make 'in' operator available for plots
            plots = []

        if prior_pred_in:
            # prior predictive analysis in-sample
            self._sample_predictive(is_prior=True)
            if "prior_pred_in" in plots:
                for pred_name in pred_name_list:
                    plot_kwargs["pred_name"] = pred_name
                    self.visualize_predictions_scatter(is_prior=True,
                                                   **plot_kwargs)

        if prior_pred_out:
            # prior predictive analysis out-of-sample
            self._sample_predictive(is_prior=True,
                                    is_grid_predictive=True)
            if "prior_pred_out" in plots:
                for pred_name in pred_name_list:
                    plot_kwargs["pred_name"] = pred_name
                    self.visualize_predictions_scatter(in_sample=False,
                                                   is_prior=True,
                                                   **plot_kwargs)

        if posterior_pred_in:
            # posterior predictive analysis in-sample
            self._sample_predictive(progressbar=progressbar)
            if "posterior_pred_in" in plots:
                for pred_name in pred_name_list:
                    plot_kwargs["pred_name"] = pred_name
                    self.visualize_predictions_scatter(**plot_kwargs)

        if posterior_pred_out:
            # posterior predictive analysis out-of-sample
            self._sample_predictive(is_grid_predictive=True,
                                    progressbar=progressbar)
            if "posterior_pred_out" in plots:
                for pred_name in pred_name_list:
                    plot_kwargs["pred_name"] = pred_name
                    self.visualize_predictions_scatter(in_sample=False,
                                                   **plot_kwargs)

        return self.idata.copy()


class ModelGLM3D(AbstractModel):
    """Simple GLM like model of precursor feature

    GLM model fitting the 2D function

    .. math::

        f(mz,scan) = α*f_{NM}(mz)*f_{N}(scan)

    With :math:`f_{NM}(mz)` being the pdf of a normal
    mixture distribution and :math:`f_{N}(scan)` being
    the pdf of a normal distribution.

    Args:
        feature_ids (List[int]): List of feature ids in batch.
        z (int): Charge of precursor.
        intensity (NDArrayFloat): Observed intensities
        scan (NDArrayFloat): Observed scan numbers.
        mz (NDArrayFloat): Observed mass to charge ratios.
        peaks (NDArrayInt): Isotopic peaks to consider as numpy nd array.
        ims_mu (float): Hyperprior. Expected value for scan number.
        ims_sigma_max (float): Hyperprior. Maximal expected standard deviation
            for scan number.
        mz_mu (float): Hyperprior. Expected value for monoisotopic
            mass to charge ratio.
        mz_sigma (float): Hyperprior. Standard deviation of prior normal
            distribution for monoisotopic peak
        alpha_lam (float): Expected value for scalar.
        me_sigma (float): Sigma for model error.
        likelihood (str, optional): Likelihood distribution. Currently
            supported: 'Normal', 'StudentT'. Defaults to 'Normal'.
        name (str,optional): Defaults to empty string.
        coords (Optional[dict[str,ArrayLike]],optional):
            Coordinates for dims of model.
    Raises:
        NotImplementedError if provided likelihood is not supported.

    """

    def __init__(self,
                 z:int,
                 feature_ids:List[int],
                 intensity:NDArrayFloat,
                 scan:NDArrayFloat,
                 mz:NDArrayFloat,
                 peaks:NDArrayInt,
                 ims_mu:float,
                 ims_sigma_max:float,
                 mz_mu:float,
                 mz_sigma:float,
                 alpha_lam:float,
                 me_sigma:float,
                 likelihood:str = "Normal",
                 name:str="",
                 coords:Optional[dict[str,ArrayLike]]=None) -> None:

        batch_size = len(feature_ids)
        super().__init__(feature_ids,batch_size,name,coords=coords)
        # accessible from outside (data and hyperpriors)
        dims_2d = ["data_point","feature"]
        dims_3d = ["data_point","feature","isotopic_peak"]
        self.intensity = pm.MutableData("intensity",intensity,
                                        broadcastable=(False,False),
                                        dims=dims_2d)

        self.ims_mu = pm.MutableData("ims_mu",ims_mu,
                                     broadcastable=(True,False),
                                        dims=dims_2d)
        self.ims_sigma_max = pm.MutableData("ims_sigma_max",ims_sigma_max,
                                            broadcastable=(True,False),
                                            dims=dims_2d)
        self.scan = pm.MutableData("scan",scan,
                                   broadcastable=(False,False),
                                   dims=dims_2d)

        self.alpha_lam = pm.MutableData("alpha_lam",alpha_lam,
                                        broadcastable=(True,False),
                                        dims=dims_2d)
        self.me_sigma = pm.MutableData("me_sigma",me_sigma,
                                        broadcastable=(True,False),
                                        dims=dims_2d)

        self.charge = pm.MutableData("charge",z,
                                     broadcastable=(True,False,True),
                                     dims=dims_3d)
        self.mz = pm.MutableData("mz",mz,
                                 broadcastable=(False,False,False),
                                 dims=dims_3d)
        self.peaks = pm.MutableData("peaks",peaks,
                                    broadcastable=(False,False,False),
                                    dims=dims_3d)
        self.mz_mu = pm.MutableData("mz_mu",mz_mu,
                                    broadcastable=(True,False,True),
                                    dims=dims_3d)
        self.mz_sigma = pm.MutableData("mz_sigma",mz_sigma,
                                       broadcastable=(True,False,True),
                                       dims=dims_3d)


        # priors
        # IMS
        self.i_t = pm.Normal("i_t",
                             mu=self.ims_mu,
                             sigma=self.ims_sigma_max/2,
                             dims=dims_2d)
        self.i_s = pm.Uniform("i_s",
                              lower=0,
                              upper=self.ims_sigma_max,
                              dims=dims_2d)

        # mass spec
        self.ms_mz = pm.Normal("ms_mz",
                               mu=self.mz_mu,
                               sigma=self.mz_sigma,
                               dims=dims_3d)
        # TODO(Tim) separate mz_sigma
        self.ms_s = pm.Exponential("ms_s",
                                   lam=self.mz_sigma,
                                   dims=dims_3d)
        self.pos = self.peaks/(self.charge)+self.ms_mz
        self.lam = 0.000594 * (self.charge)*self.ms_mz - 0.03091
        self.ws_matrix = self.lam**self.peaks/ \
                         at.gamma(self.peaks+1)* \
                         pmath.exp(-self.lam)

        # scalar α
        self.alpha = pm.Exponential("alpha",
                                    lam = self.alpha_lam,
                                    dims=dims_2d)
        # α*f_IMS(t)
        self.pi1 = self.alpha\
                   *pmath.exp(-(self.i_t-self.scan)**2/(2*self.i_s**2))
        # f_mass(mz)
        self.pi2 = pmath.sum(self.ws_matrix\
                             *pmath.exp(-(self.pos-self.mz)**2\
                                        /(2*self.ms_s**2))
                             ,axis=2)

        # f(t,mz) = α*f_IMS(t)*f_mass(MZ)
        self.pi = pm.Deterministic("mu",
                                   var=self.pi1*self.pi2,
                                   auto=True,
                                   dims=dims_2d)
        # debug deterministic:
        # self.pi = self.pi1*self.pi2
        # Model error
        self.me = pm.HalfNormal("me",
                                sigma=self.me_sigma,
                                dims=dims_2d)
        # Likelihood
        if likelihood == "Normal":
            self.obs = pm.Normal("obs",
                             mu=self.pi,
                             sigma=self.me,
                             observed=self.intensity,
                             dims=dims_2d)
        elif likelihood == "StudentT":
            self.obs = pm.StudentT("obs",
                                   nu=5,
                                   mu=self.pi,
                                   sigma=self.me,
                                   observed=self.intensity,
                                   dims=dims_2d)
        else:
            raise NotImplementedError("This likelihood is not supported")

        self._initialize_idata()