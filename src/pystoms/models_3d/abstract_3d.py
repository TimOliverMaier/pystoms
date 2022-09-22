"""Abstract super class for 3D models of IMS-MS features
"""

from logging import warning
from typing import List, Optional, Tuple
import pandas as pd
import pymc as pm
import pymc.math as pmath
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import plotly.express as px
import arviz as az
from scipy.special import factorial
from aesara import tensor as at
from aesara.tensor.sharedvar import TensorSharedVariable
import matplotlib.pyplot as plt
import os
from numpy.typing import ArrayLike
from time import time
# typing
NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int64]


class AbstractModel(pm.Model):
    """Abstract model class for model evaluation.

    This class provides the 2D GLM model subclasses with
    evaluation methods.

    Args:
        feature_ids(List[int]): List of feature ids in batch.
        batch_size (int): Number of features
            in one batch.
        name (str, optional): Name of model. Defaults to empty string.
        coords (dict[str,ArrayLike],optional):
            Coordinates for dims in model.
        random_number_generator (np.random.Generator,optional): Random number generator.
            Defaults to None.
    Attributes:
        feature_ids(List[int]): List of feature ids in batch.
        batch_size (int): Number of features in
            one batch.
        idata (az.InferenceData):
          inferenceData of current model.
        rng (Optional[np.random.Generator]): Random Number generator

    """

    def __init__(
        self,
        feature_ids: List[int],
        batch_size: int,
        random_number_generator: np.random.Generator = None,
        name: Optional[str] = "",
        coords: Optional[dict[str, ArrayLike]] = None,
    ) -> None:
        # name and model must be passed to pm.Model
        coords = {}
        coords.setdefault("feature", feature_ids)
        super().__init__(name, coords=coords)
        # instantiate inference data
        self.feature_ids = feature_ids
        self.batch_size = batch_size
        self.idata = az.InferenceData()
        self.rng = random_number_generator

    def _reset_idata(self) -> None:
        self.idata = az.InferenceData()

    def _initialize_idata(self):
        """Inits InferenceData with constant_data group"""
        prior_idata = pm.sample_prior_predictive(5, model=self, random_seed=self.rng)
        self.idata.add_groups({"constant_data": prior_idata.constant_data})

    def _get_model_shared_data(
        self, predictions_constant_data: bool = True
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
        for key, var in self.named_vars.items():
            if isinstance(var, TensorSharedVariable):
                data[key] = var.get_value()
        if predictions_constant_data:
            idata = az.from_dict(predictions_constant_data=data)
        else:
            idata = az.from_dict(constant_data=data)
        return idata

    def _sample_predictive(
        self, is_prior: bool = False, is_grid_predictive: bool = False, **kwargs
    ) -> None:
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

        # set random number generator in kwargs if not user specified
        kwargs.setdefault("random_seed", self.rng)
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
                prior_prediction = pm.sample_prior_predictive(model=self, **kwargs)
                if "mu" in prior_prediction.prior.data_vars.keys():
                    prior_prediction.prior_predictive["mu"] = prior_prediction.prior[
                        "mu"
                    ]
                warning(
                    "Prior out-of-sample predictions are currently \
                    a work around.\n Arviz will warn about non-defined \
                    InferenceData group 'prior_predictions'."
                )
                self.idata.add_groups(
                    {
                        "prior_predictions": prior_prediction.prior_predictive,
                        "predictions_constant_data": prior_prediction.constant_data,
                    }
                )

            else:
                warning(
                    "Inference data of model already\n\
                         has prior_predictions"
                )

            # else: do nothing, posterior out-of-sample predictions
            # were sampled before, providing predictions_constant_data

            self._set_observed_data()

        if is_prior and not is_grid_predictive:
            # prior predictions on coordinates of observed data
            # used for posterior sampling (in-sample predictions)
            if "prior" not in self.idata.groups():
                prior_idata = pm.sample_prior_predictive(**kwargs, model=self)
                if "mu" in prior_idata.prior.data_vars.keys():
                    prior_idata.prior_predictive["mu"] = prior_idata.prior["mu"]
                self.idata.extend(prior_idata)
            else:
                warning(
                    "Inference data of model already\n\
                        has prior_predictive"
                )

        if not is_prior and is_grid_predictive:
            # posterior out of-sample-predictions with grid
            # as predictors
            if "predictions" not in self.idata.groups():
                self._set_grid_data()
                prediction = pm.sample_posterior_predictive(
                    self.idata,
                    model=self,
                    # extend_inferencedata=True,
                    predictions=True,
                    **kwargs,
                )
                # until fix inference data must be updated
                self.idata.extend(prediction)
                self._set_observed_data()
            else:
                warning(
                    "Inference data of model already\n\
                        has predictions"
                )

        if not is_prior and not is_grid_predictive:
            # posterior predictions on coordinates of observed data
            # used for posterior sampling (in-sample predictions)
            if "posterior_predictive" not in self.idata.groups():
                pm.sample_posterior_predictive(
                    self.idata, extend_inferencedata=True, model=self, **kwargs
                )
            else:
                warning(
                    "Inference data of model already\n\
                        has posterior_predictive"
                )

    def visualize_predictions_scatter(
        self,
        size: int = 50,
        in_sample: bool = True,
        is_prior: bool = False,
        pred_name: str = "obs",
        plot_observed_data: bool = True,
        write_to_file: bool = False,
        folder_path: str = ".",
        file_name: Optional[str] = None,
        use_renderer: str = "notebook",
    ) -> None:
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
                draw_sample = np.random.choice(
                    self.idata.prior_predictions.draw.values, size
                )
                predicted_data = self.idata.prior_predictions[{"draw": draw_sample}]
            if not is_prior:
                # posterior predictions on grid data
                draw_sample = np.random.choice(self.idata.predictions.draw.values, size)
                predicted_data = self.idata.predictions[{"draw": draw_sample}]

        if in_sample:
            # original observed data as predictors
            predictors_data = self.idata.constant_data

            if is_prior:
                # prior predictions on observed data
                draw_sample = np.random.choice(
                    self.idata.prior_predictive.draw.values, size
                )
                predicted_data = self.idata.prior_predictive[{"draw": draw_sample}]
            if not is_prior:
                # posterior predictions on observed data
                draw_sample = np.random.choice(
                    self.idata.posterior_predictive.draw.values, size
                )
                predicted_data = self.idata.posterior_predictive[{"draw": draw_sample}]

        # get mz and scan values (predictors)
        # first extract variable from xarray dataset
        # because direct to_dataframe crashed
        df_mz = predictors_data.mz.to_dataframe().xs(0, level="isotopic_peak")

        df_scan = predictors_data.scan.to_dataframe()
        # then merge to one df
        df_predictors = pd.merge(df_mz, df_scan, left_index=True, right_index=True)
        # get corresponding predicted values
        # test if desired pred_name was sampled
        if pred_name not in predicted_data.data_vars.keys():
            # use first data variable in Dataset otherwise
            pred_name_new = list(predicted_data.data_vars.keys())[0]
            warning(
                f"{pred_name} not in 'prior_predictive',\
                    using {pred_name_new}"
            )
            pred_name = pred_name_new

        df_predicted = getattr(predicted_data, pred_name).to_dataframe()
        # merge to dataframe with predictors and predicted vars
        df_plot = (
            pd.merge(df_predictors, df_predicted, left_index=True, right_index=True)
            .reset_index()
            .astype({"chain": "str"})
        )
        # plotting
        for feature in self.feature_ids:
            fig = px.scatter_3d(
                data_frame=df_plot[df_plot.feature == feature],
                x="scan",
                y="mz",
                z=pred_name,
                color="chain",
                opacity=0.1,
            )
            if plot_observed_data:
                obs_data_trace = self.plot_feature_data(
                    return_fig_trace=True, feature_ids=[feature]
                )[0]
                fig.add_trace(obs_data_trace)

            if write_to_file:
                feature_path = folder_path + f"/feature{feature}/"
                if file_name is None:
                    fn_1 = "prior_predictive" if is_prior else "posterior_predictive"
                    fn_2 = "in_sample" if in_sample else "out_of_sample"
                    file_name = fn_1 + "_" + fn_2 + "_" + pred_name + ".html"

                if not file_name.endswith(".html"):
                    fn_prefix = file_name.split(".")[0]
                    file_name = fn_prefix + ".html"

                path_to_file = feature_path + file_name

                if not os.path.exists(feature_path):
                    os.makedirs(feature_path)
                fig.write_html(path_to_file)
            else:
                fig.show(renderer=use_renderer)

    def plot_feature_data(
        self,
        return_fig_trace: bool = False,
        feature_ids: Optional[List[int]] = None,
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
            y = data.mz.values[:, 0]
            z = data.intensity.values
            fig_trace = go.Scatter3d(
                x=x, y=y, z=z, mode="markers", marker=dict(color="black", size=10)
            )
            scatter_traces.append(fig_trace)
            if not return_fig_trace:
                fig = go.Figure(data=[fig_trace])
                fig.show()
        if return_fig_trace:
            return scatter_traces

    def _sample(self, **kwargs) -> az.InferenceData:
        """Method to call PyMC sampler.

        Args:
            **kwargs: Keyword arguments for
                PyMC.sample()
        Returns:
            az.InferenceData: Sample trace as InferenceData.
        """
        # ? I think this is not necessary anymore in pymc 4
        kwargs.setdefault("return_inferencedata", True)

        kwargs.setdefault("random_seed", self.rng)
        trace = pm.sample(**kwargs, model=self)
        self.idata.extend(trace)
        return trace

    def arviz_plots(
        self,
        var_names: Optional[List[str]] = None,
        save_fig: bool = True,
        path: str = ".",
    ) -> None:
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
            var_names = ["i_t", "i_s", "alpha", "ms_mz", "ms_s", "me"]
        is_data_point_assoc = list(
            filter(
                lambda v: v in ["obs", "mu"] or getattr(self, v).eval().shape[0] > 1,
                var_names,
            )
        )
        if len(is_data_point_assoc) > 0:
            raise ValueError(
                f"Following variables can not be plotted, due to high dimensionality:{is_data_point_assoc}"
            )

        # test if posterior and prior was sampled
        if "posterior" not in self.idata.groups():
            self._sample()
        if "prior" not in self.idata.groups():
            self._sample_predictive(is_prior=True)

        for idx, feature in enumerate(self.feature_ids):
            feature_path = path + f"/feature{feature}/"
            if not os.path.exists(feature_path):
                os.makedirs(feature_path)
            idata_sliced = self.idata.isel(feature=idx, data_point=0, isotopic_peak=0)
            az.plot_posterior(idata_sliced, var_names)
            if save_fig:
                plt.savefig(feature_path + "posterior.png")
                plt.close()
            else:
                plt.show()

            az.plot_trace(idata_sliced, var_names, legend=True, chain_prop={"linestyle": ("solid", "dotted", "dashed", "dashdot"),"color":("black","blue","green","red")})
            if save_fig:
                plt.tight_layout()
                plt.savefig(feature_path + "trace.png")
                plt.close()
            else:
                plt.tight_layout()
                plt.show()

            az.plot_pair(idata_sliced, var_names=var_names)
            if save_fig:
                plt.savefig(feature_path + "pairs.png")
                plt.close()
            else:
                plt.show()

            az.plot_energy(idata_sliced)
            if save_fig:
                plt.savefig(feature_path + "energy.png")
                plt.close()
            else:
                plt.show()

            az.plot_density(idata_sliced, group="prior", var_names=var_names)
            if save_fig:
                plt.savefig(feature_path + "prior.png")
                plt.close()
            else:
                plt.show()

            # posterior predictive lm
            idata_feature_sliced = self.idata.isel(feature=idx)
            az.plot_lm(y=idata_feature_sliced.observed_data.obs,
                       y_hat=idata_feature_sliced.posterior_predictive.obs,
                       num_samples=500)
            if save_fig:
                plt.savefig(feature_path + "posterior_predictive_lm.png")
                plt.close()
            else:
                plt.show()

            # prior predictive lm
            fig,ax = plt.subplots(1,1)
            az.plot_lm(y=idata_feature_sliced.observed_data.obs,
                       y_hat=idata_feature_sliced.prior_predictive.obs,
                       num_samples=500,
                       legend=False,
                       axes=ax)
            h,l = ax.get_legend_handles_labels()
            for idx,lab in enumerate(l):
                if lab == "Posterior predictive samples":
                    l[idx] = "Prior predictive samples"
            ax.legend(h,l)
            if save_fig:
                fig.savefig(feature_path + "prior_predictive_lm.png")
                plt.close()
            else:
                plt.show()

            az.plot_parallel(idata_sliced,norm_method="minmax")
            if save_fig:
                plt.savefig(feature_path + "plot_parallel.png")
                plt.close()
            else:
                plt.show()
            


    def evaluation(
        self,
        prior_pred_in: bool = True,
        posterior_pred_in: bool = True,
        prior_pred_out: bool = False,
        posterior_pred_out: bool = False,
        plots: Optional[List[str]] = None,
        reset_idata: bool = True,
        progressbar: bool = True,
        pred_name_list: Optional[List[str]] = None,
        **plot_kwargs,
    ) -> Tuple[az.InferenceData,float]:
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
            Tuple(az.InferenceData, float): Inference data of model and
                sampling time
        """

        if reset_idata:
            self.idata = az.InferenceData()
        start_sampling = time()
        self._sample(progressbar=progressbar)
        end_sampling = time()

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
                    self.visualize_predictions_scatter(is_prior=True, **plot_kwargs)

        if prior_pred_out:
            # prior predictive analysis out-of-sample
            self._sample_predictive(is_prior=True, is_grid_predictive=True)
            if "prior_pred_out" in plots:
                for pred_name in pred_name_list:
                    plot_kwargs["pred_name"] = pred_name
                    self.visualize_predictions_scatter(
                        in_sample=False, is_prior=True, **plot_kwargs
                    )

        if posterior_pred_in:
            # posterior predictive analysis in-sample
            self._sample_predictive(progressbar=progressbar)
            if "posterior_pred_in" in plots:
                for pred_name in pred_name_list:
                    plot_kwargs["pred_name"] = pred_name
                    self.visualize_predictions_scatter(**plot_kwargs)

        if posterior_pred_out:
            # posterior predictive analysis out-of-sample
            self._sample_predictive(is_grid_predictive=True, progressbar=progressbar)
            if "posterior_pred_out" in plots:
                for pred_name in pred_name_list:
                    plot_kwargs["pred_name"] = pred_name
                    self.visualize_predictions_scatter(in_sample=False, **plot_kwargs)

        return (self.idata.copy(),end_sampling-start_sampling)
