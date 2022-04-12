"""Probabilistic General linear models for LC-IMS-MS precursor features



"""

from ast import Raise
from logging import warning
from typing import List
import pandas as pd
import pymc as pm
import pymc.math as pmath
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
import arviz as az
from aesara import tensor as at
from aesara.tensor.sharedvar import TensorSharedVariable
import matplotlib.pyplot as plt
import os

#typing
NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt   = npt.NDArray[np.int64]


class AbstractModel(pm.Model):
    """Abstract model class for model evaluation.

    This class provides the 2D GLM model subclasses with
    evaluation methods.

    Args:
        name (str): Name of model.
        model (Optional[pm.Model]): PyMC model

    Attributes:
        idata (az.InferenceData):
          inferenceData of current model.
    """
    def __init__(self,name:str,model:Optional[pm.Model]) -> None:
        # name and model must be passed to pm.Model
        super().__init__(name,model)
        # instantiate inference data
        self.idata = az.InferenceData()

    def _reset_idata(self) -> None:
        self.idata = az.InferenceData()

    def _get_model_shared_data(self,
                               predictions_constant_data:bool = True
                              ) -> az.InferenceData:
        """Get model's input data from model as idata.

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
                "prior_predictions":prior_prediction.prior_predictive
                })
            else:
                warning("Inference data of model already\n\
                         has prior_predictions")
            if "predictions_constant_data" not in self.idata.groups():
                predictions_data = self._get_model_shared_data()
                self.idata.extend(predictions_data)
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
                                      path:str = "",
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
            path (str, optional): Path to folder in which output files
                are stored. Defaults to "".
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
                    .xs(0,level="mz_dim_1")\
                    .reset_index()
        df_scan = predictors_data.scan\
                    .to_dataframe()\
                    .reset_index()
        # then merge to one df
        df_predictors = pd.merge(df_mz,
                                df_scan,
                                left_on="mz_dim_0",
                                right_on="scan_dim_0")\
                                .drop(columns=["scan_dim_0"])\
                                .rename(columns={
                                        "mz_dim_0":"data_point"
                                        })
        # get corresponding predicted values
        # test if desired pred_name was sampled
        if pred_name not in predicted_data.data_vars.keys():
            # use first data variable in Dataset otherwise
            pred_name_new = list(predicted_data.data_vars.keys())[0]
            warning(f"{pred_name} not in 'prior_predictive',\
                    using {pred_name_new}")
            pred_name = pred_name_new
        pred_name_dim_0 = pred_name + "_dim_0"
        df_predicted = getattr( predicted_data,pred_name)\
                                .to_dataframe()\
                                .reset_index()\
                                .rename(columns={
                                    pred_name_dim_0:"data_point"
                                    })
        # merge to dataframe with predictors and predicted vars
        df_plot = pd.merge(
                          df_predictors,
                          df_predicted,
                          on = "data_point"
                          ).astype({"chain":"str"})
        # plotting
        fig = px.scatter_3d(data_frame=df_plot,
                            x="scan",
                            y="mz",
                            z=pred_name,
                            color="chain",
                            opacity=0.1)
        if plot_observed_data:
            obs_data_trace = self.plot_feature_data(return_fig_trace=True)
            fig.add_trace(obs_data_trace)

        if write_to_file:
            if file_name is None:
                fn_1 = "prior_predictive" \
                        if is_prior \
                        else "posterior_predictive"
                fn_2 = "in_sample" if in_sample else "out_of_sample"
                file_name = fn_1 + "_" + fn_2 + "_" + pred_name + ".html"

            if not file_name.endswith(".html"):
                fn_prefix = file_name.split(".")[0]
                file_name = fn_prefix + ".html"

            path_to_file = path+"/"+file_name

            if not os.path.exists(path):
                os.makedirs(path)
            fig.write_html(path_to_file)
        else:
            fig.show(renderer=use_renderer)

    def plot_feature_data(self,
                          return_fig_trace:bool = False
                          ) -> Optional[go.Scatter3d]:
        """plots model's input feature data.

        Args:
            return_fig_trace (bool, optional): Wether to only return
                plotly 3D scatter trace. Defaults to False.
        Returns:
            Optional[go.Scatter3d]: If `return_fig_trace` is True,
                then plotly Scatter3d trace with observed data is
                returned.
        """
        data = self.idata.constant_data
        x = data.scan.values
        y = data.mz.values[:,0]
        z = data.intensity.values
        fig_trace = go.Scatter3d(x=x,
                                 y=y,
                                 z=z,
                                 mode="markers",
                                 marker=dict(color="black",size=10))
        if return_fig_trace:
            return fig_trace
        else:
            fig = go.Figure(data=[fig_trace])
            fig.show()


    def _set_grid_data(self) -> None:
        """Set model's pm.MutableData container to grid data

        Used for prediction on a grid.
        """
        if "constant_data" not in self.idata:
            # get data directly from model instead
            data = self._get_model_shared_data(False).constant_data
        else:
            data = self.idata.constant_data
        # calculate hull boundaries of feature
        obs_x = data.mz.values[:,0].flatten()
        obs_y = data.scan.values
        peak_num = int(data.peak_num.values[0])

        # set axis limits accordingly
        xmin = obs_x.min()-1
        xmax = obs_x.max()+1
        ymin = obs_y.min()-1
        ymax = obs_y.max()+1
        # x axis and y axis , scan intervall is 1, mz 0.01

        y = np.arange(ymin,ymax,dtype=int)
        x = np.arange(xmin*100,xmax*100)/100
        x,y = np.meshgrid(x,y)
        x = np.tile(x.flatten(),(int(peak_num),1)).T
        y = y.flatten()

        # get rest of data, necessary to reset
        # these as well to run properly
        charge = data.charge.values[0]
        ims_mu = data.ims_mu.values[0]
        ims_sigma_max = data.ims_sigma_max.values[0]
        mz_mu = data.mz_mu.values[0]
        mz_sigma = data.mz_sigma.values[0]
        alpha_lam = data.alpha_lam.values[0]

        pm.set_data({"scan":y,
                      "mz":x,
                      "intensity":np.zeros_like(y,dtype="float"),
                      "charge":charge,
                      "peak_num":peak_num,
                      "peaks":np.tile(np.arange(peak_num),(y.size,1)),
                      "ims_mu":ims_mu,
                      "ims_sigma_max":ims_sigma_max,
                      "mz_mu":mz_mu,
                      "mz_sigma":mz_sigma,
                      "alpha_lam":alpha_lam,
                    },
                    model=self)

    def _set_observed_data(self) -> None:
        """Set model's pm.MutableData container (back) to observed data
        """
        if "constant_data" not in self.idata:
            # get data directly from model instead
            data = self._get_model_shared_data(False).constant_data
        else:
            data = self.idata.constant_data
        # extract original observed values
        obs_x = data.mz.values
        obs_y = data.scan.values
        obs_z = data.intensity.values
        charge = data.charge.values[0]
        peak_num = int(data.peak_num.values[0])
        ims_mu = data.ims_mu.values[0]
        ims_sigma_max = data.ims_sigma_max.values[0]
        mz_mu = data.mz_mu.values[0]
        mz_sigma = data.mz_sigma.values[0]
        alpha_lam = data.alpha_lam.values[0]
        pm.set_data({"scan":obs_y,
                      "mz":obs_x, # awful but mzs currently needed as column
                      "intensity":obs_z,
                      "charge":charge,
                      "peak_num":peak_num,
                      "peaks":np.tile(np.arange(peak_num),(obs_y.size,1)),
                      "ims_mu":ims_mu,
                      "ims_sigma_max":ims_sigma_max,
                      "mz_mu":mz_mu,
                      "mz_sigma":mz_sigma,
                      "alpha_lam":alpha_lam,
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
                    path:str = "") -> None:
        """Generate various arviz plots

        Args:
            var_names (Optional[List[str]], optional): Which variables
                to consider. If None, then
                ["i_t","i_s","alpha","ms_mz","ms_s","me"] are considered.
                Defaults to None.
            save_fig (bool, optional): Wether to save plots to png.
                Defaults to True.
            path (str, optional): Path to folder in which plots shall
                be saved. Defaults to ""
        """

        if var_names is None:
            # because list as default values are dangerous
            var_names = ["i_t","i_s","alpha","ms_mz","ms_s","me"]

        # test if posterior and prior was sampled
        if "posterior" not in self.idata.groups():
            self._sample()
        if "prior" not in self.idata.groups():
            self._sample_predictive(is_prior=True)

        if not os.path.exists(path):
            os.makedirs(path)

        az.plot_posterior(self.idata,var_names)
        if save_fig:
            plt.savefig(path+"/"+"posterior.png")
            plt.close()
        else:
            plt.show()

        az.plot_trace(self.idata,var_names)
        if save_fig:
            plt.savefig(path+"/"+"trace.png")
            plt.close()
        else:
            plt.show()

        az.plot_pair(self.idata,var_names=var_names)
        if save_fig:
            plt.savefig(path+"/"+"pairs.png")
            plt.close()
        else:
            plt.show()

        az.plot_energy(self.idata)
        if save_fig:
            plt.savefig(path+"/"+"energy.png")
            plt.close()
        else:
            plt.show()

        az.plot_density(self.idata,group="prior",var_names=var_names)
        if save_fig:
            plt.savefig(path+"/"+"prior.png")
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
        num_observed (int): Number of observed datapoints.
        num_features (int): Number of features in passed data.
        peak_num (int): Number of isotopic (incl MI) peaks
            to consider.
        z (int): Charge of precursor.
        intensity (NDArrayFloat): Observed intensities
        scan (NDArrayInt): Observed scan numbers.
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
        likelihood (str, optional): Likelihood distribution. Currently
            supported: 'Normal', 'StudentT'. Defaults to 'Normal'.
        name (str,optional): Defaults to empty string.
        model (Optional[pm.Model],optional): Defaults to None.
    Raises:
        NotImplementedError if provided likelihood is not suppored.

    """

    def __init__(self,
                 num_observed:int,
                 num_features:int,
                 peak_num:int,
                 z:int,
                 intensity:NDArrayFloat,
                 scan:NDArrayInt,
                 mz:NDArrayFloat,
                 peaks:NDArrayInt,
                 ims_mu:float,
                 ims_sigma_max:float,
                 mz_mu:float,
                 mz_sigma:float,
                 alpha_lam:float,
                 likelihood:str = "Normal",
                 name:str="",
                 model:pm.Model = None) -> None:

        super().__init__(name,model)
        # accessible from outside (data and hyperpriors)
        self.num_observed = pm.MutableData("num_observed",num_observed)
        self.num_features = pm.MutableData("num_features",num_features)
        self.peak_num = pm.MutableData("peak_num",peak_num,
                                        broadcastable=(True,False,True))
        self.charge = pm.MutableData("charge",z,
                                     broadcastable=(True,False,True))
        self.intensity = pm.MutableData("intensity",intensity,
                                        broadcastable=(False,False,True))
        self.scan = pm.MutableData("scan",scan,
                                   broadcastable=(False,False,True))
        self.mz = pm.MutableData("mz",mz,
                                 broadcastable=(False,False,False))
        self.peaks = pm.MutableData("peaks",peaks,
                                    broadcastable=(False,False,False))
        self.ims_mu = pm.MutableData("ims_mu",ims_mu,
                                     broadcastable=(True,False,True))
        self.ims_sigma_max = pm.MutableData("ims_sigma_max",ims_sigma_max,
                                            broadcastable=(True,False,True))
        self.mz_mu = pm.MutableData("mz_mu",mz_mu,
                                    broadcastable=(True,False,True))
        self.mz_sigma = pm.MutableData("mz_sigma",mz_sigma,
                                       broadcastable=(True,False,True))
        self.alpha_lam = pm.MutableData("alpha_lam",alpha_lam,
                                        broadcastable=(True,False,True))

        # priors
        # IMS
        self.i_t = pm.Normal("i_t",mu=self.ims_mu,sigma=self.ims_sigma_max/2)
        self.i_s = pm.Uniform("i_s",lower=0,upper=self.ims_sigma_max)

        # mass spec
        self.ms_mz = pm.Normal("ms_mz",mu=self.mz_mu,sigma=self.mz_sigma)
        # TODO(Tim) separate mz_sigma
        self.ms_s = pm.Exponential("ms_s",lam=self.mz_sigma)
        self.pos = self.peaks/(self.charge+1)+self.ms_mz
        self.lam = 0.000594 * (self.charge+1)*self.ms_mz - 0.03091
        self.ws_matrix = self.lam**self.peaks/ \
                         at.gamma(self.peaks+1)* \
                         pmath.exp(-self.lam)

        # scalar α
        self.alpha = pm.Exponential("alpha",lam = self.alpha_lam)
        # α*f_IMS(t)
        self.pi1 = self.alpha\
                   *pmath.exp(-(self.i_t-self.scan)**2/(2*self.i_s**2))
        # f_mass(mz)
        self.pi2 = pmath.sum(self.ws_matrix\
                             *pmath.exp(-(self.pos-self.mz)**2\
                                        /(2*self.ms_s**2))
                             ,axis=2).reshape((at.cast(self.num_observed,"int"),
                                               at.cast(self.num_features,"int"),
                                               1))

        # f(t,mz) = α*f_IMS(t)*f_mass(MZ)
        self.pi = pm.Deterministic("mu",var=self.pi1*self.pi2,auto=True)
        # debug deterministic:
        # self.pi = self.pi1*self.pi2
        # Model error
        self.me = pm.HalfNormal("me",sigma=10)
        # Likelihood
        if likelihood == "Normal":
            self.obs = pm.Normal("obs",
                             mu=self.pi,
                             sigma=self.me,
                             observed=self.intensity)
        elif likelihood == "StudentT":
            self.obs = pm.StudentT("obs",
                                   nu=5,
                                   mu=self.pi,
                                   sigma=self.me,
                                   observed=self.intensity)
        else:
            Raise(NotImplementedError("This likelihood is not supported"))
