"""Probabilistic General linear models for LC-IMS-MS precursor features



"""

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


    Attributes:
        idata (az.InferenceData):
          inferenceData of last model fit.
    """
    def __init__(self,name:str,model:Optional[pm.Model]):
        # name and model must be passed to pm.Model
        super().__init__(name,model)
        # instantiate inference data
        self.idata = az.InferenceData()

    def _reset_idata(self):
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
            **kwargs: Keyword arguments passed to predictive sampler.
        """

        # for overview on arviz inferenceData groups visit
        # https://arviz-devs.github.io/arviz/schema/schema.html
        # only prior can be sampled before posterior was sampled
        if "posterior" not in self.idata.groups() and not is_prior:
            self._sample()

        # prior out of sample predictions with grid
        # as predictors -> model surface
        if is_prior and is_grid_predictive:

            if "prior_predictions" not in self.idata.groups():
                self._set_grid_data()
                prior_prediction = pm.sample_prior_predictive(model=self,
                                                          **kwargs)
                warning("Prior out-of-sample predictions are currently \
                    a work around.\n Arviz will warn about non-defined \
                    InferenceData group 'prior_predictions'.")
                self.idata.add_groups({
                "prior_predictions":prior_prediction.prior_predictive
                })
            if "predictions_constant_data" not in self.idata.groups():
                predictions_data = self._get_model_shared_data()
                self.idata.extend(predictions_data)


            self._set_observed_data()
        # prior predictions on coordinates of observed data
        # used for posterior sampling (in sample predictions)
        if is_prior and not is_grid_predictive:
            if "prior" not in self.idata.groups():
                prior_idata = pm.sample_prior_predictive(**kwargs,model=self)
                self.idata.extend(prior_idata)


        # posterior out of sample predictions with grid
        # as predictors -> model surface
        if not is_prior and is_grid_predictive:
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
        # posterior predictions on coordinates of observed data
        # used for posterior sampling (in sample predictions)
        if not is_prior and not is_grid_predictive:
            if "posterior_predictive" not in self.idata.groups():
                pm.sample_posterior_predictive(self.idata,
                                                extend_inferencedata=True,
                                                model=self,
                                                **kwargs)

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

        Args:
            size (int,optional): Number of predictions to plot
              for each position.
            in_sample (bool, optional): If True in-sample
              predictions are plotted. Else out-of-sample
              predictions are plotted.
              Defaults to True.
            is_prior (bool, optional): If True prior
              predictions are plotted, else posterior.
              Defaults to False.
            pred_name (str,optional): predicted variable to plot.
              Defaults to 'obs'.
            plot_observed_data (bool,optional): Wether to plot observed
              data on top of scatter plot. Defaults to True.
            write_to_file (bool, optional): Wether to write plot to file.
                Defaults to False.
            path (str, optional): Path to folder in which output files
                are stored. Defaults to "".
            file_name (Optional[str], optional): Name of html output file.
                If is None, generic file name depending on `is_prior` and
                `in_sample` is used. Defaults to None.
            use_renderer (str,optional): Which plotly renderer to use.
              Defaults to 'notebook'.
        """
        # we want to plot predictors -> predicted
        # get correct data depeding on arguments

        # grid data
        if not in_sample:
            predictors_data = self.idata.predictions_constant_data
            # prior predictions on grid data (prior_predictions)
            if is_prior:
                draw_sample = np.random.choice(self.idata\
                                                .prior_predictions\
                                                .draw.values, size)
                predicted_data = self.idata.\
                                    prior_predictions[{"draw":draw_sample}]
            # posterior predictions on grid data (predictions)
            if not is_prior:
                draw_sample = np.random.choice(self.idata\
                                                .predictions\
                                                .draw\
                                                .values,
                                                size)
                predicted_data = self.idata.predictions[{"draw":draw_sample}]
        # original observed data
        if in_sample:
            predictors_data = self.idata.constant_data
            # prior predictions on observed data (prior_predictive)
            if is_prior:
                draw_sample = np.random.choice(self.idata\
                                                .prior_predictive\
                                                .draw\
                                                .values,
                                                size)
                predicted_data = self.idata\
                                    .prior_predictive[{"draw":draw_sample}]
            # posterior predictions on observed data (psoterior_predictive)
            if not is_prior:
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
            # use first data variable in DataArray otherwise
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
                file_name = fn_1 + "_" + fn_2 + ".html"

            if not file_name.endswith(".html"):
                fn_prefix = file_name.split(".")[0]
                file_name = fn_prefix + ".html"

            path_to_file = path+"/"+file_name

            if not os.path.exists(path):
                os.makedirs(path)
            fig.write_html(path_to_file)
        else:
            fig.show(renderer=use_renderer)

    def plot_feature_data(self,return_fig_trace:bool = False):
        """plots model's input feature data.

        Args:
            return_fig_trace (bool, optional): Wether to only return
                plotly 3D scatter trace. Defaults to False.
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

        Used for prediction on a grid
        """
        if "constant_data" not in self.idata:
            data = self._get_model_shared_data(False).constant_data
        else:
            data = self.idata.constant_data
        # extract hull boundaries of feature
        obs_x = data.mz.values.flatten()
        obs_y = data.scan.values
        charge = data.charge.values[0]
        peak_num = int(data.peak_num.values[0])
        ims_mu = data.ims_mu.values[0]
        ims_sigma = data.ims_sigma.values[0]
        mz_mu = data.mz_mu.values[0]
        mz_sigma = data.mz_sigma.values[0]
        alpha_lam = data.alpha_lam.values[0]
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

        pm.set_data({"scan":y,
                      "mz":x, # awful but mzs currently needed as column
                      "intensity":np.zeros_like(y,dtype="float"),
                      "charge":charge,
                      "peak_num":peak_num,
                      "peaks":np.tile(np.arange(peak_num),(y.size,1)),
                      "ims_mu":ims_mu,
                      "ims_sigma":ims_sigma,
                      "mz_mu":mz_mu,
                      "mz_sigma":mz_sigma,
                      "alpha_lam":alpha_lam,
                    },
                    model=self)

    def _set_observed_data(self) -> None:
        """Set model's pm.MutableData container to observed data
        """
        data = self.idata.constant_data
        # extract original observed values
        obs_x = data.mz.values
        obs_y = data.scan.values
        obs_z = data.intensity.values
        charge = data.charge.values[0]
        peak_num = int(data.peak_num.values[0])
        ims_mu = data.ims_mu.values[0]
        ims_sigma = data.ims_sigma.values[0]
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
                      "ims_sigma":ims_sigma,
                      "mz_mu":mz_mu,
                      "mz_sigma":mz_sigma,
                      "alpha_lam":alpha_lam,
                    },
                    model=self)

    def _sample(self,**kwargs):

        kwargs.setdefault("return_inferencedata", True)

        trace = pm.sample(**kwargs,model=self)

        self.idata.extend(trace)

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
        if "posterior" not in self.idata.groups():
            self._sample()
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


    def evaluation(self,
                   prior_pred_in:bool = False,
                   posterior_pred_in:bool = True,
                   prior_pred_out:bool = False,
                   posterior_pred_out:bool = False,
                   plots:Optional[List[str]] = None,
                   reset_idata:bool = True,
                   progressbar:bool = True,
                   **plot_kwargs
                   ) -> az.InferenceData:
        """Evaluate precursor feature model.

        This function is wrapping several steps
        such as sampling of the model ,predictive
        analyses an plotting.

        Args:
            prior_pred_in (bool, optional): Wether to perform prior
                predictive check (in-sample). Defaults to False.
            posterior_pred_in (bool, optional): Wether to perform posterior
                predictive check (in-sample). Defaults to True.
            prior_pred_out (bool, optional): Wether to perform prior
                predictive check (out-of-sample). Defaults to False.
            posterior_pred_out (bool, optional): Wether to perform psoterior
                predictive check (out-of-sample). Defaults to False.
            plots (Optional[List[str]],optional): List of plots to generate.
                Possible entries :  'prior_pred_in','prior_pred_out',
                'posterior_pred_in','posterior_pred_out'.
                Defaults to None.
            reset_idata (bool, optional): Wether to reset
                inferenceData. Defaults to True.
            progressbar (bool,optional): Wether to plot progressbar.
                Defaults to True.
            **plot_kwargs: Keyword Arguments passed to
                `visualize_predictions_scatter` method.
        Returns:
            [az.InferenceData]: Inference data of model.
        """

        if reset_idata:
            self.idata = az.InferenceData()
        self._sample(progressbar=progressbar)

        # make 'in' operator available for plots
        if plots is None:
            plots = []

        # prior predictive analysis in-sample
        if prior_pred_in:
            self._sample_predictive(is_prior=True)
            if "prior_pred_in" in plots:
                self.visualize_predictions_scatter(is_prior=True,
                                                   **plot_kwargs)

        # prior predictive analysis out-of-sample
        if prior_pred_out:
            self._sample_predictive(is_prior=True,
                                    is_grid_predictive=True)
            if "prior_pred_out" in plots:
                self.visualize_predictions_scatter(in_sample=False,
                                                   is_prior=True,
                                                   **plot_kwargs)

        # posterior predictive analysis in-sample
        if posterior_pred_in:
            self._sample_predictive(progressbar=progressbar)
            if "posterior_pred_in" in plots:
                self.visualize_predictions_scatter(**plot_kwargs)

        # posterior predictive analysis out-of-sample
        if posterior_pred_out:
            self._sample_predictive(is_grid_predictive=True,
                                    progressbar=progressbar)
            if "posterior_pred_out" in plots:
                self.visualize_predictions_scatter(in_sample=False,
                                                   **plot_kwargs)

        return self.idata.copy()


class ModelGLM3D(AbstractModel):
    # TODO(Tim) Add latex model function and properly describe model
    # in docstring
    """Simple GLM like model of precursor feature




    """
    def __init__(self,
                 num_observed:int,
                 peak_num:int,
                 z:int,
                 intensity:NDArrayFloat,
                 scan:NDArrayInt,
                 mz:NDArrayFloat,
                 ims_mu:float,
                 ims_sigma:float,
                 mz_mu:float,
                 mz_sigma:float,
                 alpha_lam:float,
                 name:str="",
                 model:pm.Model = None):
        """docstring of constructor"""
        super().__init__(name,model)
        # accessible from outside (data and hyperpriors)
        self.peak_num = pm.MutableData("peak_num",peak_num)
        self.charge = pm.MutableData("charge",z)
        self.intensity = pm.MutableData("intensity",intensity)
        self.scan = pm.MutableData("scan",scan)
        self.mz = pm.MutableData("mz",np.tile(mz,(peak_num,1)).T)
        self.peaks = pm.MutableData("peaks",np.tile(np.arange(peak_num),
                            (num_observed,1)))
        self.ims_mu = pm.MutableData("ims_mu",ims_mu)
        self.ims_sigma = pm.MutableData("ims_sigma",ims_sigma)
        self.mz_mu = pm.MutableData("mz_mu",mz_mu)
        self.mz_sigma = pm.MutableData("mz_sigma",mz_sigma)
        self.alpha_lam = pm.MutableData("alpha_lam",alpha_lam)
        """
        # this lets prior predictive fail, because self.intensity zero
        intensity_sum = at.sum(self.intensity)
        mz_mean = at.dot(self.mz[:,0],self.intensity)/intensity_sum
        scan_mean = at.dot(self.scan,self.intensity)/intensity_sum
        scan_range = at.max(self.scan)-at.min(self.scan)
        """
        # priors
        # IMS
        self.i_t = pm.Normal("i_t",mu=self.ims_mu,sigma=self.ims_sigma/2)
        self.i_s = pm.Uniform("i_s",lower=0,upper=self.ims_sigma)

        # mass spec
        self.ms_mz = pm.Normal("ms_mz",mu=self.mz_mu,sigma=self.mz_sigma)
        self.ms_s = pm.Exponential("ms_s",lam=10)
        self.pos = self.peaks/(self.charge+1)+self.ms_mz
        self.lam = 0.000594 * (self.charge+1)*self.ms_mz - 0.03091
        self.ws_matrix = self.lam**self.peaks/ \
                         at.gamma(self.peaks)* \
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
                             ,axis=1)

        # f(t,mz) = α*f_IMS(t)*f_mass(MZ)
        self.pi = pm.Deterministic("mu",var=self.pi1*self.pi2,auto=True)
        # debug deterministic:
        # self.pi = self.pi1*self.pi2
        # Model error
        self.me = pm.HalfNormal("me",sigma=10)
        # Likelihood
        self.obs = pm.Normal("obs",
                             mu=self.pi,
                             sigma=self.me,
                             observed=self.intensity)

