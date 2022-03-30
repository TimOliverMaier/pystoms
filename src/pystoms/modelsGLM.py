"""Probabilistic General linear models for LC-IMS-MS precursor features



"""
from logging import warning
import pandas as pd
import pymc as pm
import pymc.math as pmath
import numpy as np
#from scipy.special import factorial
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Optional
import arviz as az
from aesara import tensor as at

class AbstractModel(pm.Model):
    """Abstract model class for model evaluation.

    This class provides the 2D GLM model subclasses with
    evaluation methods.


    Attributes:
        idata (Optional[az.InferenceData]):
          inferenceData of last model fit.
    """
    def __init__(self,name:str,model:Optional[pm.Model]):
        # name and model must be passed to pm.Model
        super().__init__(name,model)
        # instantiate inference data
        self.idata :Optional[az.InferenceData] = None

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
        """

        # for overview on arviz inferenceData groups visit
        # https://arviz-devs.github.io/arviz/schema/schema.html
        # only prior can be sampled before posterior was sampled
        if self.idata is None and not is_prior:
            self._sample()

        # prior out of sample predictions with grid
        # as predictors -> model surface
        if is_prior and is_grid_predictive:
            self._set_grid_data()
            prior_prediction = pm.sample_prior_predictive(model=self,
                                                          **kwargs)
            warning("Prior out-of-sample predictions are currently a work around.\n\
              Arviz will warn about non-defined InferenceData group 'prior_predictions'.")
            self.idata.add_groups({"prior_predictions":prior_prediction.prior_predictive})
            self._set_observed_data()
        # prior predictions on coordinates of observed data
        # used for posterior sampling (in sample predictions)
        if is_prior and not is_grid_predictive:
            with self as model:
                prior_idata = pm.sample_prior_predictive(**kwargs)
            if self.idata is not None:
                self.idata.extend(prior_idata)
            else:
                self.idata = prior_idata

        # posterior out of sample predictions with grid
        # as predictors -> model surface
        if not is_prior and is_grid_predictive:
            self._set_grid_data()
            prediction = pm.sample_posterior_predictive(self.idata,
                                                        model=self,
                                                        extend_inferencedata=True,
                                                        predictions=True,
                                                        **kwargs)
            # until fix inference data must be updated
            self.idata.extend(prediction)
            self._set_observed_data()
        # posterior predictions on coordinates of observed data
        # used for posterior sampling (in sample predictions)
        if not is_prior and not is_grid_predictive:
            with self as model:
                pp = pm.sample_posterior_predictive(self.idata,
                                                    extend_inferencedata=True,
                                                    **kwargs)

    def visualize_predictions_scatter(self,size:int = 50, in_sample:bool = True, is_prior:bool = False, pred_name:str = "obs",plot_observed_data:bool = True) -> None:
        """Plotting posterior/prior predictions.

        Args:
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
        """

        if not in_sample:
            predictors_data = self.idata.predictions_constant_data
            if is_prior:
                draw_sample = np.random.choice(self.idata\
                                                .prior_predictions\
                                                .draw.values, size)
                predicted_data = self.idata.\
                                    prior_predictions[{"draw":draw_sample}]
            if not is_prior:
                draw_sample = np.random.choice(self.idata\
                                                .predictions\
                                                .draw\
                                                .values,
                                                size)
                predicted_data = self.idata.predictions[{"draw":draw_sample}]

        if in_sample:
            predictors_data = self.idata.constant_data
            if is_prior:
                draw_sample = np.random.choice(self.idata\
                                                .prior_predictive\
                                                .draw\
                                                .values,
                                                size)
                predicted_data = self.idata\
                                    .prior_predictive[{"draw":draw_sample}]
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

        df_predictors = pd.merge(df_mz,
                                df_scan,
                                left_on="mz_dim_0",
                                right_on="scan_dim_0")\
                                .drop(columns=["scan_dim_0"])\
                                .rename(columns={
                                        "mz_dim_0":"data_point"
                                        })
        # get corresponding predicted values
        if pred_name not in predicted_data.data_vars.keys():
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
        fig = px.scatter_3d(data_frame=df_plot,
                            x="scan",
                            y="mz",
                            z=pred_name,
                            color="chain",
                            opacity=0.1)
        if plot_observed_data:
            obs_data_trace = self.plot_feature_data(return_fig_trace=True)
            fig.add_trace(obs_data_trace)
        fig.show(renderer="notebook")

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

        data = self.idata.constant_data
        # extract hull boundaries of feature
        obs_x = data.mz.values.flatten()
        obs_y = data.scan.values
        charge = data.charge.values[0]
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

        pm.set_data({"scan":y,
                      "mz":x, # awful but mzs currently needed as column
                      "intensity":np.zeros_like(y,dtype="float"),
                      "charge":charge,
                      "peak_num":peak_num,
                      "peaks":np.tile(np.arange(peak_num),(y.size,1)),
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

        pm.set_data({"scan":obs_y,
                      "mz":obs_x, # awful but mzs currently needed as column
                      "intensity":obs_z,
                      "charge":charge,
                      "peak_num":peak_num,
                      "peaks":np.tile(np.arange(peak_num),(obs_y.size,1)),
                    },
                    model=self)

    def _sample(self,**kwargs):
        if "return_inferencedata" not in kwargs:
            kwargs["return_inferencedata"] = True

        with self as model:
            trace = pm.sample(**kwargs)

        if self.idata is not None:
            self.idata.extend(trace, join="right")
        else:
            self.idata = trace

    def evaluation(self,is_new_feature:bool = True,**kwargs):
        if is_new_feature:
          self.idata = None
        self._sample(**kwargs)


class ModelGLM3D(AbstractModel):
    def __init__(self,num_observed:int,peak_num:int,z,intensity,scan,mz,name:str="",model:pm.Model=None):
        super().__init__(name,model)

        self.peak_num = pm.MutableData("peak_num",peak_num)
        self.charge = pm.MutableData("charge",z)
        self.intensity = pm.MutableData("intensity",intensity)
        self.scan = pm.MutableData("scan",scan)
        self.mz = pm.MutableData("mz",np.tile(mz,(peak_num,1)).T)
        self.peaks = pm.MutableData("peaks",np.tile(np.arange(peak_num),
                            (num_observed,1)))

        intensity_sum = at.sum(self.intensity)
        mz_mean = at.dot(self.mz[:,0],self.intensity)/intensity_sum
        scan_mean = at.dot(self.scan,self.intensity)/intensity_sum
        scan_range = at.max(self.scan)-at.min(self.scan)


        self.I_t = pm.Normal("I_t",mu=scan_mean,sigma=scan_range/2)
        self.I_s = pm.Uniform("I_s",lower=0,upper=scan_range)

        self.MS_mz = pm.Normal("MS_mz",mu=mz_mean,sigma=10)
        self.MS_s = pm.Exponential("MS_s",lam=10)
        self.pos = self.peaks/(self.charge+1)+self.MS_mz
        self.lam = 0.000594 * (self.charge+1)*self.MS_mz - 0.03091
        self.ws_matrix = self.lam**self.peaks/ \
                         at.gamma(self.peaks)* \
                         pmath.exp(-self.lam)

        self.alpha = pm.Exponential("alpha",lam = 1/at.max(self.intensity))

        self.pi1 = self.alpha*pmath.exp(-(self.I_t-self.scan)**2/(2*self.I_s**2))
        self.pi2 = pmath.sum(self.ws_matrix*pmath.exp(-(self.pos-self.mz)**2/(2*self.MS_s**2)),axis=1)
        self.pi = pm.Deterministic("mu",var=self.pi1*self.pi2)

        self.me = pm.HalfNormal("me",sigma=10)
        self.obs = pm.Normal("obs",mu=self.pi,sigma=self.me,observed=self.intensity)

