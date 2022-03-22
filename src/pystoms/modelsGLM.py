"""Probabilistic General linear models for LC-IMS-MS precursor features



"""
import pymc as pm
import pymc.math as pmath
import numpy as np
#from scipy.special import factorial
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

    def ppc_surface(x,y,α,pos_ms,σ_ms,w_ms,pos_ims,σ_ims):
                shape = x.shape
                x = x.flatten().reshape((-1,1))
                y = y.flatten()
                z = np.zeros_like(y,dtype=np.float64)
                for i in range(num):
                    α_i = α[i]

                    σ_ms_i = σ_ms[i]
                    pos_ms_i = pos_ms[i]
                    w_ms_i = w_ms[i]

                    σ_ims_i = σ_ims[i]
                    pos_ims_i = pos_ims [i]

                    t1 = np.exp(-0.5*np.power(pos_ims_i-y,2)/np.power(σ_ims_i,2))
                    t2 = np.sum(w_ms_i*np.exp(-0.5*np.power(pos_ms_i-x,2)/np.power(σ_ms_i,2)),axis=1)
                    z += α_i*t1*t2
                z /= num
                return(np.reshape(z,shape))
        def _plotPosterior3D(self,num:int = 40):

        trace = self.trace
        # extract hull boundaries of feature
        obs_x = trace.constant_data.mz.values.flatten()
        obs_y = trace.constant_data.scan.values
        obs_z = trace.constant_data.intensity.values
        # set axis limits accordingly
        xmin = obs_x.min()-1
        xmax = obs_x.max()+1
        ymin = obs_y.min()-1
        ymax = obs_y.max()+1
        # x axis and y axis , scan intervall is 1, mz 0.01

        y = np.arange(ymin,ymax)
        x = np.arange(xmin*100,xmax*100)/100
        x,y = np.meshgrid(x,y)
        # get positions of chosen values from posterior
        num_chains = trace.posterior.dims["chain"]
        num_draws = trace.posterior.dims["draw"]
        sample_idx = np.random.choice(range(num_draws),num)
        # now go through chains and calculate mean surface for chosen paramters
        zs = []
        # instantiate figure
        fig = make_subplots(rows=num_chains//2,cols=2,specs=[[{"type":"scatter3d"}]*2]*(num_chains//2))
        for chain_i in range(num_chains):
            # select parameters from posterior
            α = trace.posterior["alpha"][chain_i].values[sample_idx]
            pos_ms = trace.posterior["pos"][chain_i].values[sample_idx]
            σ_ms = trace.posterior["MS_s"][chain_i].values[sample_idx]
            w_ms = trace.posterior["ws"][chain_i].values[sample_idx]
            pos_ims = trace.posterior["I_t"][chain_i].values[sample_idx]
            σ_ims = trace.posterior["I_s"][chain_i].values[sample_idx]
            # calculate z (=intensity values)
            z = _fxy(x,y,α,pos_ms,σ_ms,w_ms,pos_ims,σ_ims)
            # add to list and plots
            zs.append(z)
            fig.add_trace(go.Surface(x=x,y=y,z=z,opacity=0.3,showscale=False),row=chain_i//2+1,col=chain_i%2+1)
            fig.add_trace(go.Scatter3d(x=obs_x,y=obs_y,z=obs_z,mode="markers",marker=dict(size=5,color="black")),row=chain_i//2+1,col=chain_i%2+1)
        fig.update_layout(autosize=False,width=1000,height=1000)
        fig.show()


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
            # TODO(Tim): add out of sample prior pred as variable to predictions
            raise NotImplementedError("Out of sample prior predictions \
                                       are not yet supported")

        # prior predictions on coordinates of observed data
        # used for posterior sampling (in sample predictions)
        if is_prior and not is_grid_predictive:
            with self as model:
                prior = pm.sample_prior_predictive(**kwargs)
            prior_idata = az.from_pymc3(prior=prior)
            if self.idata is not None:
                self.idata.extend(prior_idata)
            else:
                self.idata = prior_idata

        # posterior out of sample predictions with grid
        # as predictors -> model surface
        if not is_prior and is_grid_predictive:
            data = self.idata.constant_data
            # extract hull boundaries of feature
            obs_x = data.mz.values.flatten()
            obs_y = data.scan.values
            charge = data.charge.values[0]
            # set axis limits accordingly
            xmin = obs_x.min()-1
            xmax = obs_x.max()+1
            ymin = obs_y.min()-1
            ymax = obs_y.max()+1
            # x axis and y axis , scan intervall is 1, mz 0.01

            y = np.arange(ymin,ymax,dtype=int)
            x = np.arange(xmin*100,xmax*100)/100
            x,y = np.meshgrid(x,y)
            x = x.flatten().reshape((x.size,1))
            y = y.flatten()
            #TODO(Tim) This does not work, must be tailored
            pm.set_data({"scan":y,
                         "mz":x, # awful but mzs currently needed as column
                         "intensity":np.zeros_like(y),
                         "charge":charge
                        },
                        model=self)
            prediction = pm.sample_posterior_predictive(self.idata.posterior,model=self)
            az.from_pymc3_predictions(prediction,
                                      idata_orig=self.idata,
                                      inplace=True)
        # posterior predictions on coordinates of observed data
        # used for posterior sampling (in sample predictions)
        if not is_prior and not is_grid_predictive:
            with self as model:
                pp = pm.sample_posterior_predictive(self.idata,
                                                    **kwargs)
                pp_idata = az.from_pymc3(posterior_predictive=pp)
                self.idata.extend(pp_idata)

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

        self.peak_num = pm.Data("peak_num",peak_num)
        self.num_observed = pm.Data("num_observed",num_observed)
        self.charge = pm.Data("charge",z)
        self.intensity = pm.Data("intensity",intensity)
        self.scan = pm.Data("scan",scan)
        self.mz = pm.Data("mz",np.tile(mz,(peak_num,1)).T)

        intensity_sum = pmath.sum(self.intensity)
        mz_mean = pmath.dot(self.mz[:,0],self.intensity)/intensity_sum
        scan_mean = pmath.dot(self.scan,self.intensity)/intensity_sum
        scan_range = at.max(self.scan)-at.min(self.scan)


        self.I_t = pm.Normal("I_t",mu=scan_mean,sigma=scan_range/2)
        self.I_s = pm.Uniform("I_s",lower=0,upper=scan_range)

        self.MS_mz = pm.Normal("MS_mz",mu=mz_mean,sigma=10)
        self.MS_s = pm.Exponential("MS_s",lam=10)

        self.peaks = at.tile(at.arange(at.cast(self.peak_num,"int32")),
                            (at.cast(self.num_observed,"int32"),1))
        self.pos = pm.Deterministic("pos",
                         var = self.peaks/(self.charge+1)+self.MS_mz)
        self.lam = pm.Deterministic("lam",
                         var = 0.000594 * (self.charge+1)*self.MS_mz - 0.03091)
        self.ws = pm.Deterministic("ws",
                         var = self.lam**self.peaks[0,:]/ \
                         at.gamma(self.peaks[0,:])* \
                         pmath.exp(-self.lam))
        self.ws_matrix = at.tile(self.ws,
                                (at.cast(self.num_observed,"int32"),1))
        self.alpha = pm.Exponential("alpha",lam = 1/at.max(self.intensity))
        self.pi1 = self.alpha*pmath.exp(-(self.I_t-self.scan)**2/(2*self.I_s**2))
        self.pi2 = pmath.sum(self.ws_matrix*pmath.exp(-(self.pos-self.mz)**2/(2*self.MS_s**2)),axis=1)
        self.pi = self.pi1*self.pi2
        self.me = pm.HalfNormal("me",sigma=10)
        self.obs = pm.Normal("obs",mu=self.pi,sigma=self.me,observed=self.intensity)

