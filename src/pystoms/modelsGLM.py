import pymc3 as pm
import pymc3.math as pmath
import numpy as np
from scipy.special import factorial
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AbstractModel(pm.Model):
    def __init__(self,name,model):
        super().__init__(name,model)
    
    def _plotPosterior3D(self,num:int = 40):
        def _fxy(x,y,α,pos_ms,σ_ms,w_ms,pos_ims,σ_ims):
            shape = x.shape
            x = x.flatten().reshape((-1,1))
            y = y.flatten()
            z = np.zeros_like(y)
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

    def _sample(self,resample:bool=False,**kwargs):
        if self.trace == None or resample:
            with self as model:
                self.trace = pm.sample(kwargs)
        else:
            print("This model was already sampled, if you want to resample pass 'resample=True'")
            return 

    def evaluation(self,resample:bool=False,**kwargs):
        self._sample(resample,kwargs)
        self._plotPosterior3D()


class ModelGLM3D(AbstractModel):
    def __init__(self,n,z,intensity,scan,mz,name:str="",model:pm.Model=None):
        super().__init__(name,model)
        mz_mean = np.average(mz,weights=intensity)
        scan_mean = np.average(scan,weights=intensity)
        scan_range = scan.max()-scan.min()
        self.charge = pm.Data("charge",z)
        self.intensity = pm.Data("intensity",intensity)
        self.scan = pm.Data("scan",scan)
        self.mz = pm.Data("mz",np.reshape(mz,(mz.size,1)))
        
        self.Var("I_t",pm.Normal.dist(mu=scan_mean,sigma=scan_range/2))
        self.Var("I_s",pm.Uniform.dist(lower=0,upper=scan_range))      

        self.Var("MS_mz",pm.Normal.dist(mu=mz_mean,sigma=10))
        self.Var("MS_s",pm.Exponential.dist(lam=10))
        #pm.Deterministic("peaks",var=np.arange(n))
        self.peaks = np.arange(n)
        pm.Deterministic("pos",var = self.peaks/(self.charge+1)+self.MS_mz)
        pm.Deterministic("lam",var = 0.000594 * (self.charge+1)*self.MS_mz - 0.03091)
        pm.Deterministic("ws", var = self.lam**self.peaks/factorial(self.peaks)*pmath.exp(-self.lam))
        self.Var("alpha",pm.Exponential.dist(lam = 1/self.intensity.get_value().max()))
        PI1 = self.alpha*pmath.exp(-(self.I_t-self.scan)**2/(2*self.I_s**2))
        PI2 = pmath.sum(self.ws*pmath.exp(-(self.pos-self.mz.get_value())**2/(2*self.MS_s**2)),axis=1)
        PI = PI1*PI2
        self.Var("me",pm.HalfNormal.dist(sigma=10))
        self.Var("obs",pm.Normal.dist(mu=PI,sigma=self.me),data=self.intensity)


