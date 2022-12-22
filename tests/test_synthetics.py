import pytest
import numpy as np
from scipy.integrate import quad
from functools import partial
from pystoms.synthetics import IsotopicAveragineDistribution

@pytest.fixture(scope="module")
def seed():
    return 2022

@pytest.fixture(scope="module")
def rng(seed):
    return np.random.default_rng(seed)

class TestIsotopeAveragineDistribution:

    args1 = {
        "x":np.arange(2000,2100)/10,
        "mass":805.3,
        "charge":4,
        "sigma":0.05,
        "num_peaks":6
    }
    args2 = {
        "x":np.arange(3040,3140)/10,
        "mass":305.3,
        "charge":1,
        "sigma":[0.01,0.01,0.05,0.05,0.05,0.05,0.1],
        "num_peaks":7
    }

    def get_loc(self,loc_at_0,arguments):
        if loc_at_0:
            loc=0
        else:
            loc = arguments["mass"]/arguments["charge"]
        return loc

    @pytest.fixture(scope="class",params=[args1,args2])
    def arguments(self,request):
        return request.param.copy()

    @pytest.fixture(scope="class")
    def iso_dist(self):
        return IsotopicAveragineDistribution()

    @pytest.fixture(scope="class",params=[True,False])
    def loc_at_0(self,request):
        return request.param

    def test_pdf(self,iso_dist,arguments,loc_at_0):
        loc = self.get_loc(loc_at_0,arguments)
        x = arguments.pop("x")-arguments["mass"]/arguments["charge"]+loc
        pdf = partial(iso_dist.pdf,**arguments,loc=loc)
        y = pdf(x=x)
        assert y.shape == x.shape, f"Array of densities y shape {y.shape} does not correspond to input array x shape {x.shape}"
        # assert that pdf integrates to 1
        I = quad(pdf,x.min(),x.max())[0]
        assert np.isclose(I,1), f"Integral of pdf is not 1, but {I}."
        # assert that all densities are >= 0
        assert np.all(y>=0), "Probability densities below zero detected."

    def test_rvs(self,iso_dist,arguments,rng,loc_at_0):
        loc = self.get_loc(loc_at_0,arguments)
        arguments.pop("x")
        samples = iso_dist.rvs(**arguments,loc=loc,size=5000,random_state=rng)
        assert samples.size == 5000

    def test_rvs_to_intensities(self,iso_dist,arguments,rng,loc_at_0):
        loc = self.get_loc(loc_at_0,arguments)
        arguments.pop("x")
        bins_pos,intensities = iso_dist.rvs_to_intensities(**arguments,loc=loc,random_state=rng,bin_width=0.0001)

    def test_averagine_isotopic(self,iso_dist,arguments):
        mass = arguments["mass"]
        num_peaks = arguments["num_peaks"]
        lambdas = iso_dist._averagine_isotopic(mass,num_peaks)
        assert np.isclose(sum(lambdas),1)