from pystoms.synthetics import SyntheticPeptideFeature
import pymc3 as pm
import arviz as az
import numpy as np
import pymc3.math as pmath
from scipy.special import factorial
import pandas as pd
import matplotlib.pyplot as plt
import os


class AbstractModel(pm.Model):
    """Wrapper class providing methods

    This class is a subclass of `pymc3.Model`
    and super class of implemented models.
    It is supposed to provide with methods
    for sampling, evaluation and prior/posterior analysis.
    These methods use the `pymc3` functions for
    this purpose.

    This class is not supposed to be used externally.

    """

    # subclassing pm.Model analogue to https://docs.pymc.io/api/model.html
    def __init__(self, name, model, syn):
        # here name is usually passed to pm.model, but
        # to prevent long variable names "" is passed to pm.Model
        # and model name is stored in AbstractModel layer
        super().__init__("", model)
        self.syn = syn
        # modelname is an straight-forward workaround, to have a model
        # identifier without altering variable names.
        self.modelname = name
        # some methods test if trace is existent
        # to avoid resampling
        self.trace = None

    def sample(self, save_to_file: bool = False, path_output_dir: str = "TraceData"):
        """Uses pm.sample to sample trace

        Args:
            save_to_file (bool, optional): If True, trace is stored in .nc format.
                Defaults to False.
            path_output_dir (str, optional): Relative path to folder where to store trace data.
                Defaults to "TraceData".
        """
        # Sampling
        with self as model:
            self.trace = pm.sample(return_inferencedata=True, tune=3000)
        # Storing
        if save_to_file:
            # TraceData/ModelName/traceData.nc"
            path = path_output_dir + "/" + self.modelname + "/"
            if not os.path.exists(path):
                os.makedirs(path)
            self.trace.to_netcdf(path + "traceData.nc")

    def evaluation(
        self,
        prior_pred: bool = False,
        post_pred: bool = False,
        post_sample_n: int = 100,
        save_fig: bool = False,
        path_output_dir: str = "Evaluation",
    ):
        """Evaluation of model

        Evaluates model by comparison of infered values to original
        syntheticPeptideFeature parameters. Optionally, prior and posterior
        predictive checks can be performed.

        Args:
            prior_pred (bool, optional): If True, prior predictive check
                is performed. Defaults to False.
            post_pred (bool, optional): If True, posterior predictive check
                is performed. Defaults to False.
            post_sample_n (int, optional): Number of samples (per chain) to draw
                for posterior predictive check. Defaults to 100.
            save_fig (bool, optional): If true, generated figures
                and data is stored, not printed to console. Defaults to False.
            path_output_dir (str, optional): relative path of folder to store
                output in. Defaults to "Evaluation".

        Raises:
            ValueError: If model has no connected syntheticPeptideFeature.
            NameError: If model has no modelname.

        Returns:
            pd.DataFrame: Summary of model fit.
        """
        # Associated synthetic is necessary, since it provides
        # the reference values
        if self.syn == None:
            raise ValueError(
                "This model instance did not receive synthetic data or no SyntheticPeptideFeature instance was passed."
            )
        # model must have name if evaluation shall be stored as files
        if save_fig and self.modelname == "":
            raise NameError("For exporting figures, model must be named.")
        # generate relative path e.g. "Evaluation/model1/"
        if save_fig:
            path = path_output_dir + "/" + self.modelname + "/"
            if not os.path.exists(path):
                os.makedirs(path)
        # if no sampling was performed yet, evaluation calls sample method
        if self.trace == None:
            self.sample(save_fig, path_output_dir)
        # prior predictive check is executed in priorAnalysis method
        if prior_pred:
            self.priorAnalysis(save_fig, path_output_dir)

        # get dictionary with reference values for all infered variables
        ref_vals = self._getReferenceValues()

        # posterior plot with reference values as verical lines
        az.plot_posterior(self.trace, ref_val=ref_vals)
        if save_fig:
            plt.savefig(path + "posteriorAnalysis.png", dpi=300)
            plt.close()
        else:
            plt.show()

        # trace plot
        az.plot_trace(
            self.trace, chain_prop={"color": ["green", "red", "blue", "black"]}
        )
        if save_fig:
            plt.savefig(path + "trace.png", dpi=300)
            plt.close()
        else:
            plt.show()

        # plot pairs
        az.plot_pair(self.trace)
        if save_fig:
            plt.savefig(path + "pairs.png", dpi=300)
            plt.close()
        else:
            plt.show()

        # forest plots for all variables
        # axis must be seperated, since values of variables
        # are not standardized
        # find number of variables to plot in current model
        plot_length = len(self.trace.posterior.keys())
        # variables in rows , chains in left column and r_hat in right column
        Fig, axs = plt.subplots(
            nrows=plot_length, ncols=2, figsize=(plot_length * 4, 10)
        )

        for variable, ax_i in zip(self.trace.posterior.keys(), axs):
            az.plot_forest(self.trace, r_hat=True, ax=ax_i, var_names=[variable])

        Fig.tight_layout()

        if save_fig:
            plt.savefig(path + "forest.png", dpi=300)
            plt.close()
        else:
            plt.show()

        # posterior predictive check is executed in posterior Analysis method
        if post_pred:
            self.posteriorAnalysis(post_sample_n, save_fig, path_output_dir)

        # model summary table, export as csv
        model_summary = self._summary(ref_vals)
        # index is model variables, we need those in a column
        model_summary.reset_index(inplace=True)
        model_summary.rename(columns={"index": "variable"}, inplace=True)
        if save_fig:
            model_summary.to_csv(path + "summary.csv")
        else:
            print(model_summary)

        # return summary, this is done for syntheticsSet evaluation
        # in pySTOMS.utilites
        return model_summary

    def _summary(self, ref_values: dict):
        """Generates model summary with az.summary

        Additionally to az.summary, reference values
        of variables are stored, as well as their
        deviation (in %) to the corresponding posterior's mean
        and their position in the posterior distribution.

        Args:
            ref_values (dict): Dictionary of reference values,
                as used in az.plot_posterior.

        Returns:
            pd.DataFrame: Summary dataframe.
        """
        # Dictionary for az.plot_posterior refernce values
        # is oddly formatted. Therefore this condensation:
        ref_values_condensed = {}
        for variable in ref_values:
            ref_values_condensed[variable] = ref_values[variable][0]["ref_val"]

        # calculate reference value positions as dictionary {"variable_name":position}
        ref_positions = self._getReferencePositions(ref_values_condensed)

        # get summary from arviz
        summary = az.summary(self.trace, round_to="none")

        summary["reference"] = summary.index.map(ref_values_condensed)
        summary["deviation_(%)"] = (
            (summary["reference"] - summary["mean"]) / summary["mean"] * 100
        )
        summary["ref_pos_(%)"] = summary.index.map(ref_positions)
        summary["model_name"] = self.modelname
        # to do: handling of discrete variables
        # to do: single chain analysis

        return summary

    def _getReferencePositions(self, ref_values_condensed: dict):
        """Calculate position of refernce values in
        posterior distributions

        Args:
            ref_values_condensed (dict): Reference values of variables.

        Returns:
            dict: positions of reference values.
        """
        reference_positions = {}
        # for discrete variables:
        # how many draws in posterior are equal?
        # for continuous:
        # how many draws in posterior are less or equal?
        discrete_variables = [v.name for v in self.disc_vars]
        continuous_variables = [v.name for v in self.cont_vars]

        for variable, ref_value in ref_values_condensed.items():
            # test if variable (transformed var) is discrete
            trans_var = variable + "_log__"
            if variable in discrete_variables or trans_var in discrete_variables:
                # all samples for this variable
                sample = self.trace.posterior[variable].data.flatten()
                # test if not nan or inf
                is_fin = np.isfinite(sample)
                # how man finite samples are equal (in %) ?
                is_equal = sample == ref_value
                reference_positions[variable] = is_equal.mean(where=is_fin) * 100
            # test if variable (transformed var) is continuous
            if variable in continuous_variables or trans_var in continuous_variables:
                sample = self.trace.posterior[variable].data.flatten()
                is_fin = np.isfinite(sample)
                is_se = sample <= ref_value
                reference_positions[variable] = is_se.mean(where=is_fin) * 100
            else:
                # not a variable in this model
                continue
        return reference_positions

    def _getReferenceValues(self):
        """Get reference values of variables from
        connected syntheticPeptideFeature instance

        Returns:
            dict: reference values stored in dictionary
                formatted for az.plot_posterior method.
        """

        # Reference values to show location in infered posteriors
        ref_vals = {}

        # Reference Values for retention time models
        ref_vals["emg_μ"] = [{"ref_val": self.syn.emg_μ}]
        ref_vals["emg_σ"] = [{"ref_val": self.syn.emg_σ}]
        ref_vals["emg_λ"] = [{"ref_val": self.syn.emg_λ}]

        # Reference Values for isotopic models
        ref_vals["charge_state"] = [{"ref_val": self.syn.charge - 1}]
        ref_vals["mass"] = [{"ref_val": self.syn.mass}]
        ref_vals["λ_a"] = [{"ref_val": 0.000594}]
        ref_vals["sigma"] = [{"ref_val": self.syn.isotopic_sigma}]
        ref_vals["mz"] = [{"ref_val": self.syn.mass / self.syn.charge}]
        ref_vals["λ"] = [{"ref_val": 0.000594 * self.syn.mass - 0.03091}]

        return ref_vals

    def priorAnalysis(
        self, save_fig: bool = False, path_output_dir: str = "Evaluation"
    ):
        """Perform prior predictive check using pymc3 and arviz.

        Args:
            save_fig (bool, optional): If True, plot is exported to png. Defaults to False.
            path_output_dir (str, optional): Path to folder to store file in. Defaults to "Evaluation".

        Raises:
            NameError: If model has no name.
        """
        if save_fig and self.modelname == "":
            raise NameError("For exporting figures, model must be named.")
        if save_fig:
            path = path_output_dir + "/" + self.modelname + "/"
            if not os.path.exists(path):
                os.makedirs(path)
        with self as model:
            self.prior = pm.sample_prior_predictive()
            az.plot_ppc(az.from_pymc3(prior=self.prior), group="prior")
            if save_fig:
                plt.savefig(path + "priorPredicitveAnalysis.png", dpi=300)
                plt.close()
            else:
                plt.show()

    def posteriorAnalysis(
        self,
        post_sample_n: int = 100,
        save_fig: bool = False,
        path_output_dir: str = "Evaluation",
    ):
        """Performs posterior predictive check using pymc3 and arviz.


        Args:
            post_sample_n (int, optional): Number samples to draw from posterior (per chain).
                Defaults to 100.
            save_fig (bool, optional): If True, plot is saved to png. Defaults to False.
            path_output_dir (str, optional): Relative path for output. Defaults to "Evaluation".

        Raises:
            NameError: If model has no name.
        """
        if self.trace == None:
            self.sample()
        if save_fig and self.modelname == "":
            raise NameError("For exporting figures, model must be named.")
        if save_fig:
            path = path_output_dir + "/" + self.modelname + "/"
            if not os.path.exists(path):
                os.makedirs(path)
        # randomly subset trace to reduce computational load
        rand_ind = np.random.choice(
            self.trace.posterior["draw"], size=post_sample_n, replace=False
        )
        subset_trace = self.trace.sel(draw=rand_ind)
        subset_trace.posterior["draw"] = np.arange(post_sample_n)

        with self as model:
            self.posterior_predictive = pm.sample_posterior_predictive(
                subset_trace, keep_size=True
            )
            az.plot_ppc(az.from_pymc3(posterior_predictive=self.posterior_predictive))
            if save_fig:
                plt.savefig(path + "posteriorPreditiveAnalysis.png", dpi=300)
                plt.close()
            else:
                plt.show()

            chains = subset_trace.posterior["chain"].data
            num_chains = len(chains)
            nrows = num_chains // 2 + num_chains % 2
            ncols = 2
            Fig = plt.figure()
            Fig.set_dpi(300)

            for chain in chains:
                # this fails if observed variable is not called 'obs' anymore
                ppc = {}
                ppc["obs"] = self.posterior_predictive["obs"][chain]
                ax = Fig.add_subplot(nrows, ncols, chain + 1)
                az.plot_ppc(az.from_pymc3(posterior_predictive=ppc), ax=ax)
                ax.set_title(f"Chain {chain}")
                ax.get_legend().remove()

            handles, labels = ax.get_legend_handles_labels()
            Fig.legend(handles, labels, bbox_to_anchor=(0.7, 0.01))
            plt.tight_layout()

            if save_fig:
                plt.savefig(path + "posteriorPredperChain.png", dpi=300)
                plt.close()
            else:
                plt.show()

    def __repr__(self):
        with self as model:
            pm.model_to_graphviz()

    def __str__(self):
        with self as model:
            pm.model_to_graphviz()


class ModelEMG(AbstractModel):
    r"""Model for peptide elution profile.

    Fitting exponentially modified gaussian distribution (EMG)
    to retention times (:math:`rt`). Inference of EMG parameters :math:`μ,σ,λ`.

    Model design:

    .. math::

        &rt ∼ \mathrm{EMG}(μ,σ,λ)

        &μ ∼ \mathrm{Normal}\left(\hat{μ},\hat{σ}\right)

        &σ ∼ \mathrm{Exponential}\left(\frac{1}{\hat{σ}}\right)

        &λ ∼ \mathrm{Exponential}\left(\frac{1}{\hat{λ}}\right)

    With :math:`\hat{μ},\hat{σ}` defaulting to the mean and standard deviation
    of the passed retention time data, respectively.
    :math:`\hat{λ}` defaults to 1.

    Attributes:
        emg_μ (pm.Normal): Location of EMG distribution.
        emg_σ (pm.Exponential): Width of EMG distribution.
        emg_λ (pm.Exponential): Tailing of EMG distribution.
        obs (pm.ExGaussian): EMG distribution.

    """

    def __init__(
        self,
        rt_values: np.ndarray,
        μ_hat: float = None,
        σ_hat: float = None,
        λ_hat: float = 1.0,
        name: str = "",
        model: pm.Model = None,
        syn: SyntheticPeptideFeature = None,
    ):
        """Initialize elution profile model.

        Model elution profile via EMG distribution.
        EMG is fitted to retention times (rt).

        Args:
            rt_values (np.ndarray): Retention time data to fit model to.
            μ_hat (float,optional): Mean of prior normal distribution of EMG
                μ parameter. If not provided, mean of data is taken. Defaults to None.
            σ_hat (float,optional): Expected value of prior exponential distribution
                of EMG σ parameter. If not provided, standard deviation of data is taken.
                Defaults to None.
            λ_hat (float,optional): Expected value of prior exponential distribution
                of EMG λ parameter. Defaults to 1.
            name (str,optional): Name of model.
            model (pm.Model,optional): ?pymc3 model.
            syn (SyntheticPeptideFeature, optional): SyntheticPeptideFeature instance data is sampled from.
                Needed for model evaluation. Defaults to None.
        """
        if μ_hat == None:
            μ_hat = rt_values.mean()
        if σ_hat == None:
            σ_hat = np.sqrt(rt_values.var())

        super().__init__(name, model, syn)
        # Priors
        self.Var("emg_μ", pm.Normal.dist(mu=μ_hat, sigma=σ_hat))
        self.Var("emg_σ", pm.Exponential.dist(lam=1 / σ_hat))
        self.Var("emg_λ", pm.Exponential.dist(lam=1 / λ_hat))
        # Likelihood
        self.Var(
            "obs",
            pm.ExGaussian.dist(mu=self.emg_μ, sigma=self.emg_σ, nu=self.emg_λ),
            data=rt_values,
        )


class ModelIsotopicPattern(AbstractModel):
    r"""Model for Isotopic pattern.

    Fitting normal mixture distribution to mass to charge ratios (:math:`mz`).
    Instances of this class do not estimate charge :math:`z` nor averagine
    model parameter :math:`λ_{a}`.
    Charge (:math:`z`) of peptide must be provided. Data is transformed for fit:

    :math:`m=mz*z`

    This model allows for inference of peptide mass :math:`m`
    and standard deviation :math:`σ` of normal distributions
    around mass peaks.

    Model design:

    .. math::

        &mz*z ∼ \mathrm{NormalMixture}\left(\vec{w},\vec{μ},\vec{σ}\right)

        &\vec{w} = f_{A}(m,n)

        &μ_{i} = m+i, i ∈ \{0,1,...,n-1\}

        &σ_{i} ∼ \mathrm{Exponential}\left(\frac{1}{σ_{iso}}\right)

        &m ∼ \mathrm{Normal}(μ_{m},σ_{m})



    With weights of normal mixture distribution :math:`\vec{w}` set by
    averagine model :math:`f_{A}`.
    Positions of normal mixture model :math:`\vec{μ}` are set
    to monoisotopic mass :math:`m` and :math:`n-1` first isotopic peaks.
    For all normal distributions of mixture the same standard deviation
    :math:`σ_{i}` is used. Hyperpriors :math:`μ_{m},σ_{m},σ_{iso}` can be
    passed to constructor. If no hyperpriors are provided, :math:`μ_{m}`
    is set to mean of observed masses. :math:`σ_{m},σ_{iso}` default to
    1 and 0.1, respectively.


    Attributes:
        mass (pm.Normal): Monoisotopic peptide mass.
        sigma (pm.Exponential): Widths of normal distributions of mixture.
        obs (pm.NormalMixture): Normal mixture distribution.

    """

    def __init__(
        self,
        mz_values: np.ndarray,
        z: int = 1,
        n: int = 6,
        μ_mass: float = None,
        σ_mass: float = 1,
        σ_iso: float = 0.1,
        name: str = "",
        model: pm.Model = None,
        syn: SyntheticPeptideFeature = None,
    ):
        """Initialize isotopic pattern model.

        Model isotopic pattern via gaussian mixture distribution.
        This model allows for inference of peptide mass and standard deviation
        of normal distributions of mixture.
        This model does not allow for inference of peptide charge
        and averagine model parameter :math:`λ_{a}`.

        Args:
            mz_values (np.ndarray): Detected mz values.
            z (int, optional): Charge of peptide. Defaults to 1.
            n (int, optional): Number of peaks to consider in model. Defaults to 6.
            μ_mass (float,optional): Position of prior normal distribution of mass.
                If not provided, mean of detected masses is taken. Defaults to None.
            σ_mass (float,optional): Standard deviation of prior normal distribution
                for mass. Defaults to 1.
            σ_iso (float,optional): Expected value of standard deviation of normal
                distributions of mixture. Defaults to 0.1.
            name (str, optional): Name of model. Defaults to "".
            model (pm.Model, optional): ?pymcmodel?. Defaults to None.
            syn (SyntheticPeptideFeature, optional): SyntheticPeptideFeature instance data is sampled from.
                Needed for model evaluation. Defaults to None.
        """
        obs_masses = mz_values * z

        if μ_mass == None:
            μ_mass = obs_masses.mean()

        super().__init__(name, model, syn)

        self.Var("mass", pm.Normal.dist(mu=μ_mass, sigma=σ_mass))
        num_iso = np.arange(n)
        masses = self.mass + num_iso
        lambda_averagine = 0.000594 * masses[0] - 0.03091
        weights = (
            pmath.exp(-lambda_averagine)
            * lambda_averagine**num_iso
            / factorial(num_iso)
        )
        weights_n = weights / pmath.sum(weights)
        self.Var("sigma", pm.Exponential.dist(lam=1 / σ_iso))

        # Likelihood
        self.Var(
            "obs",
            pm.NormalMixture.dist(w=weights_n, mu=masses, sigma=self.sigma),
            data=obs_masses,
        )


class ModelChargeIsotopicPattern(AbstractModel):
    r"""Model for Isotopic pattern.

    Fitting normal mixture distribution to mass to charge ratios (:math:`mz`).
    Instances of this class also estimate charge :math:`z` but not averagine model
    parameter :math:`λ_{a}`.


    This model allows for inference of peptide mass :math:`m`, charge :math:`z`
    and standard deviation σ of normal distributions around
    peaks. Note, that charge :math:`z` is represented by charge state :math:`z'`.

    :math:`z=z'+1`

    Model design:

    .. math::

        &obs ∼ \mathrm{NormalMixture}(\vec{w},\vec{μ},\vec{σ})

        &\vec{w} = f_{A}(mz*(z'+1))

        &μ_{i} = mz+\frac{i}{z'+1}, i ∈ \{0,1,...,n-1\}

        &σ_{i} ∼ \mathrm{Exponential}\left(1/σ_{iso}\right)

        &mz ∼ \mathrm{Normal}(μ_{m},σ_{m})

        &z' ∼ \mathrm{Poisson}(z_{start})

    With weights of normal mixture distribution :math:`\vec{w}` set by
    averagine model :math:`f_{A}`.
    Positions of normal mixture model :math:`\vec{μ}` are set
    to monoisotopic mz :math:`\frac{m}{z'+1}` and :math:`n-1` first isotopic peaks.
    For all normal distributions of mixture the same standard deviation
    :math:`σ_{i}` is used. Hyperpriors :math:`μ_{m},σ_{m},σ_{iso},z_{start}` can be
    passed to constructor. If no hyperpriors are provided, :math:`μ_{m}`
    is set to mean of observed mz values. :math:`σ_{m},σ_{iso}` default to
    1 and 0.1, respectively. :math:`z_{start}` defaults to 2.


    Attributes:
        mz (pm.Normal): Monoisotopic peptide mass to charge value.
        sigma (pm.Exponential): Widths of normal distributions of mixture.
        obs (pm.NormalMixture): Normal mixture distribution.
        charge_state (pm.Poisson): Charge state of peptide. Charge of peptide
            is charge_state+1.

    """

    def __init__(
        self,
        mz_values: np.ndarray,
        z_start: int = 2,
        n: int = 6,
        μ_mass: float = None,
        σ_mass: float = 1,
        σ_iso: float = 0.1,
        name: str = "",
        model: pm.Model = None,
        syn: SyntheticPeptideFeature = None,
    ):
        """Initialize isotopic pattern model

        This model allows for inference of peptide mass, charge and
        standard deviation of normal distributions of the normal mixture.

        Args:
            mz_values (np.ndarray): Data to fit model to.
            z_start (int, optional): Expected charge. Defaults to 2.
            n (int, optional): Number of peaks to consider. Defaults to 6.
            μ_mass (float, optional): Position of prior normal distribution for mz. If not provided,
                average of data is taken. Defaults to None.
            σ_mass (float, optional): Standard deviation of prior normal distribution for mz. Defaults to 1.
            σ_iso (float, optional): Expected value for standard deviation of normal distribution around mass peaks. Defaults to 0.1.
            name (str, optional): Name of model. Defaults to "".
            model (pm.Model, optional): ?pymc model?. Defaults to None.
            syn (SyntheticPeptideFeature, optional): SyntheticPeptideFeature instance data is sampled from.
                Needed for model evaluation. Defaults to None.
        """

        if μ_mass == None:
            μ_mass = mz_values.mean()

        super().__init__(name, model, syn)
        self.Var("charge_state", pm.Poisson.dist(mu=z_start))

        self.Var("mz", pm.Normal.dist(mu=μ_mass, sigma=σ_mass))
        num_iso = np.arange(n)
        lambda_averagine = 0.000594 * self.mz * (self.charge_state + 1) - 0.03091
        weights = (
            pmath.exp(-lambda_averagine)
            * lambda_averagine**num_iso
            / factorial(num_iso)
        )
        weights_n = weights / pmath.sum(weights)
        self.Var("sigma", pm.Exponential.dist(lam=1 / σ_iso))
        mzs = self.mz + num_iso / (self.charge_state + 1)
        # Likelihood
        self.Var(
            "obs",
            pm.NormalMixture.dist(w=weights_n, mu=mzs, sigma=self.sigma),
            data=mz_values,
        )


class ModelChargeAveragineIsotopicPattern(AbstractModel):
    r"""Model for Isotopic pattern.

    Fitting normal mixture distribution to mass to charge ratios (:math:`mz`).
    Instances of this class also estimate charge :math:`z` and averagine model
    parameter :math:`λ_{a}`.


    This model allows for inference of peptide mass :math:`m`, charge :math:`z`
    standard deviation σ of normal distributions around
    peaks and :math:`λ_{a}` of averagine model.
    Note, that charge :math:`z` is represented by charge state :math:`z'`.

    :math:`z=z'+1`

    Model design:

    .. math::

        &obs ∼ \mathrm{NormalMixture}(\vec{w},\vec{μ},\vec{σ})

        &\vec{w} = f'_{A}(mz*(z'+1),λ_{a})

        &μ_{i} = mz+\frac{i}{z'+1}, i ∈ \{0,1,...,n-1\}

        &σ_{i} ∼ \mathrm{Exponential}\left(\frac{1}{σ_{iso}}\right)

        &mz ∼ \mathrm{Normal}(μ_{m},σ_{m})

        &z' ∼ \mathrm{Poisson}(z_{start})

        &λ_{a} ∼ \mathrm{Exponential}\left(\frac{1}{\hat{λ_{a}}}\right)

    With weights of normal mixture distribution :math:`\vec{w}` set by
    averagine model :math:`f'_{A}`. Here, also taking :math:`λ_{a}` as parameter.
    Positions of normal mixture model :math:`\vec{μ}` are set
    to monoisotopic mz :math:`\frac{m}{z'+1}` and :math:`n-1` first isotopic peaks.
    For all normal distributions of mixture the same standard deviation
    :math:`σ_{i}` is used. Hyperpriors :math:`μ_{m},σ_{m},σ_{iso},z_{start},\hat{λ_{a}}`
    can be passed to constructor. If no hyperpriors are provided, :math:`μ_{m}`
    is set to mean of observed mz values. :math:`σ_{m},σ_{iso}` default to
    1 and 0.1, respectively. :math:`z_{start}` defaults to 2. :math:`\hat{λ_{a}}`
    defaults to 0.01.


    Attributes:
        mz (pm.Normal): Monoisotopic peptide mass to charge ratio.
        sigma (pm.Exponential): Widths of normal distributions of mixture.
        obs (pm.NormalMixture): Normal mixture distribution.
        charge_state (pm.Poisson): Charge state of peptide. Charge of peptide
            is charge_state+1.
        λ_a (pm.Exponential): Factor of averagine model. (:math:`λ=λ_{a}*m+0.03091`)

    """

    def __init__(
        self,
        mz_values: np.ndarray,
        z_start: int = 2,
        n: int = 6,
        μ_mass: float = None,
        σ_mass: float = 1,
        σ_iso: float = 0.1,
        λ_a_hat=0.01,
        name: str = "",
        model: pm.Model = None,
        syn: SyntheticPeptideFeature = None,
    ):
        """Initialize isotopic pattern model.

        This model allows for inference of peptide mass, charge,
        standard deviation of normal distributions of normal mixture and
        averagine parameter :math:`λ_{a}`

        Args:
            mz_values (np.ndarray): Data to fit model to.
            z_start (int, optional): Expectation of charge state. Defaults to 2.
            n (int, optional): Number of peaks to consider. Defaults to 6.
            μ_mass (float, optional): Position of prior normal distribution for mz. If not provided,
                average of data is taken. Defaults to None.
            σ_mass (float, optional): Standard deviation of prior normal distribution for mz. Defaults to 1.
            σ_iso (float, optional): Expected value for standard deviation of normal distribution around mass peaks. Defaults to 0.1.
            λ_a_hat (float, optional): Expected value for λ_a. Defaults to 0.01.
            name (str, optional): Name of model. Defaults to "".
            model (pm.Model, optional): ?Pymc model?. Defaults to None.
            syn (SyntheticPeptideFeature, optional): SyntheticPeptideFeature instance data is sampled from.
                Needed for model evaluation. Defaults to None.
        """

        if μ_mass == None:
            μ_mass = mz_values.mean()

        super().__init__(name, model, syn)
        self.Var("charge_state", pm.Poisson.dist(mu=z_start))

        self.Var("mz", pm.Normal.dist(mu=μ_mass, sigma=σ_mass))
        num_iso = np.arange(n)
        self.Var("λ_a", pm.Exponential.dist(lam=1 / λ_a_hat))
        lambda_averagine = self.λ_a * self.mz * (self.charge_state + 1) - 0.03091
        weights = (
            pmath.exp(-lambda_averagine)
            * lambda_averagine**num_iso
            / factorial(num_iso)
        )
        weights_n = weights / pmath.sum(weights)
        self.Var("sigma", pm.Exponential.dist(lam=1 / σ_iso))
        mzs = self.mz + num_iso / (self.charge_state + 1)
        # Likelihood
        self.Var(
            "obs",
            pm.NormalMixture.dist(w=weights_n, mu=mzs, sigma=self.sigma),
            data=mz_values,
        )


class ModelChargePoissonIsotopicPattern(AbstractModel):
    r"""Model for Isotopic pattern.

    Fitting normal mixture distribution to mass to charge ratios (:math:`mz`).
    Instances of this class also estimate charge :math:`z` and expected/average
    count of isotopes in molecule :math:`λ`.

    This model allows for inference of peptide mass :math:`m`, charge :math:`z`
    standard deviation σ of normal distributions around
    peaks and :math:`λ`, the average count of isotopic atoms inside the molecule
    behind the observed feature.
    Note, that charge :math:`z` is represented by charge state :math:`z'`.

    :math:`z=z'+1`

    Model design:

    .. math::

        &obs ∼ \mathrm{NormalMixture}(\vec{w},\vec{μ},\vec{σ})

        &w_{i} = \frac{\mathrm{PoissonPDF(i|λ)}}{\sum_{j=0}^{n-1}\mathrm{PoissonPDF(j|λ)}}, i ∈ \{0,1,...,n-1\}

        &μ_{i} = mz+\frac{i}{z'+1}, i ∈ \{0,1,...,n-1\}

        &σ_{i} ∼ \mathrm{Exponential}\left(\frac{1}{σ_{iso}}\right)

        &mz ∼ \mathrm{Normal}(μ_{m},σ_{m})

        &z' ∼ \mathrm{Poisson}(z_{start})

        &λ ∼ \mathrm{Exponential}\left(\frac{1}{\hat{λ}}\right)

    Positions of normal mixture model :math:`\vec{μ}` are set
    to monoisotopic mz :math:`\frac{m}{z'+1}` and :math:`n-1` first isotopic peaks.
    For all normal distributions of mixture the same standard deviation
    :math:`σ_{i}` is used. Hyperpriors :math:`μ_{m},σ_{m},σ_{iso},z_{start},\hat{λ}`
    can be passed to constructor. If no hyperpriors are provided, :math:`μ_{m}`
    is set to mean of observed mz values. :math:`σ_{m},σ_{iso}` default to
    1 and 0.1, respectively. :math:`z_{start}` defaults to 2. :math:`\hat{λ}`
    defaults to averagine assumption :math:`0.000594 * μ_{m}*(z_{start}+1) - 0.03091`.


    Attributes:
        mz (pm.Normal): Monoisotopic peptide mass to charge ratio.
        sigma (pm.Exponential): Widths of normal distributions of mixture.
        obs (pm.NormalMixture): Normal mixture distribution. Model for observed
            mass to charge ratios.
        charge_state (pm.Poisson): Charge state of peptide. Charge of peptide
            is charge_state+1.
        λ (pm.Exponential): λ of Poisson distribution behind normal mixture weights.
            Expected isotopic atoms in molecule.

    """

    def __init__(
        self,
        mz_values: np.ndarray,
        z_start: int = 2,
        n: int = 6,
        μ_mass: float = None,
        σ_mass: float = 1,
        σ_iso: float = 0.1,
        λ_hat=None,
        name: str = "",
        model: pm.Model = None,
        syn: SyntheticPeptideFeature = None,
    ):
        """Initialize poisson isotopic pattern model.

        This model allows for inference of peptide mass, charge,
        standard deviation of normal distributions of normal mixture and
        expected/average count of isotopic atoms :math:`λ`

        Args:
            mz_values (np.ndarray): Data to fit model to.
            z_start (int, optional): Expectation of charge state. Defaults to 2.
            n (int, optional): Number of peaks to consider. Defaults to 6.
            μ_mass (float,optional): Position of prior normal distribution for mz. If not provided,
                average of data is taken. Defaults to None.
            σ_mass (float,optional): Standard deviation of prior normal distribution for mz. Defaults to 1.
            σ_iso (float,optional): Expected value for standard deviation of normal distribution around mass peaks. Defaults to 0.1.
            λ_hat (float,optional): Expected value for λ. Defaults to :math:`0.000594 * μ_{m}*(z_{start}+1) - 0.03091`.
            name (str, optional): Name of model. Defaults to "".
            model (pm.Model, optional): ?Pymc model?. Defaults to None.
            syn (SyntheticPeptideFeature, optional): SyntheticPeptideFeature instance data is sampled from.
                Needed for model evaluation. Defaults to None.
        """

        if μ_mass == None:
            μ_mass = mz_values.mean()
        if λ_hat == None:
            # default lambda_hat is set to averagine λ
            λ_hat = 0.000594 * μ_mass * (z_start + 1) - 0.03091
        super().__init__(name, model, syn)
        self.Var("charge_state", pm.Poisson.dist(mu=z_start))

        self.Var("mz", pm.Normal.dist(mu=μ_mass, sigma=σ_mass))
        num_iso = np.arange(n)
        self.Var("λ", pm.Exponential.dist(lam=1 / λ_hat))

        weights = pm.Poisson.dist(mu=self.λ).logp(num_iso)

        # equal to:
        # weights = pmath.exp(-self.λ)*self.λ**num_iso/factorial(num_iso)

        weights_n = weights / pmath.sum(weights)
        self.Var("sigma", pm.Exponential.dist(lam=1 / σ_iso))
        mzs = self.mz + num_iso / (self.charge_state + 1)
        # Likelihood
        self.Var(
            "obs",
            pm.NormalMixture.dist(w=weights_n, mu=mzs, sigma=self.sigma),
            data=mz_values,
        )
