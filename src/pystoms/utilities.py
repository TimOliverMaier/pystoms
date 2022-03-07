from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as sps
from pystoms.synthetics import SyntheticPeptideFeature
from pystoms import models
import pandas as pd
import os
from numpy.typing import ArrayLike
import seaborn as sns


class CategoryDist():
    """Categorial distribution

    Instances of this class allow for sampling
    from categorial distributions.

    Attributes:
        categories: List or array with all categories. Order is
            important.
        probabilites: List or array with probabilites. Probability
            at position i corresponds to category at position i.
    Examples:
        >>>my_fair_coin = CategoryDist(["Coin","Head"],[0.5,0.5])
        >>>print(my_fair_coin.rvs())
        "Head"
        
    """
    def __init__(self,categories:ArrayLike,probabilites:ArrayLike):
        """Inits categorical distribution.

        Args:
            categories (ArrayLike): Array with categories as strings.
            probabilites (ArrayLike): Array with corresponding
                probabilites.
        Raises:
            ValueError if length of categories and probabilities
                not the same, or probabilities do not sum to 1.
        """
        if sum(probabilites) != 1:
            raise ValueError("Probabilities do not sum to 1")
        if len(probabilites) != len(categories):
            raise ValueError("Not as many categories provided as probabilites")

        self.categories = categories
        self.probabilites = probabilites

    def rvs(self,n:int=1):
        """Sampling from categorical distribution.

        Sampling with replacement.

        Args:
            n (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
            np.ndarray or scalar: Draws from distribution.
        """
        sample = np.random.choice(self.categories,n,True,self.probabilites)
        if n == 1:
            sample = sample[0]
            
        return sample

class SyntheticsSet():
    """Wrapper for handling a set of synthetic Peptide Features and models.

    Stores synthetics in a list and assigns models to them. Via the evaluation
    method this class can give insights in model robustness over a large variety
    of synthetic feature data.

    Attributes:
        SetName: Name of the synthetics set. 
        SPF_args_stats: Dictionary of syntheticPeptideFature parameters
            and their distributions, if they are considered a random variable.
        syns: List of synthetic peptide features.
        syn_names: List of names of synthetic features.
        modelsRT (after assignModels call): Assigned retention time models.
        modelsMZ (after assignModels call): Assigned isotopic distribution models.
        summariesRT (after evaluateAll call): List of retention time model summaries.
        summariesMZ (after evaluateAll call): List of isotopic pattern model summaries.
    """


    # This class variable stores default settings of 
    # SyntheticsSet. All instances of 
    # pySTOMS.synthetics.SyntheticPeptideFeature
    # within a SyntheticSet are either passed fixed or
    # random parameters, drawn from a specified distribution.
    # Wether a parameter is random by default is stored in
    # first position of tuples in SyntheticsSet._Default_SPF_args_stats.
    # The fixed default parameter or default distribution is stored
    # in the second position.

    _Default_SPF_args_stats = {
                    "mass":(True,sps.uniform(loc=200,scale=2000)),
                    "charge":(True,sps.randint(low=1,high=6)),
                    "emg_μ":(True,sps.uniform(1,100)),
                    "emg_σ":(True,sps.norm(1,0.5)),
                    "emg_λ":(True,sps.norm(1,0.5)),
                    "elution_noise":(False,0),
                    "isotopic_noise":(False,0),
                    "num_peaks":(False,6),
                    "isotopic_sigma":(True,sps.uniform(0.01,0.05)),
                    "scan_intervall":(False,0.1),
                    "ms_resolution":(False,10),
                    "isotopic_pattern_model":(True,CategoryDist(["averagine","non_averagine"],[0.7,0.3]))
    }

    def __init__(self,SetName:str,num_syns:int=20,custom_args_stats:dict=None,**kwargs):
        """Inits a set of synthetic peptide features.

        Args:
            SetName (str): Name of Set.
            num_syns (int, optional): Number of `pySTOMS.synthetics.SyntheticPeptideFeature``
                to create in set. Defaults to 20.
            custom_args_stats (dict, optional): A dictionary like 'SyntheticsSet._Default_SPF_args_stats`.
                Here non-default scipy distribution instances can be defined for parametes. Defaults to None.
            kwargs: Fixed parmeters for created synthetics.
        """
        self.SetName = SetName
        # copy from default values
        self.SPF_args_stats = SyntheticsSet._Default_SPF_args_stats.copy()
        # if custom dictionary was provided, update defaults
        # This option exists to allow for passing distributions for several parameters
        if custom_args_stats != None:
            self.SPF_args_stats.update(custom_args_stats)
        # now if fixed parameters were passed via kwargs update dictionary again
        for key,value in kwargs.items():
            self.SPF_args_stats[key]=(False,value)
        # create array with num_syns synthetic peptide features
        self.syns = np.zeros(num_syns,dtype=SyntheticPeptideFeature)
        self.syn_names = []
        for i in range(num_syns):
            # draw parameters that are random
            params = self._getSynParams()
            self.syns[i] = SyntheticPeptideFeature(**params)
            self.syn_names.append(SetName+"_"+str(i))
        

    def _getSynParams(self):
        """Generating parameter dictionary to pass to
            pySTOMS.synthetics.SyntheticPeptideFeature

        Returns:
            dictionary: Dictionary with parameters of 
                pySTOMS.synthetics.SyntheticPeptideFeature
                as key and their value as values.
        """
        # go through updated parameter distribution dictionary
        # and store fixed or drawn values in params
        # with name of parameter as key
        params = {}
        for arg,value in self.SPF_args_stats.items():
            if value[0]:
                # Parameter is random
                params[arg] = value[1].rvs()
            else:
                # Parameter is fixed
                params[arg] = value[1]
        return params


    def assignModels(self,sample_size=1000,rt_model:str="ModelEMG",mz_model:str="ModelChargeAveragineIsotopicPattern"):
        """Assigning stochastic models to synthetic features.

        Args:
            sample_size (int, optional): Size of sample drawn from synthetics for
                model fit. Defaults to 1000.
            rt_model (str, optional): Model to use for retention times. Defaults to "ModelEMG".
            mz_model (str, optional): Model to use for isotopic pattern. Defaults to "ModelChargeAveragineIsotopicPattern".

        Raises:
            NotImplementedError: If provided model not known or not yet supported
        """
        
        print("Assigning models to synthetics...")

        # First we need to sample data from synthetics for fit:
        print(f"Sampling {sample_size} data points from synthetics.")
        samples = [syn.sample2D(sample_size) for syn in self.syns]
        print("Samping done.")

        # Then we initialize bayesian models for each sample
        print("Initializing models.")
        zipper = zip(samples,self.syns,self.syn_names)
        
        # Retention time models
        print("Initialize RT models")
        if rt_model == "ModelEMG":
            self.modelsRT = [models.ModelEMG(s["Retention_Time_(min)"],syn = o,name=n+"_"+rt_model) for s,o,n in zipper]
        else:
            raise NotImplementedError("Retention Time Model not supported")
        print("RT models done.")

        # Isotopic pattern models
        # rebuild of iterator necessary
        zipper = zip(samples,self.syns,self.syn_names)
        print("Initialize MZ models")
        if mz_model =="ModelChargeAveragineIsotopicPattern":
            self.modelsMZ = [models.ModelChargeAveragineIsotopicPattern(s["m/z"],syn=o,name=n+"_"+mz_model) for s,o,n in zipper]
        elif mz_model == "ModelChargePoissonIsotopicPattern":
            self.modelsMZ = [models.ModelChargePoissonIsotopicPattern(s["m/z"],syn=o,name=n+"_"+mz_model) for s,o,n in zipper]
        elif mz_model == "ModelChargeIsotopicPattern":
            self.modelsMZ = [models.ModelChargeIsotopicPattern(s["m/z"],syn=o,name=n+"_"+mz_model) for s,o,n in zipper]
        elif mz_model == "ModelIsotopicPattern": 
            self.modelsMZ = [models.ModelIsotopicPattern(s["m/z"],z=o.charge,syn=o,name=n+"_"+mz_model) for s,o,n in zipper]
        else:
            raise NotImplementedError("Isotopic Model not supported")
        print("MZ models done.")


    def evaluateAll(self,prior_pred:bool=False,post_pred:bool=False,post_sample_n:int=100,save_fig:bool=False,path_output_dir:str="Evaluation_Set"):
        """Evaluate models assigned to synthetics in set.

        Args:
            prior_pred (bool, optional): If True, prior prediction check
                is performed. Defaults to False.
            post_pred (bool, optional): If True, posterior prediction check
                is performed. Defaults to False.
            post_sample_n (int, optional): Number of samples to draw (per chain) for posterior prediction check.
                Defaults to 100.
            save_fig (bool, optional): If True write evaluation to file, else write to console.
                Defaults to False.
            path_output_dir (str, optional): Path to Folder for output if save_fig==True.
                Defaults to "Evaluation_Set".
        """

        # evaluation method of models returns arviz summary of 
        # model fit. Those are stored in below lists to display
        # SyntheticsSet summary.
        self.summariesRT = []
        self.summariesMZ = []

        for modelRT,modelMZ,syn_name,syn in zip(self.modelsRT,self.modelsMZ,self.syn_names,self.syns):
            # output path for current synthetic, make directory if necessary
            path = path_output_dir+"/"+self.SetName+"/"+syn_name
            if save_fig and not os.path.exists(path):
                os.makedirs(path)
            # show summary plots of synthetic peptide feature
            syn.show(save_fig,path)
            # evaluation
            try:
                self.summariesRT.append(modelRT.evaluation(prior_pred,post_pred,post_sample_n,save_fig,path))
            except:
                print(f"There was an error in evaluation of retention time model for {syn_name}")
            try:
                self.summariesMZ.append(modelMZ.evaluation(prior_pred,post_pred,post_sample_n,save_fig,path))
            except:
                print(f"There was an error in evaluation of isotopic model for {syn_name}")
        # show or print statistic summaries
        self._showSummary(save_fig,path_output_dir)

    def _showSummary(self,save_fig:bool=False,path_output_dir:str="Evaluation_Set"):
        """Plotting SyntheticsSet summary.

        Args:
            save_fig (bool, optional): If True, plots are exported to png.
                Defaults to False.
            path_output_dir (str, optional): Output path.
                Defaults to "Evaluation_Set".
        """
        if save_fig:
            path = path_output_dir+"/"+self.SetName+"/"
            if not os.path.exists(path):
                os.makedirs(path)

        # bring summaries of each model together
        # to two (Retention Time models and Isotopic models)
        # big dataframes       
        summariesRT_conc = pd.concat(self.summariesRT)
        summariesMZ_conc = pd.concat(self.summariesMZ)

        # plotting deviations to mean and position 
        # of reference values in posteriors
        # for retention time models
        summariesRT_conc_view = summariesRT_conc.pivot(index="model_name",columns="variable",values=["ref_pos_(%)","deviation_(%)"])
        devRT = sns.pairplot(summariesRT_conc_view["deviation_(%)"])
        devRT.figure.suptitle("Deviations")
        devRT.figure.supxlabel("Deviation to Mean (%)")
        devRT.figure.supylabel("Deviation to Mean (%)")
        devRT.figure.tight_layout()
        if save_fig:
            plt.savefig(path+"deviationsRT.png",dpi=300)
            plt.close()
        else:
            plt.show()

        posRT = sns.pairplot(summariesRT_conc_view["ref_pos_(%)"])
        posRT.figure.suptitle("Positions of Reference Values in Posterior")
        posRT.figure.supxlabel("Values \u2264 Ref. (%)")
        posRT.figure.supylabel("Values \u2264 Ref. (%)")
        posRT.figure.tight_layout()
        if save_fig:
            plt.savefig(path+"positionsRT.png",dpi=300)
            plt.close()
        else:
            plt.show()
        
        # plotting deviations to mean and position
        # of reference values in posteriors
        # for isotopic models
        summariesMZ_conc_view = summariesMZ_conc.pivot(index="model_name",columns="variable",values=["ref_pos_(%)","deviation_(%)"])
        # this needs to be improved if furtgher discrete vars are added to models
        devMZ = sns.pairplot(summariesMZ_conc_view["deviation_(%)"].drop(columns="charge_state"))
        devMZ.figure.suptitle("Deviations")
        devMZ.figure.supxlabel("Deviation to Mean (%)")
        devMZ.figure.supylabel("Deviation to Mean (%)")
        devMZ.figure.tight_layout()
        if save_fig:
            plt.savefig(path+"deviationsMZ.png",dpi=300)
            plt.close()
        else:
            plt.show()

        posMZ = sns.pairplot(summariesMZ_conc_view["ref_pos_(%)"])
        posMZ.figure.suptitle("Positions of Reference Values in Posterior")
        posMZ.figure.supxlabel("Values \u2264 Ref. (%)")
        posMZ.figure.supylabel("Values \u2264 Ref. (%)")
        posMZ.figure.tight_layout()
        if save_fig:
            plt.savefig(path+"positionsMZ.png",dpi=300)
            plt.close()
        else:
            plt.show()

