import numpy as np
from scipy.special import factorial 
import matplotlib.pyplot as plt
from scipy.stats import exponnorm,norm, rv_continuous
import pandas as pd
import os
class Isotopic_Averagine_Distribution(rv_continuous):
    r"""Isotopic pattern distribution according to averagine model
    

    Subclass of scipy.rv_continuous with custom _pdf function. 
    Isotopic distribution is modeled as a gaussian mixture of a set of n
    normal distributions around the monoisotopic and isotopic mass to charge values:
    The weights are determined by averagine model. 

    :math:`f(x)=\frac{1}{\sqrt{2\pi}\sigma}\sum_{i=1}^{n}w_i e^{-0.5(\frac{x-\mu_i}{\sigma})^2}`


    This class is instantiated without any arguments:

    >>>my_dist = Isotopic_Averagine_Distribution()
    
    Parameters of distribution are given to pdf function:

    >>>pdf_values = my_dist.pdf(x:np.ndarray,mass:float,charge:int,sigma:float,num_peaks:int)
    
    The pdf method of scipy.rv_continuous is then calling customized internal _pdf method, which returns
    array of pdf evaluations at positions in input array x.

    Attributes:
        See scipy.continuous
        
    """
    def __init__(self,averagine_style):
        super().__init__()
        self.averagine_style=averagine_style
    
    def _pdf(self,x:np.ndarray,mass:np.ndarray,charge:np.ndarray,sigma:np.ndarray,num_peaks:np.ndarray) -> np.ndarray:
        """Calculates probability density function (PDF)

        Overwrites scipy.rv_continuous._pdf. Is internally called by Isotopic_Averagine_Distribution.pdf
        method. Calculates PDF for given position and parameters by evaluating and adding pdf of num_peaks
        weighted normal distributions. Means of these normal distributions are
        [mass/charge,(mass+1)/charge,...,(mass+num_peaks-1)/charge]. Sigmas are sigma for all 
        distributions. Weights of distributions are calculated based on averagine model. Number of 
        peaks to consider is set by num_peaks.

        Args:
            x: Positions to calculate PDF at. Numpy array.
            mass: Monoisotopic mass of the peptide [Da]. Given to ._pdf as np.ndarray 
                of length=1 by .pdf method.
            charge: Charge of peptide. Given to ._pdf as np.ndarray 
                of length=1 by .pdf method.
            sigma: Standard deviation of gauss bells. Given to ._pdf as np.ndarray 
                of length=1 by .pdf method.
            num_peaks: Number of peaks to consider. Given to ._pdf as np.ndarray 
                of length=1 by .pdf method.
            
        Returns:
            np.ndarray. Evaluations of PDF at positions in x.
      
        Raises:
            ValueError if self.averagine_style stores unsupported averagine style.
            Supported are "averagine" and "non_averagine".
        """
        # means of normal distributions are [mass/charge,(mass+1)/charge,...,(mass+num_peaks-1)/charge]
        means = (np.repeat(mass,num_peaks)+np.arange(num_peaks))/charge
        # sigmas of normal distributions are sigma for all
        sigmas = np.repeat(sigma,num_peaks)
        # weights are calculated by averagine model depending on mass and num_peaks
        if self.averagine_style == "averagine":
            weights = self._averagine_isotopic(mass,num_peaks)
        # not averagine like
        elif self.averagine_style == "non_averagine":
            weights = self._non_averagine_isotopic(mass,num_peaks)
        else:
            raise ValueError("Averagine style not supported")
        # calculation of pdf(x)
        x_pdf = np.zeros(len(x))
        # evaluate position array for each normal dist. w*N(μ,σ) and add up
        for μ,w,σ in zip(means,weights,sigmas):
            x_pdf += w*1/(σ*np.sqrt(2*np.pi))*np.exp(-0.5*((x-μ)/σ)**2)
        return x_pdf   

    @staticmethod
    def _averagine_isotopic(mass:np.ndarray,num_peaks:np.ndarray):
        """Calculates weights for isotopic distribution
        
        Calculates weights for isotopic pattern (gaussian mixture)
        distribution based on averagine model and normalization.

        Args:
            mass: Monoisotopic mass of peptide. 
            num_peaks: Number of considered peaks. 
        Returns:
            np.ndarray. Array of weights. 
        Raises:
            None. 
        """
        #averagine approx. Adopted from Hildebrandt Github
        λ = 0.000594 * mass - 0.03091 
        n = num_peaks[0]
        iso_w = np.fromiter((np.exp(-λ)*np.power(λ,k)/factorial(k) for k in range(n)),float) 
        # normalization
        iso_w_norm = iso_w/iso_w.sum()
        return iso_w_norm
    
    @staticmethod
    def _non_averagine_isotopic(mass:np.ndarray,num_peaks:np.ndarray):
        """Calculates weights for isotopic distribution
        
        Decoy method. Returns averagine weights inversed.

        Args:
            mass: Monoisotopic mass of peptide. 
            num_peaks: Number of considered peaks. 
        Returns:
            np.ndarray. Array of weights. 
        Raises:
            None. 
        """
        #averagine approx. Adopted from Hildebrandt Github
        λ = 0.000594 * mass - 0.03091 
        n = num_peaks[0]
        iso_w = np.fromiter((np.exp(-λ)*np.power(λ,k)/factorial(k) for k in range(n)),float) 
        # normalization
        iso_w_norm = iso_w/iso_w.sum()
        return np.flip(iso_w_norm,0)


class SyntheticPeptideFeature():
    r"""Synthetic peptide feature generator.
    

    Generates and stores synthetic peptide feature data.
    Each instance of this class holds synthetic feature
    data for a single peptide.

    Elution profile of peptide is represented by an
    exponentially modified gaussian (EMG) distribution with a
    normally distributed additive noise.

    Isotopic pattern distribution is represented by
    Isotopic_Averagine_Distribution (IAD) with a normally
    distributed additive noise.

    Synthetic Data is generated in 4D format for a range of
    retention times (rt) and mass to charge values (mz):
    (rt, elution-curve(rt), mz, intensity)

    Elution curve column can be discarded to yield typical 3D
    format. Intensity is calculated by function :math:`I(rt,mz)`:
    .. math::

        I(rt,mz) &= (EMG(rt)+X_{EMG})*(IAD(mz)+X_{IAD})

        X_{EMG} &∼ N(0,elution_noise)

        X_{IAD} &∼ N(0,isotopic_noise)


    If correctBaseline set to True, all intensities below zero are
    set to zero.
    

    Attributes:
        mass: Monoisotopic mass of the peptide [Da].
        charge: Charge of peptide.
        emg_μ: Average retention time [min] of the peptide.
            Corresponds to μ parameter of exponentially
            mofidied gauss distribution (EMG).
        emg_σ: Width of elution profile. Corresponds to
            σ of EMG.
        emg_λ: Describes tailing of elution profile.
            Corresponds to λ of EMG.
        elution_noise: Noise on EMG curve. Defaults to 0.01.
            Standard deviation of normally distributed random variable with 
            mean=0, realizations of this RV are added to EMG(t) for each
            timepoint t.
        isotopic_noise: Noise on isotopic pattern. Defaults to 0.01. 
            Standard deviation of normally distributed random variable with 
            mean=0, realizations of this RV are added to isotopic_distribution(mz) for each
            mass to charge ratio m/z.
        n: Number of peaks in isotopic pattern to consider.
        isotopic_sigma: Standard deviation of isotopic peaks.
            Defaults to 0.05.
        scan_intervall: Time in minutes between two MS1 scans.
            Defaults to 6 s (0.1 min).
        resolution: Resolution of mass spectrometer in bins per Dalton. 
            Defaults to 10 bins/Da
        model: Model for generation of isotopic pattern.
            Defaults to Averagine Model.
        correctBaseline: If True, all intensity values <0 are set to 0.
        feature_data: Pandas DataFrame with synthetic feature data:
            columns: ["Retention_Time_(min)","Elution_Profile(t)","m/z","Intensity"] 
    """

    # list of currently supported models for isotopic distribution
    _supported_models = ["averagine","non_averagine"]
    
    def __init__(self,mass:float,charge:int,emg_μ:float,emg_σ:float,emg_λ:float,elution_noise:float=0.01,isotopic_noise:float=0.01,num_peaks:int=6,isotopic_sigma:float=0.05,scan_intervall:float=0.1,ms_resolution:int=10,isotopic_pattern_model:str="averagine",correctBaseline:bool=True):
        """Inits SyntheticPeptideFeature class.

    Inits a synthetic peptide feature with peptide mass, charge and 
    elution profile parameters. Calls internal _generator method
    to generate synthetic feature data and stores it in feature_data.

    Args:
        mass: Monoisotopic mass of the peptide [Da].
        charge: Charge of peptide.
        emg_μ: Average retention time [min] of the peptide.
            Corresponds to μ parameter of exponentially
            mofidied gauss distribution (EMG).
        emg_σ: Width of elution profile. Corresponds to
            σ of EMG.
        emg_λ: Describes tailing of elution profile.
            Corresponds to λ of EMG.
        elution_noise: Noise on EMG curve. Defaults to 0.01.
            Standard deviation of normally distributed random variable with 
            mean=0, realizations of this RV are added to EMG(t) for each
            timepoint t.
        isotopic_noise: Noise on isotopic pattern. Defaults to 0.01. 
            Standard deviation of normally distributed random variable with 
            mean=0, realizations of this RV are added to isotopic_distribution(mz) for each
            mass to charge ratio m/z.
        num_peaks: Number of peaks in isotopic pattern to consider.
        isotopic_sigma: Standard deviation of isotopic peaks.
            Defaults to 0.05.
        scan_intervall: Time in minutes between two MS1 scans.
            Defaults to 6 s (0.1 min).
        ms_resolution: Resolution of mass spectrometer in bins per Dalton. 
            Defaults to 10 bins/Da
        isotopic_pattern_model: Model for generation of isotopic pattern.
            Defaults to Averagine Model.
        correctBaseline: If True all negativ intensities (due to noise simulation)
            are set to 0.
    Returns:
      None
      
    Raises:
      ValueError if model for isotopic pattern is unkown or unsupported
        """
        self.mass = mass
        self.charge = charge

        self.emg_μ = emg_μ
        self.emg_σ = emg_σ
        self.emg_λ = emg_λ
        self.elution_noise = elution_noise
        
        self.n = num_peaks

        self.scan_intervall = scan_intervall
        self.resolution = ms_resolution

        if isotopic_pattern_model not in SyntheticPeptideFeature._supported_models:
            raise ValueError("Unkown/unsupported isotopic pattern model)")
        
        self.model=isotopic_pattern_model
        self.isotopic_noise = isotopic_noise
        self.isotopic_sigma = isotopic_sigma
        self.correctBaseline = correctBaseline
        self.feature_data = self._generate()
        
    def _generate(self):
        """Generates synthetic feature data.

    Generates synthetic peptide feature data based on 
    chosen model for isotopic pattern, peptide mass, peptide charge,
    elution profile parameters and noise parameters.
    
    Args:
        None
    Returns:
        pandas dataframe with columns: ["Retention_Time_(min)","Elution_Profile(t)","m/z","Intensity"]

    Raises:
        ValueError if model for isotopic distribution is not implemented yet
        """
        # EMG for elution curve
        K = 1/(self.emg_σ*self.emg_λ)
        emg = exponnorm(K=K,loc=self.emg_μ,scale=self.emg_σ)
        # from 1 percentile to 99 percentile of EMG have MS1 scans every scan_intevall seconds
        emg_frames = np.arange(emg.ppf(0.01),emg.ppf(0.99),self.scan_intervall)
        total_scans = len(emg_frames)
        # emg_frames are retention timepoints with measured MS1, emg_values are 
        # values of emg function at that timepoint
        emg_values = emg.pdf(emg_frames)


        # normally dist. noise on EMG
        # sample additive noise for each emg_value
        elution_noise_vars = norm(loc=0,scale=self.elution_noise).rvs(size=total_scans)
        # add on emg_values and set to zero if <0 (unless correctBaseline is undesired and set to false)
        tmp_elution = emg_values + elution_noise_vars
        if self.correctBaseline:
            elution_profile = np.where(tmp_elution>0,tmp_elution,0)
        else:
            elution_profile = tmp_elution
        
        # isotopic distribution
        # model is averagine or non averagine
        if self.model == "averagine" or self.model == "non_averagine":
            
            
            iso = Isotopic_Averagine_Distribution(self.model)
            # from first m/z peak -1 to last m/z peak +1
            m_z_axis = np.arange(self.mass/self.charge-1,(self.mass+self.n-1)/self.charge+1+1/self.resolution,1/self.resolution)
            m_z_intensity = iso.pdf(m_z_axis,self.mass,self.charge,self.isotopic_sigma,self.n)
            measurements_per_frame = len(m_z_axis)
            
            # for each rt timepoint draw new noise samples and add them on isotopic dist.
            # then multiply with elution profile at this timepoint
            data3D = np.zeros((total_scans,measurements_per_frame))
            for frame_idx,ep_value in enumerate(elution_profile):
                isotopic_noise_vars = norm(loc=0,scale=self.isotopic_noise).rvs(size=measurements_per_frame)
                tmp_iso = (m_z_intensity+isotopic_noise_vars)*ep_value

                # correct <0 values
                if self.correctBaseline:
                    data3D[frame_idx] = np.where(tmp_iso>0,tmp_iso,0)
                else:
                    data3D[frame_idx] = tmp_iso
            
            df = pd.DataFrame(data3D,columns=m_z_axis)
            
            df.insert(0,"Elution_Profile(t)",elution_profile)
            df.insert(0,"Retention_Time_(min)",emg_frames)
            df_melted = df.melt(id_vars=["Retention_Time_(min)","Elution_Profile(t)"],var_name="m/z",value_name="Intensity")
            return df_melted

        else:
            raise ValueError("Unkown/unsupported isotopic pattern model)")
        
          
    def export(self,path:str,summary:bool = True):
        """Exports feature data and plots summary.

    Exports feature data to csv file in path. If summary is set to True, this method
    summarizes peptide feature data by several plots (e.g. elution profile plot, 
    sum of all intensities at given retention time and m/z value, respectively).
    
    Args:
        path: Path to csv output file.
        summary: If True, summary plots are printed. Defaults to
        True.
    Returns:
        None
    Raises:
        None
        """
        # export data to csv file without index column
        self.feature_data.to_csv(path,index=False)

        if not summary:
            return
        self.show()

    def show(self,save_fig:bool=False,path_output_dir:str="Synthetic"):
        """Shows/plots feature data
        Args:
            save_fig (bool,optiona): If True, figures are not shown, but exported to png.
            path_output_dir (str,optional): Relative path of folder for output.
        Return:
            None
        Raises:
            None
        """
        path = path_output_dir+"/"
        if save_fig and not os.path.exists(path):
                os.makedirs(path)
        # ---------- summary plots ----------
        # unmelt dataframe to plot elution profile vs retention time
        temp = self.feature_data.pivot(index=["Retention_Time_(min)","Elution_Profile(t)"],columns="m/z",values="Intensity").reset_index()
        temp.plot.line(x="Retention_Time_(min)",y="Elution_Profile(t)",legend=None)
        plt.xlabel("Retention Time [min]")
        plt.ylabel("Elution Profile")
        if save_fig:
            plt.savefig(path+"Elution_Profile.png",dpi=300)
            plt.close()
        else:
            plt.show()
        # sum of all intensities per retention time
        temp2 = self.feature_data.groupby(["Retention_Time_(min)"]).sum().reset_index()
        temp2.plot.line(x="Retention_Time_(min)",y="Intensity",legend=None)
        plt.xlabel("Retention Time [min]")
        plt.ylabel("Sum of all Intensities")
        if save_fig:
            plt.savefig(path+"SumMZIntensities.png",dpi=300)
            plt.close()
        else:
            plt.show()
        # sum of all intensities per m/z
        temp3 = self.feature_data.groupby(["m/z"]).sum().reset_index()
        temp3.plot.line(x="m/z",y="Intensity",legend=None)
        plt.xlabel("m/z")
        plt.ylabel("Sum of all Intensities")
        if save_fig:
            plt.savefig(path+"SumRTIntensities.png",dpi=300)
            plt.close()
        else:
            plt.show()

        # plot all MS1 spectra
        for idx,row in temp.iterrows():
            plt.plot(row[2:])
        plt.xlabel("m/z")
        plt.ylabel("Intensity")
        plt.title("All MS1 Spectra")
        if save_fig:
            plt.savefig(path+"allspectra.png",dpi=300)
            plt.close()
        else:
            plt.show()

    def sample2D(self,size:int,writeToCsv:bool=False,path:str="sample3D.csv"):
        """Samples 2D tuples from feature data.
        
        Samples (Retention Time, m/z value) tuples from feature data DataFrame.
        Samples with replacement, weighted by intensity value.
        
        Args:
            size: Size of sample.
            writeToCsv: If True sampled data is written to csv.
            path: Path for csv output
        Returns:
            Dataframe with size (RT,M/Z) samples.
        """

        df = self.feature_data.sample(n=size,replace=True,weights="Intensity",ignore_index=True)[["Retention_Time_(min)","m/z"]]
        if writeToCsv:
            df.to_csv(path,index=False)
        return df

class FeatureData:
    """Loads and stores synthetic and experimental peptide feature data.

    This class stores feature 3D data (experimental or synthetic) and
    allows sampling 2D data from it. 
    
    
    Attributes:
        path: Path to input csv of synthetic or 
            experimental feature (columns = ["Retention_Time_(min)","m/z","Intensity"])
        data3D: Pandas Dataframe of loaded data, 
            columns = ["Retention_Time_(min)","m/z","Intensity"]
        
    """
    
    def __init__(self,csv_path:str):
        """Initializes FeatureData instance with path to input csv data.
 

        Args:
            csv_path: Path to csv file to read peptide feature from.
        Returns:
            None
        """
        
        self.path = csv_path
        self.data3D = pd.read_csv(self.path)

    def sample2D(self,size:int,writeToCsv:bool=False,path:str="sample3D.csv"):
        """Samples 2D tuples from data3D of feature.
        
        Samples (Retention Time, m/z value) tuples from data3D DataFrame.
        Samples with replacement, weighted by intensity value.
        
        Args:
            size: Size of sample.
            writeToCsv: If True sampled data is written to csv.
            path: Path for csv output
        Returns:
            Dataframe with size (RT,M/Z) samples.
        """

        df = self.data3D.sample(n=size,replace=True,weights="Intensity",ignore_index=True)[["Retention_Time_(min)","m/z"]]
        if writeToCsv:
            df.to_csv(path,index=False)
        return df
        
