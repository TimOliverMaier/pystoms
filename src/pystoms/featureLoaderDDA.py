"""This module is providing a class to load a precursor peptide feature
aquired by data depending acquistion. The loader recognizes monoisotopic mass
and charge of feature as determined during DDA procedure."""

from cmath import inf
from scipy.stats import poisson, norm
from scipy.optimize import curve_fit
from pyproteolizard.data import PyTimsDataHandle, TimsFrame
from pystoms.clustering import precursorDBSCAN3D
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FeatureLoaderDDA():
    """ Class to load a precursor peptide feature
        by it's Id from a DDA dataset.
    """
    def _fetchPrecursor(self)-> None:
        """get row data from experiment's precursors table
        """
        featureDataRow = self.datasetPointer.get_precursor_by_id(self.precursorID)
        self.monoisotopicMz = featureDataRow["MonoisotopicMz"].values[0]
        self.charge = featureDataRow["Charge"].values[0]
        self.scanNumber = featureDataRow["ScanNumber"].values[0]
        self.frameID = featureDataRow["Parent"].values[0]

    def _getPrecursorSummary(self) -> None:
        """returns precursor row data
        """
        summary = DataFrame({"MonoisotopicMz":self.monoisotopicMz, 
                                "Charge":self.charge,                 
                                "ScanNumber":self.scanNumber,         
                                "FrameID":self.frameID                
                                })
        return(summary)
    
    def _getScanBoundaries(self,datapoints:np.ndarray,IMSmodel:str="gaussian",cutoffL:float=0.05,cutoffR:float=0.95) -> tuple:
        """Estimate minimum scan and maximum scan.

        Args:
            datapoints (np.ndarray): Scan, Itensity data from monoisotopic peak. 
                                     Structure of 2D array [[scan1,intensity_n],...,[scan_n,intensity_n]]
            IMSmodel (str, optional): Model of an IMS peak. Defaults to "gaussian".
            cutoffL (float, optional): Probability mass to ignore on "left side". Defaults to 0.05.
            cutoffR (float, optional): Probability mass to ignore on "right side". Defaults to 0.95.

        Returns:
            tuple (int,int): (lower scan bound, upper scan bound)
        """
        # model functions to fit
        def _gauss(data,α,μ,σ):
            return α*1/(σ*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((data-μ)/(σ),2))
        
        # extract data
        x = datapoints.T[0]
        y = datapoints.T[1]
        
        if IMSmodel == "gaussian":
            # fit model function
            param_opt,param_cov = curve_fit(_gauss,x,y,bounds=([y.min(),x.min(),0],[y.max(),x.max(),inf]))
            
            # instantiate a normal distribution with calculated parameters
            fit_dist = norm(param_opt[1],param_opt[2])
            # calculate lower and upper quantile
            lower = fit_dist.ppf(cutoffL)
            upper = fit_dist.ppf(cutoffR)
            
            return(int(lower//1),int(upper//1+1))

        else:
            raise NotImplementedError("This model is not implemented")

    
    def __init__(self, dataHandle: PyTimsDataHandle, precursorID: int):
        self.datasetPointer = dataHandle
        self.precursorID = precursorID
        # get summary data
        self._fetchPrecursor()

    def loadHullData3D( self,
                    intensityMin:int = 10,
                    IMSmodel:str="gaussian",
                    MZpeakwidth:float=0.1,
                    AveragineProbMassTarget:float = 0.95,
                    plotFeature:bool=False,
                    ScanRange:int=80) -> DataFrame:
        """Estimate convex hull of feature and return datapoints inside hull.

        Args:
            intensityMin (int, optional): Minimal peak intensity considered as signal. Defaults to 10.
            IMSmodel (str, optional): Model use in estimation of feature's width in IMS dimension. Defaults to "gaussian".
            MZpeakwidth (float, optional): Expected width of a peak in mz dimension. Defaults to 0.1.
            AveragineProbMassTarget (float, optional): Probability mass of averagine model's poisson distribution covered
                                                    with extracted isotopic peaks . Defaults to 0.95.
            plotFeature (bool, optional): If true a scatterplot of feature is printed. Defaults to False.
            ScanRange (int, optional): This parameter is handling the number of scans used to infer the
                                       scan bounadaries of the monoisotopic peak. Defaults to 80.

        Returns:
            DataFrame: Dataframe with points in convex hull (scan,mz,intensity)
        """
        # bounds of monoisotopic peak based on arguments
        scanMinInit = int(self.scanNumber//1)-ScanRange//2
        scanMaxInit = int(self.scanNumber//1)+ScanRange//2
        mzMinInit = self.monoisotopicMz-MZpeakwidth
        mzMaxInit = self.monoisotopicMz+MZpeakwidth
        
        # extract monoisotopic peak
        frameInit = self.datasetPointer.get_frame(self.frameID).filter_ranged(scanMinInit,scanMaxInit,mzMinInit,mzMaxInit,intensityMin)
        
         # via averagine calculate how many peaks should be considerd.
        peakN = self.getNumPeaks(self.monoisotopicMz,self.charge,AveragineProbMassTarget)
        mzMin_estimated = self.monoisotopicMz-MZpeakwidth
        mzMax_estimated = self.monoisotopicMz+(peakN-1)*1/self.charge+MZpeakwidth
        
        if IMSmodel in ["gaussian"]:
                
            # calculate profile of monoisotopic peak
            monoProfileData = self.getMonoisotopicProfile(  self.monoisotopicMz,
                                                            self.scanNumber,
                                                            frameInit,
                                                            ScanRange//2,
                                                            MZpeakwidth/2)
            # estimate scan boundaries
            scanMin_estimated,scanMax_estimated = self._getScanBoundaries(monoProfileData,IMSmodel)
        elif IMSmodel == "DBSCAN":
            
            clusteredData, MIClusterID = precursorDBSCAN3D(frameInit,addPoint = (self.monoisotopicMz,self.scanNumber),plot=plotFeature)
            
            if MIClusterID != -1:
                MICluster = clusteredData[clusteredData.Cluster == MIClusterID]
                scanMax_estimated = int(MICluster.Scan.max())
                scanMin_estimated = int(MICluster.Scan.min())
            else:
                raise ValueError("Monoisotopic peak cluster could not be found, use different method for\
                             Eestimation of scan width")
       
        # extract feature's hull data
        frame = self.datasetPointer.get_frame(self.frameID).filter_ranged(  scanMin_estimated,
                                                                                scanMax_estimated,
                                                                                mzMin_estimated,
                                                                                mzMax_estimated,
                                                                                intensityMin)
        scans = frame.scan()
        mzs = frame.mz()
        intensity = frame.intensity()
        
        # plot
        if plotFeature:
            scatter3D = plt.figure()
            ax = scatter3D.add_subplot(111,projection="3d")
            ax.scatter(mzs,scans,intensity)
        # return as Dataframe
        return DataFrame({"Scan":scans,"Mz":mzs,"Intensity":intensity})
    
    
    def loadHullData4D():
        raise NotImplementedError("Extraction of 4D feature is not yet implemented, \
         use .loadData3D for IMS,mz,Intensitiy extraction")

    @staticmethod
    def getNumPeaks(monoisotopicMz: float,charge: int,probMassTarget: float = 0.95) -> int:
        """Calculation of number of isotopic peaks
        by averagine model. 

        Args:
            monoisotopicMz (float): Position of monoisotopic peak.
            charge (int): Charge of peptide
            probMassTarget(float, optional): Minimum probability mass of poisson 
            distribtuion, that shall be covered (beginning with monoisotopic peak).
            Defaults to 0.95.
        Returns:
            int: Number of relevant peaks
        """
        # calculate λ of averagine poisson distribution
        mass = monoisotopicMz * charge
        λ = 0.000594 * mass - 0.03091
        poisson_averagine = poisson(λ)

        # find number of peaks necessary to cover for
        # given probMassTarget
        probMassCovered = 0
        peakNumber = 0
        while probMassCovered < probMassTarget:
            # calculation of probabilty mass of a single peak
            peakMass = poisson_averagine.pmf(peakNumber)
            probMassCovered += peakMass
            peakNumber += 1
            # DEBUG
            # print(f"{peakNumber} peak. Probability Mass : {peakMass}")
        
        return peakNumber

    @staticmethod
    def getMonoisotopicProfile( monoisotopicMZ:float,
                                scanNumber:float,
                                frameSlice:TimsFrame,
                                scanRange:int = 20,
                                mzRange:float=0.05) -> np.ndarray:
        """Gets profile of monoisotopic peak in IMS dimension.

        Sums up peaks per scan that have a mz value close enough (mzRange)
        to monoisotopic peak mz.

        Args:
            monoisotopicMZ (float): Mz value of peak.
            scanNumber (float): ScanNumber of peak.
            frameSlice (TimsFrame): Slice of monoisotopic peak
            scanRange (int, optional): Number of scans to consider. Defaults to 20.
            mzRange (float, optional): Maximal distance of a peak to monoisotopic mz
                                         to be considered in calculation. Defaults to 0.05.
        Returns:
            np.ndarray: 2D Array of structure [[scan,intensity],...]
        """
        # lowest scan number and highest scan number
        scanL = int(scanNumber//1)-scanRange//2
        scanU = scanL + scanRange
        consideredScans = np.arange(scanL,scanU)
        
        # extract values from passed TimsFrame slice of MI peak
        scans = frameSlice.scan().copy()
        mzs = frameSlice.mz().copy()
        intensities = frameSlice.intensity().copy()
        
        idxs = np.zeros((scanRange,2))

        for i,scan_i in enumerate(consideredScans):
            # only view points in mz range and current scan
            intensities_ma = np.ma.MaskedArray(intensities, mask = (scans!=scan_i)|(mzs<monoisotopicMZ-mzRange)|(mzs>monoisotopicMZ+mzRange))
            # sum these points up (intensities) and store in 2D array
            intensity_cumulated = np.ma.sum(intensities_ma) if intensities_ma.count()>0 else 0
            idxs[i] = [scan_i,intensity_cumulated]

        
        return idxs

    
    

