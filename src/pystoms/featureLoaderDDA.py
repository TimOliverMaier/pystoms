"""This module is providing a class to load a precursor peptide feature
aquired by data depending acquistion. The loader recognizes monoisotopic mass
and charge of feature as determined during DDA procedure."""

from cmath import inf
from scipy.stats import poisson, norm
from scipy.optimize import curve_fit
from pyproteolizard.data import PyTimsDataHandle, TimsFrame
from pandas import DataFrame
import numpy as np

class FeatureLoaderDDA():
    """ Class to load a precursor peptide feature
        by it's Id from a DDA dataset.
    """
    def _fetchPrecursor(self):
        """ get row data from precursors table"""
        featureDataRow = self.datasetPointer.get_precursor_by_id(self.precursorID)
        self.monoisotopicMz = featureDataRow["MonoisotopicMz"].values[0]
        self.charge = featureDataRow["Charge"].values[0]
        self.scanNumber = featureDataRow["ScanNumber"].values[0]
        self.frameID = featureDataRow["Parent"].values[0]

    def _getPrecursorSummary(self):
        """returns precursor row data
        """
        summary = DataFrame({"MonoisotopicMz":self.monoisotopicMz, \
                                "Charge":self.charge,                 \
                                "ScanNumber":self.scanNumber,         \
                                "FrameID":self.frameID                \
                                })
        return(summary)
    
    def _getScanBoundaries(self,datapoints:np.ndarray,model:str="gaussian",cutoffL:float=0.05,cutoffR:float=0.95) -> tuple:
        
        def _gauss(data,α,μ,σ):
            return α*1/(σ*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((data-μ)/(σ),2))
        
        x = datapoints.T[0]
        y = datapoints.T[1]
        
        if model == "gaussian":
            param_opt,param_cov = curve_fit(_gauss,x,y,bounds=([y.min(),x.min(),0],[y.max(),x.max(),inf]))
            
            fit_dist = norm(param_opt[1],param_opt[2])
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

    def loadData3D(self,marginLeft:float = 0.1,marginRight:float = 0.1,intensityMin:int = 10):
        """Load Precursor datapoints. Using averagine model for number of isotopic peaks.

        Args:
            marginLeft (float): Mz region lower than monoisotopic peak to consider
            marginRight (float): Mz region lower than monoisotopic peak to consider
            intensityMin (int): Minimum considered intensity.
            
        """
        scanMinInit = int(self.scanNumber//1-40)
        scanMaxInit = int(self.scanNumber//1+40)
        mzMinInit = self.monoisotopicMz-0.1
        mzMaxInit = self.monoisotopicMz+0.1
        
        frameInit = self.datasetPointer.get_frame(self.frameID).filter_ranged(scanMinInit,scanMaxInit,mzMinInit,mzMaxInit,intensityMin)
        
        monoProfileData = self.getMonoisotopicProfile(  self.monoisotopicMz,
                                                        self.scanNumber,
                                                        frameInit)
        scanMin_estimated,scanMax_estimated = self._getScanBoundaries(monoProfileData)
        peakN = self.getNumPeaks(self.monoisotopicMz,self.charge)
        mzMin_estimated = self.monoisotopicMz-marginLeft
        mzMax_estimated = self.monoisotopicMz+(peakN-1)*self.charge+marginRight
        
        frame = self.datasetPointer.get_frame(self.frameID).filter_ranged(  scanMin_estimated,
                                                                                scanMax_estimated,
                                                                                mzMin_estimated,
                                                                                mzMax_estimated,
                                                                                intensityMin)
        

        
        scans = frame.scan()
        mzs = frame.mz()
        intensity = frame.intensity()
        
        #scatterplot(x=mzs,y=scans,c=intensity)

        return DataFrame({"Scan":scans,"Mz":mzs,"Intensity":intensity})
    
    
    def loadData4D():
        raise NotImplementedError("Extraction of 4D feature is not yet implemented, \
         use .loadData3D for IMS,mz,Intensitiy extraction")

    @staticmethod
    def getNumPeaks(monoisotopicMz: float,charge: int,probMassTarget: float = 0.95):
        """Calculation of number of isotopic peaks
        by averagine model. 

        Args:
            monoisotopicMz (float): Position of monoisotopic peak.
            charge (int): Charge of peptide
            probMassTarget(float, optional): Total probability mass of poisson 
            distribtuion, that shall be covered (beginning with monoisotopic peak).
            Defaults to 0.95.
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
    def findClosestPeaks(scanRange:int, monoisotopicMz:float,scanNumber:float,frameSlice:TimsFrame):
        """Finds closest signal to recorded monoisotopic peak
        """
        scanL = scanNumber//1-scanRange//2
        scanU = scanL + scanRange
        consideredScans = np.arange(scanL,scanU)
        
        scans = frameSlice.scan().copy()
        mzs = frameSlice.mz().copy()
        
        
        mzs_dif = mzs-monoisotopicMz
        idxs = np.zeros(scanRange,dtype=np.int16)

        for i,scan_i in enumerate(consideredScans):
            mzs_dif_ma = np.ma.MaskedArray(mzs_dif, mask = scans!=scan_i)
            idx = np.ma.argmin(mzs_dif_ma)
            idxs[i] = idx
        print(idxs)
        return idxs

    @staticmethod
    def getMonoisotopicProfile(monoisotopicMZ:float,scanNumber:float,frameSlice:TimsFrame,scanRange:int = 20,mzRange:float=0.05):
        """_summary_

        Args:
            monoisotopicMZ (float): _description_
            scanNumber (float): _description_
            frameSlice (TimsFrame): _description_
            scanRange (int, optional): _description_. Defaults to 10.
            mzRange (float, optional): _description_. Defaults to 0.05.
        """
        scanL = scanNumber//1-scanRange//2
        scanU = scanL + scanRange
        consideredScans = np.arange(scanL,scanU)
        
        scans = frameSlice.scan().copy()
        mzs = frameSlice.mz().copy()
        intensities = frameSlice.intensity().copy()
        
        idxs = np.zeros((scanRange,2))

        for i,scan_i in enumerate(consideredScans):
            intensities_ma = np.ma.MaskedArray(intensities, mask = (scans!=scan_i)|(mzs<monoisotopicMZ-mzRange)|(mzs>monoisotopicMZ+mzRange))
            intensity_cumulated = np.ma.sum(intensities_ma)
            idxs[i] = [scan_i,intensity_cumulated]

        
        return idxs


    

