"""This module is providing a class to load a precursor peptide feature
aquired by data depending acquistion. The loader recognizes monoisotopic mass
and charge of feature as determined during DDA procedure."""

from urllib.request import DataHandler
from scipy.stats import poisson
from pyproteolizard.data import PyTimsDataHandle
from pandas import DataFrame
from seaborn import scatterplot

class FeatureLoaderDDA():
    """ Class to load a precursor peptide feature
        by it's Id in a DDA dataset.
    """
    def _fetchPrecursor(self):
        """ get row data from precursors table"""
        featureDataRow = self.datasetPointer.get_precursor_by_id(self.precursorID)
        self.monoisotopicMz = featureDataRow["MonoisotopicMz"]
        self.charge = featureDataRow["Charge"]
        self.scanNumber = featureDataRow["ScanNumber"]
        self.frameID = featureDataRow["Parent"]

    def _getPrecursorSummary(self):
        """returns precursor row data
        """
        summary = DataFrame({"MonoisotopicMz":self.monoisotopicMz, \
                                "Charge":self.charge,                 \
                                "ScanNumber":self.scanNumber,         \
                                "FrameID":self.frameID                \
                                })
        return(summary)

    def __init__(self, dataHandle: PyTimsDataHandle, precursorID: int):
        self.datasetPointer = dataHandle
        self.precursorID = precursorID
        # get summary data
        self._fetchPrecursor()

    def loadData3D(self,marginLeft,marginRight,intensityMin,scanRange):

        peakN = self.getNumPeaks(self.monoisotopicMz,self.charge)
        scanMin = self.scanNumber-scanRange//2
        scanMax = self.scanNumber+scanRange//2
        mzMin = self.monoisotopicMz-marginLeft
        mzMax = self.monoisotopicMz+(peakN-1)/self.charge+marginRight
        frame = self.datasetPointer.get_frame(self.frameID).filter_ranged(scanMin,scanMax,mzMin,mzMax,intensityMin)
        self.Data3D = frame 

        scans = frame.scan()
        mzs = frame.mz()
        intensity = frame.intensity()

        scatterplot(x=mzs,y=scans,c=intensity)

    

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
