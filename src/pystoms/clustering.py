from sklearn.cluster import DBSCAN
from pyproteolizard.data import TimsFrame
from pyproteolizard.utility import *
import numpy as np
import pandas as pd

def precursorDBSCAN3D(frame:TimsFrame,
                      eps:float = 1.7, 
                      min_samples:int = 3, 
                      metric:str = "euclidean",
                      scan_scaling:float =-.4, 
                      resolution:int = 50000) -> pd.DataFrame:
    """Analogus function to pyproteolizard.data.cluster_precursors_dbscan.
       This function executes DBSCAN on 3D data (scan,mz,intenstiy) of a 
       given (possibly filtered) frame.

    Args:
        frame (TimsFrame): Frame to execute DBSCAN on
        eps (float, optional): DBSCAN parameter epsilon, maximal distance 
            for two points to be considered as neighbours. Defaults to 1.7.
        min_samples (int, optional): DBSCAN parameter minimum neighbours 
            for being core point. Defaults to 3.
        metric (str, optional): Metric of DBSCAN. Defaults to "euclidean".
        scan_scaling (float, optional): a scale factor for ion mobility, 
            will be calculated as index / 2^scan_scaling. Defaults to -.4.
        resolution (int, optional): mass spectometer's resolution. Defaults to 50000.

    Returns:
        pd.DataFrame: Dataframe with columns: ["Scans","Mz","DBSCAN_Label"]. 
    """

    # get Data from frame
    scans = frame.scan().copy()
    mzs = frame.mz().copy()

    # rescale data
    mzs_scaled = peak_width_preserving_mz_transform(mzs,resolution)
    scans_scaled = scans/np.power(2,scan_scaling)

    # bin scans and mzs to one array [[scan_1,mz_1],...,[scan_j,mz_j]]
    X = np.stack([mzs_scaled,scans_scaled],axis=1)

    # run DBSCAN
    Clustering = DBSCAN(eps=eps,min_samples=min_samples,metric=metric).fit(X)
    cluster = Clustering.labels_

    return pd.DataFrame(np.stack([scans,mzs,cluster],axis=1),columns=["Scan","Mz","DBSCAN_Label"])
