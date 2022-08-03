"""Clustering functions for LC-IMS-MS data.

Currently, there is only the precursor_dbscan_3d, allowing for
detection of 2D peaks in the IMS-MS space.
"""

from sklearn.cluster import DBSCAN
from proteolizarddata.data import TimsFrame
from proteolizardalgo.utility import peak_width_preserving_mz_transform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def precursor_dbscan_3d(frame:TimsFrame,
                      eps:float = 1.7,
                      min_samples:int = 3,
                      metric:str = "euclidean",
                      scan_scaling:float =-.4,
                      resolution:int = 50000,
                      add_point:tuple = (None,None),
                      plot:bool = False) -> tuple:
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
        resolution (int, optional): mass spectometer's resolution.
          Defaults to 50000.
        add_point (tuple, optional): Add artifical point to find cluster
          of e.g. MI peak. (mz,scan). Defaults to (None,None).
        plot (bool, optional): If true plot of datapoints is
          printed (if passed with added MI peak)

    Returns:
        tuple (pd.DataFrame,int): Dataframe with columns:
          ["Scans","Mz","DBSCAN_Label"]. And label of added point
          (-1 if no point was added).
    """

    # get Data from frame
    scans = frame.scan().copy()
    mzs = frame.mz().copy()

    # rescale data
    mzs_scaled = peak_width_preserving_mz_transform(mzs,resolution)
    scans_scaled = scans/np.power(2,scan_scaling)

    # bin scans and mzs to one array [[scan_1,mz_1],...,[scan_j,mz_j]]
    x = np.stack([mzs_scaled,scans_scaled],axis=1)

    # if extra peak was passed scale this peak accodringly and add
    # on end of point arrays
    if add_point[0] is not None:
        mi_mz_scaled = peak_width_preserving_mz_transform(add_point[0],
          resolution)
        mi_scan_scaled = add_point[1]/np.power(2,scan_scaling)
        if plot:
            plt.scatter(mzs,scans,alpha=0.2)
            plt.scatter(add_point[0],add_point[1])
            plt.show()
        x = np.vstack([x,[mi_mz_scaled,mi_scan_scaled]])

    # run DBSCAN
    clustering = DBSCAN(eps=eps,min_samples=min_samples,metric=metric).fit(x)
    cluster = clustering.labels_

    # Get cluster id of passed extra peak
    # (e.g. monoisotopic peak data from Precursors table)
    mi_cluster_id = -1
    if add_point[0] is not None:
        print(mi_cluster_id)
        mi_cluster_id = cluster[-1]
        cluster = np.delete(cluster,-1)
    return_dataframe = pd.DataFrame(np.stack([scans,mzs,cluster],axis=1),
      columns=["Scan","Mz","Cluster"])

    # return labeled data and cluster ID of passed extra peak
    return (return_dataframe,mi_cluster_id)
