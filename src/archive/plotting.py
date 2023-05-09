"""module with helper function concerning
   visualization"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from pystoms.aligned_feature_data import AlignedFeatureData


def plot_marginals(
    feature: AlignedFeatureData, plot_path: Optional[str] = None
) -> None:
    """plots marginal mz, scan and intensities

    Args:
        feature (AlignedFeatureData): Precursor feature to visualize.
        plot_path (Optional[str]): If provided save figure in
            'plot_path/featureid/marginals.png'
    """
    feature_dataset = feature.feature_data
    feature_ids = feature.accepted_feature_ids
    for feature_id in feature_ids:
        feature_data = feature_dataset.sel({"feature": feature_id}).to_dataframe()
        charge = feature_data["Charge"].values[0]
        Fig, axs = plt.subplot_mosaic("AB;CC", figsize=(5, 5), sharey=True)
        Fig.set_dpi(300)
        axs["A"].hist(x=feature_data.Intensity.values)
        axs["A"].set_xlabel("Intensity")
        axs["A"].set_ylabel("Count")
        axs["A"].annotate(
            f"Max: {np.max(feature_data.Intensity.values)}",
            xy=(0.5, 0.9),
            xycoords="axes fraction",
        )
        axs["B"].hist(x=feature_data.Scan.values)
        axs["B"].set_xlabel("Scan")
        axs["B"].annotate(
            f"n: {len(feature_data.Scan.values)}",
            xy=(0.7, 0.9),
            xycoords="axes fraction",
        )
        axs["C"].hist(x=feature_data.Mz.values, bins=100)
        axs["C"].annotate(f"Charge: {charge}", xy=(0.8, 0.9), xycoords="axes fraction")
        axs["C"].set_xlabel("Mz")
        axs["C"].set_ylabel("Count")
        plt.tight_layout()
        if plot_path is None:
            plt.show()
        else:
            if not os.path.exists(f"{plot_path}/feature{feature_id}/"):
                os.mkdir(f"{plot_path}/feature{feature_id}/")
            plt.savefig(f"{plot_path}/feature{feature_id}/marginals.png")
