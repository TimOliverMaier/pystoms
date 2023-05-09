from typing import Optional, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import numpy as np
import arviz as az
from numpy.typing import ArrayLike
from matplotlib import colormaps


def plot_feature_im_mz(
    mz: ArrayLike, scan: ArrayLike, intensity: ArrayLike, standardized: bool = True
):
    """
    Summary plot of a precursor feature
    (inside a fixed frame).


    :param mz: Mass-to-charge-ratio values
    :type mz: ArrayLike
    :param scan: Scan values
    :type scan: ArrayLike
    :param intensity: Intensity values
    :type intensity: ArrayLike
    :param standardized: Wether values are standardized (alters labels)
    :type standardized: bool, optional
    :return: Summary Figure
    :rtype: plt.Figure
    """
    if standardized:
        suffix = " (std)"
    else:
        suffix = ""

    Fig, axs = plt.subplot_mosaic("AA;BC", figsize=(6, 4))

    # Intensity vs Mz
    axs["A"].scatter(x=mz, y=intensity)
    axs["A"].set_xlabel(f"Mass-to-Charge-Ratio{suffix}")
    axs["A"].set_ylabel(f"Intensity{suffix}")
    axs["A"].set_title("A", loc="left", fontdict={"fontweight": "bold"})

    # Intensity vs Scan
    axs["B"].scatter(x=scan, y=intensity)
    axs["B"].set_xlabel(f"Scan Number{suffix}")
    axs["B"].set_ylabel(f"Intensity{suffix}")
    axs["B"].set_title("B", loc="left", fontdict={"fontweight": "bold"})

    # 2D Intensity vs (mz, scan)
    intensity_scatter = axs["C"].scatter(x=mz, y=scan, c=intensity)
    axs["C"].set_xlabel(f"Mass-to-Charge-Ratio{suffix}")
    axs["C"].set_ylabel(f"Scan{suffix}")
    axs["C"].set_title("C", loc="left", fontdict={"fontweight": "bold"})
    Fig.colorbar(intensity_scatter, ax=axs["C"], label="Intensity")

    Fig.tight_layout()
    return Fig


def is_oos_plot_lm(
    idata: az.InferenceData,
    x_name: str,
    x_hidden: str,
    y_name: str,
    obs_name: str,
    random_state: np.random.Generator,
    num_samples: int = 40,
    group: str = "posterior",
    prior_predictions: Optional[az.InferenceData] = None,
):
    """
    In-sample and out-of-sample prior/posterior
    prediction plots for 2D regression models, similar to
    `arviz.plot_lm`

    :param idata: InferenceData (with in-sample and out-of-sample
                  posterior-predictive and in-sample prior-predictive samples)
    :type idata: az.InferenceData
    :param x_name: Name of variable to consider as x
    :type x_name: str
    :param x_hidden: Name of variable that is marginalized
    :type x_hidden: str
    :param y_name: Name of variable to consider as y
    :type y_name: str
    :param obs_name: Name of observed variable
    :type obs_name: str
    :param random_state: random seed
    :type random_state: np.random.Generator
    :param num_samples: Number of predictive samples per x data point, defaults to 40
    :type num_samples: int, optional
    :param group: Wether to visualize prior- or posterior-predictive-checks,
                  defaults to "posterior"
    :type group: str, optional
    :param prior_predictions: Out-of-sample prior-predictive samples, defaults to None
    :type prior_predictions: Optional[az.InferenceData], optional
    :raises ValueError: If name of variable is not found.
    :return: Figure in-sample (scatter), Figure out-of-sample (scatter), Figure out-of-sample (line)
    :rtype: Tuple[plt.Figure]
    """
    if group == "prior" and prior_predictions is None:
        raise ValueError("For prior oos plot, prior predictions must be provided")

    if group == "posterior":
        # number of chains and draws must be the same
        # in in-sample and out-of-sample data
        chain_num = idata.posterior.dims["chain"]
        n_draws = idata.posterior.dims["draw"]
        # select draws
        choice_sample = random_state.choice(np.arange(n_draws), num_samples)

        # find y in in-sample data
        if y_name in idata.posterior_predictive:
            y_prediction_in_sample = (
                idata.posterior_predictive.get(y_name)
                .isel(draw=choice_sample)
                .to_dataframe()
                .reset_index()
            )
        elif y_name in idata.posterior:
            y_prediction_in_sample = (
                idata.posterior.get(y_name)
                .isel(draw=choice_sample)
                .to_dataframe()
                .reset_index()
            )
        else:
            raise ValueError(
                f"{y_name} for in-sample predictions plot was neither found in posterior-predictive nor posterior group"
            )

        # for out-of sample data y must be in predictions
        if y_name in idata.predictions:
            y_prediction_out_sample = (
                idata.predictions.get(y_name)
                .isel(draw=choice_sample)
                .to_dataframe()
                .reset_index()
            )
        else:
            raise ValueError(
                f"{y_name} for out-of-sample predictions plot was not found in predictions group"
            )

    elif group == "prior":
        # number of chains and draws must be the same
        # in in-sample and out-of-sample data
        chain_num = idata.prior_predictive.dims["chain"]
        n_draws = idata.prior_predictive.dims["draw"]
        # select draws
        choice_sample = random_state.choice(np.arange(n_draws), num_samples)

        # find y in in-sample data
        if y_name in idata.prior_predictive:
            y_prediction_in_sample = (
                idata.prior_predictive.get(y_name)
                .isel(draw=choice_sample)
                .to_dataframe()
                .reset_index()
            )
        elif y_name in idata.prior:
            y_prediction_in_sample = (
                idata.prior.get(y_name)
                .isel(draw=choice_sample)
                .to_dataframe()
                .reset_index()
            )
        else:
            raise ValueError(
                f"{y_name} for in-sample predictions plot was neither found in prior-predictive nor prior group"
            )
        # For out-of-sample data y must be in prior_predictive
        # Here, it could also be in `prior` (for the posterior case `posterior`
        # still holds in-sample data) but since no
        # prior_predictions group exists the here used idata is completely new
        # however for consistency only `prior_predictive` is searched here
        if y_name in prior_predictions.prior_predictive:
            y_prediction_out_sample = (
                prior_predictions.prior_predictive.get(y_name)
                .isel(draw=choice_sample)
                .to_dataframe()
                .reset_index()
            )
        else:
            raise ValueError(
                f"{y_name} for out-of-sample predictions plot was not found in prior-predictions group"
            )

    x_observed = idata.constant_data.get(x_name).to_dataframe().reset_index()
    y_observed = idata.observed_data.get(obs_name).to_dataframe().reset_index()
    x_oos = (
        idata.predictions_constant_data.get([x_name, x_hidden])
        .to_dataframe()
        .reset_index()
    )

    observed_data = x_observed.merge(y_observed, on="data_point").sort_values(
        by="data_point"
    )
    is_data = x_observed.merge(y_prediction_in_sample, on="data_point").sort_values(
        by="data_point"
    )
    oos_data = x_oos.merge(y_prediction_out_sample, on="data_point").sort_values(
        by="data_point"
    )

    color_map = colormaps["twilight"]

    if chain_num == 1:
        Fig_is, ax_is = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
        Fig_oos, ax_oos = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
        Fig_oos_line, ax_oos_line = plt.subplots(1, 1, figsize=(6, 4), sharex=True)

        ax_is.scatter(
            x=observed_data.loc[:, x_name],
            y=observed_data.loc[:, obs_name],
            color="black",
            label="observed",
        )
        ax_is.scatter(
            x=is_data.loc[:, x_name],
            y=is_data.loc[:, y_name],
            label="predicted",
            alpha=0.2,
        )

        ax_oos.scatter(
            x=observed_data.loc[:, x_name],
            y=observed_data.loc[:, obs_name],
            color="black",
            label="observed",
        )
        ax_oos.scatter(
            x=oos_data.loc[:, x_name],
            y=oos_data.loc[:, y_name],
            label="predicted",
            alpha=0.2,
        )

        ax_oos_line_observed = ax_oos_line.scatter(
            x=observed_data.loc[:, x_name],
            y=observed_data.loc[:, obs_name],
            color="black",
        )

        for color, draw in enumerate(choice_sample):
            # scale color int to [0,1] for colormap
            color /= num_samples
            rows_draw = oos_data.draw == draw
            for hidden_x in oos_data[x_hidden].unique():
                rows_draw_hidden_x = (rows_draw) & (oos_data[x_hidden] == hidden_x)
                ax_oos_line.plot(
                    oos_data.loc[rows_draw_hidden_x, x_name],
                    oos_data.loc[rows_draw_hidden_x, y_name],
                    color=color_map(color),
                    alpha=0.7,
                )

        Fig_is.supxlabel(x_name)
        Fig_is.supylabel(y_name)
        Fig_oos.supxlabel(x_name)
        Fig_oos.supylabel(y_name)
        Fig_oos_line.supxlabel(x_name)
        Fig_oos_line.supylabel(y_name)

        Fig_is.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.05))
        Fig_oos.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.05))
        line = mlines.Line2D([], [], color="black", label="predicted")
        Fig_oos_line.legend(
            [ax_oos_line_observed, line],
            ["observed", "predicted"],
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.05),
        )
        Fig_is.tight_layout()
        Fig_oos.tight_layout()
        Fig_oos_line.tight_layout()

        return Fig_is, Fig_oos, Fig_oos_line

    ncol = 2
    nrow = int(chain_num // 2 + chain_num % 2)
    figsize = (6, 2 * nrow)

    Fig_is, axs_is = plt.subplots(nrow, ncol, figsize=figsize, sharex=True)
    Fig_oos, axs_oos = plt.subplots(nrow, ncol, figsize=figsize, sharex=True)
    Fig_oos_line, axs_oos_line = plt.subplots(nrow, ncol, figsize=figsize, sharex=True)
    # remove unnecessary axes at the end of plot in case of uneven chain number
    for ax_is, ax_oos, ax_oos_line in zip(
        axs_is.flatten()[chain_num:],
        axs_oos.flatten()[chain_num:],
        axs_oos_line.flatten()[chain_num:],
    ):
        ax_is.remove()
        ax_oos.remove()
        ax_oos_line.remove()

    for c, ax_is, ax_oos, ax_oos_line in zip(
        range(chain_num),
        axs_is.flatten()[:chain_num],
        axs_oos.flatten()[:chain_num],
        axs_oos_line.flatten()[:chain_num],
    ):

        observed_is = ax_is.scatter(
            x=observed_data.loc[:, x_name],
            y=observed_data.loc[:, obs_name],
            color="black",
        )
        predicted_is = ax_is.scatter(
            x=is_data.loc[is_data.chain == c, x_name],
            y=is_data.loc[is_data.chain == c, y_name],
            alpha=0.2,
        )
        ax_is.set_title(f"Chain {c}")
        ax_is.set_xlabel(None)
        ax_is.set_ylabel(None)

        observed_oos = ax_oos.scatter(
            x=observed_data.loc[:, x_name],
            y=observed_data.loc[:, obs_name],
            color="black",
        )
        predicted_oos = ax_oos.scatter(
            x=oos_data.loc[oos_data.chain == c, x_name],
            y=oos_data.loc[oos_data.chain == c, y_name],
            alpha=0.2,
        )

        ax_oos.set_title(f"Chain {c}")
        ax_oos.set_xlabel(None)
        ax_oos.set_ylabel(None)

        observed_oos_line = ax_oos_line.scatter(
            x=observed_data.loc[:, x_name],
            y=observed_data.loc[:, obs_name],
            color="black",
        )
        rows_chain = oos_data.chain == c
        for color, draw in enumerate(choice_sample):
            # scale color int to [0,1] for colormap
            color /= num_samples
            rows_draw_chain = (rows_chain) & (oos_data.draw == draw)
            for hidden_x in oos_data[x_hidden].unique():
                rows_draw_chain_hidden_x = (rows_draw_chain) & (
                    oos_data[x_hidden] == hidden_x
                )
                ax_oos_line.plot(
                    oos_data.loc[rows_draw_chain_hidden_x, x_name],
                    oos_data.loc[rows_draw_chain_hidden_x, y_name],
                    color=color_map(color),
                    alpha=0.7,
                )

        ax_oos_line.set_title(f"Chain {c}")
        ax_oos_line.set_xlabel(None)
        ax_oos_line.set_ylabel(None)

    Fig_is.supxlabel(x_name)
    Fig_is.supylabel(y_name)
    Fig_is.legend(
        [observed_is, predicted_is],
        ["observed", "predicted"],
        loc="lower center",
        ncols=2,
        bbox_to_anchor=(0.5, 0.05),
    )
    Fig_is.tight_layout()

    Fig_oos.supxlabel(x_name)
    Fig_oos.supylabel(y_name)
    Fig_oos.legend(
        [observed_oos, predicted_oos],
        ["observed", "predicted"],
        loc="lower center",
        ncols=2,
        bbox_to_anchor=(0.5, 0.05),
    )
    Fig_oos.tight_layout()

    Fig_oos_line.supxlabel(x_name)
    Fig_oos_line.supylabel(y_name)
    line = mlines.Line2D([], [], color="black", label="predicted")
    Fig_oos_line.legend(
        [observed_oos_line, line],
        ["observed", "predicted"],
        loc="lower center",
        ncols=2,
        bbox_to_anchor=(0.5, 0.05),
    )
    Fig_oos_line.tight_layout()

    return Fig_is, Fig_oos, Fig_oos_line


def chain_comparison_posterior_plot(
    idata: az.InferenceData,
    var_name: str,
    coords: Optional[Dict] = None,
    sharex: bool = False,
    sharey: bool = False,
):
    """
    Compare posterior of `var_name` between chains.

    :param idata: InferenceData with posterior samples.
    :type idata: az.InferenceData
    :param var_name: Name of variable to plot.
    :type var_name: str
    :param coords: Selection of coordinates to consider, defaults to None
    :type coords: Optional[Dict], optional
    :param sharex: Wether subplots share x-axis, defaults to False
    :type sharex: bool, optional
    :param sharey: Wether subplots share y-axis, defaults to False
    :type sharey: bool, optional
    :return: Figure
    :rtype: plt.Figure
    """
    chain_num = idata.posterior.dims["chain"]

    if coords is not None:
        posterior = (
            idata.posterior.get(var_name).sel(coords).to_dataframe().reset_index()
        )
    else:
        posterior = idata.posterior.get(var_name).to_dataframe().reset_index()

    ncol = 2
    nrow = int(chain_num // 2 + chain_num % 2)
    figsize = (6, 2 * nrow)

    Fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharex=sharex, sharey=sharey)
    # remove unnecessary axes at the end of plot in case of uneven chain number
    for ax in axs.flatten()[chain_num:]:
        ax.remove()

    for c, ax in zip(range(chain_num), axs.flatten()[:chain_num]):
        sns.histplot(
            data=posterior.loc[posterior.chain == c, :], x=var_name, kde=True, ax=ax
        )
        ax.set_title(f"Chain {c}")
        ax.set_xlabel(None)
        ax.set_ylabel(None)
    Fig.supylabel("Count")
    Fig.supxlabel(var_name)
    Fig.tight_layout()
    return Fig
