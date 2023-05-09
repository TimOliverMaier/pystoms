import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib import colormaps


def plot_feature_im_mz(mz, scan, intensity):
    Fig, axs = plt.subplot_mosaic("AA;BC", figsize=(6, 4))
    axs["A"].scatter(x=mz, y=intensity)
    axs["A"].set_xlabel("Mass-to-Charge-Ratio (std)")
    axs["A"].set_ylabel("Intensity (std)")
    axs["A"].set_title("A", loc="left", fontdict={"fontweight": "bold"})

    axs["B"].scatter(x=scan, y=intensity)
    axs["B"].set_xlabel("Scan Number (std)")
    axs["B"].set_ylabel("Intensity (std)")
    axs["B"].set_title("B", loc="left", fontdict={"fontweight": "bold"})

    intensity_scatter = axs["C"].scatter(x=mz, y=scan, c=intensity)
    axs["C"].set_xlabel("Mass-to-Charge-Ratio (std)")
    axs["C"].set_ylabel("Scan (std)")
    axs["C"].set_title("C", loc="left", fontdict={"fontweight": "bold"})
    Fig.colorbar(intensity_scatter, ax=axs["C"], label="Intensity")

    Fig.tight_layout()
    return Fig


def is_oos_plot_lm(
    idata,
    x_name: str,
    x_hidden: str,
    y_name: str,
    obs_name: str,
    random_state: np.random.Generator,
    num_samples: int = 40,
    group: str = "posterior",
    prior_predictions=None,
):

    if group == "prior" and prior_predictions is None:
        raise ValueError("For prior oos plot, prior predictions must be provided")

    if group == "posterior":
        chain_num = idata.posterior.dims["chain"]
        n_draws = idata.posterior.dims["draw"]
        choice_sample = random_state.choice(np.arange(n_draws), num_samples)
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
        chain_num = idata.prior_predictive.dims["chain"]
        n_draws = idata.prior_predictive.dims["draw"]
        choice_sample = random_state.choice(np.arange(n_draws), num_samples)
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
        if y_name in prior_predictions:
            y_prediction_out_sample = (
                prior_predictions.get(y_name)
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
    idata, var_name, coords=None, sharex=False, sharey=False
):
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
