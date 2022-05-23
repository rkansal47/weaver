from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.ROOT)

from coffea import hist


def roc_input(
    events,
    var,
    label_sig,
    label_bkg,
    weight_hist=None,
    bins=None,
    sig_mask=None,
    bkg_mask=None,
    roc_mask=None,
):
    """
    Return input of ROC curve (true labels, predicted values of scores, and weight)
    Arguments:
    - events: table with score and labels
    - var: discriminator score
    - label_sig: name of signal label
    - label_bkg: name of background label
    - weight_hist: weight histogram (in case of applying weights)
    - bins: bins for weighting (in case of applying weights)
    - sig_mask: signal mask (in case of masking signal with variables in table)
    - bkg_mask: background mask (in case of masking background events using variables in table)
    - roc_mask: mask to apply to both signal and background jets
    """
    mask_sig = events[label_sig] == 1
    mask_bkg = events[label_bkg] == 1

    if sig_mask is not None:
        mask_sig = (mask_sig) & (sig_mask)
    if bkg_mask is not None:
        mask_bkg = (mask_bkg) & (bkg_mask)
    if roc_mask is not None:
        mask_sig = (mask_sig) & (roc_mask)
        mask_bkg = (mask_bkg) & (roc_mask)
    scores_sig = events[var][mask_sig].to_numpy()
    scores_bkg = events[var][mask_bkg].to_numpy()
    predict = np.concatenate((scores_sig, scores_bkg), axis=None)
    siglabels = np.ones(scores_sig.shape)
    bkglabels = np.zeros(scores_bkg.shape)
    truth = np.concatenate((siglabels, bkglabels), axis=None)

    weight = None
    if weight_hist is not None:
        weight_sig = weight_hist[np.digitize(events["fj_pt"][mask_sig].to_numpy(), bins) - 1]
        weight_bkg = np.ones(scores_bkg.shape)
        weight = np.concatenate((weight_sig, weight_bkg), axis=None)

    return truth, predict, weight


def get_roc(
    events,
    score_name,
    label_sig,
    label_bkg,
    weight_hist=None,
    bins=None,
    sig_mask=None,
    bkg_mask=None,
    roc_mask=None,
    ratio=False,
    mask_num=None,
    mask_den=None,
):
    """
    Get both the ROC input and the curve
    Arguments:
    - inputs for roc_input
    """
    if ratio:
        truth, predict, weight = roc_input(events, score_name, label_sig, label_bkg)
        fprs, tprs, threshold = roc_curve(truth, predict, sample_weight=weight)
        weight_sig = np.sum(mask_num & (events[label_sig] == 1)) / np.sum(
            mask_den & (events[label_sig] == 1)
        )
        weight_bkg = np.sum(mask_num & (events[label_bkg] == 1)) / np.sum(
            mask_den & (events[label_bkg] == 1)
        )
        fprs = fprs * weight_bkg
        tprs = tprs * weight_sig
    else:
        truth, predict, weight = roc_input(
            events,
            score_name,
            label_sig,
            label_bkg,
            weight_hist,
            bins,
            sig_mask,
            bkg_mask,
            roc_mask,
        )
        fprs, tprs, threshold = roc_curve(truth, predict, sample_weight=weight)

    return (fprs, tprs)


def plot_roc(
    odir,
    label_sig,
    label_bkg,
    fp_tp,
    label,
    title,
    pkeys=None,
    ptcut=None,
    msdcut=None,
):
    """
    Plot ROC
    Arguments:
    - odir: output directory
    - label_sig: signal label
    - label_bkg: background label
    - fp_tp: (false positive rates, true positive rates) tuple
    - label: label for output name of .png
    - title: title of ROC curve
    - ptcut: pT cut applied
    - msdcut: mSD cut applied
    - axs: axis in which to plot ROC curve
    """

    fig, axs = plt.subplots(1, 1, figsize=(16, 16))

    def get_round(x_effs, y_effs, to_get=[0.01, 0.02, 0.03]):
        effs = []
        for eff in to_get:
            for i, f in enumerate(x_effs):
                if round(f, 2) == eff:
                    effs.append(y_effs[i])
                    break
        return effs

    def get_intersections(x_effs, y_effs, to_get=0.01):
        x_eff = 0
        for i, f in enumerate(y_effs):
            if f >= to_get:
                x_eff = x_effs[i]
                break
        return x_eff

    def get_x_intersections(x_effs, y_effs, to_get=0.01):
        y_eff = 0
        for i, f in enumerate(x_effs):
            if f >= to_get:
                y_eff = y_effs[i]
                break
        return y_eff
    
    # draw intersections at 1% mis-tag rate
    ik = 0
    markers = ["v", "^", "o", "s", "p", "P", "h"]
    ratiomethod = False
    for k, it in fp_tp.items():
        if pkeys and k not in pkeys:
            continue
        leg = k.replace("_score", "")
        if "ratio" in leg:
            ratiomethod = True
        leg = leg.replace("-ratio", "")
        fp = it[0]
        tp = it[1]
        if ratiomethod:
            axs.plot(tp, fp, lw=3, label=r"{}".format(leg))
        else:
            axs.plot(tp, fp, lw=3, label=r"{}, AUC = {:.1f}%".format(leg, auc(fp, tp) * 100))
        y_eff = 0.01
        x_eff = get_intersections(tp, fp, y_eff)
        axs.hlines(
            y=y_eff, xmin=0.00001, xmax=0.99999, linewidth=1.3, color="dimgrey", linestyle="dashed"
        )
        axs.vlines(
            x=x_eff, ymin=0.00001, ymax=y_eff, linewidth=1.3, color="dimgrey", linestyle="dashed"
        )
        if x_eff < 0.45:
            y_eff_2 = get_x_intersections(tp, fp, 0.45)
            axs.hlines(
                y=y_eff_2,  xmin=0.00001, xmax=0.99999, linewidth=1.3, color="dimgrey", linestyle="dashed"
            )
            axs.vlines(
                x=0.45, ymin=0.00001, ymax=y_eff_2, linewidth=1.3, color="dimgrey", linestyle="dashed"
            )
        ik += 1

    axs.legend(loc="lower right", fontsize=25)
    axs.grid(which="minor", alpha=0.5)
    axs.grid(which="major", alpha=0.5)
    axs.set_xlabel(r"Tagging efficiency %s" % label_sig, fontsize=40)
    axs.set_ylabel(r"Mistagging rate %s" % label_bkg, fontsize=40)
    axs.set_ylim(0.0001, 1)
    axs.set_xlim(0.0001, 1)
    axs.set_yscale("log")
    if ptcut:
        axs.text(0.05, 0.5, ptcut, fontsize=30)
    if msdcut:
        axs.text(0.05, 0.3, msdcut, fontsize=30)
    if ratiomethod:
        axs.text(0.45, 0.005, r"Rescaled by pass $m_{SD}&p_T$/pass $p_T$", fontsize=32)
    axs.set_title(title, fontsize=40)

    plt.tight_layout()

    fig.savefig("%s/roc_%s_ylog.pdf" % (odir, label))
    fig.savefig("%s/roc_%s_ylog.png" % (odir, label))

    # also save version down to 10^-5
    # axs.set_ylim(0.00001,1)
    # axs.set_xlim(0.0001,1)
    # fig.savefig("%s/roc_%s_ylogm5.pdf"%(odir,label))
    # fig.savefig("%s/roc_%s_ylogm5.png"%(odir,label))

    axs.set_yscale("linear")


def plot_1d(odir, hist_val, vars_to_plot, label):
    """
    Plot 1d histograms for validation
    Arguments:
    - odir: output directory
    - hist_val: histogram
    - vars_to_plot: variables to project
    - label: label of plot
    """
    for density in [True, False]:
        fig, axs = plt.subplots(1, len(vars_to_plot), figsize=(len(vars_to_plot) * 8, 8))
        for i, m in enumerate(vars_to_plot):
            if len(vars_to_plot) == 1:
                axs_1 = axs
            else:
                axs_1 = axs[i]
            x = hist_val.sum(*[ax for ax in hist_val.axes() if ax.name not in {"process", m}])
            hist.plot1d(x, ax=axs_1, overlay="process", density=density)
            axs_1.set_ylabel("Jets")
        fig.tight_layout()
        if density:
            fig.savefig(f"{odir}/{label}_density.pdf")
            fig.savefig(f"{odir}/{label}_density.png")
        else:
            fig.savefig(f"{odir}/{label}.pdf")
            fig.savefig(f"{odir}/{label}.png")


def computePercentiles(data, percentiles):
    """
    Computes the cuts that we should make on the tagger score so that we obtain this efficiency in our process after the cut
    Arguments:
    - data: values of tagger score
    - percentiles: list of percentiles
    """
    mincut = 0.0
    tmp = np.quantile(data, np.array(percentiles))
    tmpl = [mincut]
    for x in tmp:
        tmpl.append(x)
    perc = [0.0]
    for x in percentiles:
        perc.append(x)
    return perc, tmpl


def plot_var_aftercut(odir, hists_to_plot, labels, xlabel, tag, ptcut):
    # get histogram divided by integral so that we can get the correct ratio
    hscales = []
    for h in hists_to_plot:
        s = np.sum(h.values())
        y = h * (1 / s)
        hscales.append(y.values())
    ratio = []
    for i, h in enumerate(hscales):
        ratio.append(hscales[i] / hscales[0])

    fig, axs = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 8),
        gridspec_kw={"height_ratios": (3, 1)},
        sharex=True,
    )
    axs_1 = axs[0]
    raxs_1 = axs[1]
    fig.subplots_adjust(hspace=0.07)
    colors = ["k", "r", "g", "b"]
    error_opts = {
        "linestyle": "none",
        "marker": ".",
        "markersize": 10.0,
        "elinewidth": 1,
    }
    hep.histplot(
        hists_to_plot,
        ax=axs_1,
        xerr=True,
        density=True,
        # binwnorm=1,
        color=colors,
        yerr=True,
        label=labels,
        histtype="errorbar",
        **error_opts,
    )
    major_ticks = np.arange(25, 275, 25)
    minor_ticks = np.arange(25, 275, 10)
    major_ticks[0] = 30
    minor_ticks[0] = 30
    major_ticks[-1] = 250
    minor_ticks[-1] = 250
    axs_1.set_xticks(major_ticks)
    axs_1.set_xticks(minor_ticks, minor=True)
    axs_1.grid(which="major", alpha=0.6)
    # axs_1.grid(which="minor", alpha=0.6)
    axs_1.set_ylabel("A.U.")
    axs_1.set_xlim(30, 250)
    raxs_1.set_xlim(30, 250)
    for i, r in enumerate(ratio):
        if i == 0:
            fmt = "-"
        else:
            fmt = "o"
        raxs_1.errorbar(
            x=hists_to_plot[0].axes[0].centers,
            y=r,
            xerr=hists_to_plot[0].axes[0].widths / 2,
            # yerr=np.sqrt(r),
            color=colors[i],
            fmt=fmt,
        )
    raxs_1.set_xticks(major_ticks)
    raxs_1.set_xticks(minor_ticks, minor=True)
    raxs_1.grid(which="major", alpha=0.6)
    raxs_1.set_ylabel(r"$\frac{\epsilon_B}{Inclusive}$")
    raxs_1.set_ylim(0.5, 1.5)

    raxs_1.set_xlabel(xlabel)
    axs_1.set_xlabel("")
    axs_1.text(50, 0.01, ptcut, fontsize=15)
    axs_1.legend()
    axs_1.get_shared_x_axes().join(axs_1, raxs_1)
    fig.tight_layout()
    fig.savefig("%s/%s_scoresculpting.pdf" % (odir, tag))
    fig.savefig("%s/%s_scoresculpting.png" % (odir, tag))
