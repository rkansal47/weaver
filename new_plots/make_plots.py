"""
Takes inferenced root files (output of weaver inference) and evaluates the tagger.

Author(s): Raghav Kansal
"""

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import uproot
import os
import argparse

plt.rcParams.update({"font.size": 16})
plt.style.use(hep.style.CMS)
hep.style.use("CMS")

from plot_methods import (
    create_branches,
    reweight_qcd_pt,
    make_cuts,
    plot_hists,
)


def add_bool_arg(parser, name, help, default=False, no_name=None):
    """Add a boolean command line argument for argparse"""
    varname = "_".join(name.split("-"))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=varname, action="store_true", help=help)
    if no_name is None:
        no_name = "no-" + name
        no_help = "don't " + help
    else:
        no_help = help
    group.add_argument("--" + no_name, dest=varname, action="store_false", help=no_help)
    parser.set_defaults(**{varname: default})


# for local interactive testing
args = type("test", (object,), {})()
args.inferences_dir = "../../inferences/04_18_ak8_qcd_oneweight"
args.plot_dir = "../../plots/06_28_javier_plots"
args.sample_files = ["qcd", "HHbbVV"]
args.sample_names = ["QCD", "HHbbVV"]


# from command line args
parser = argparse.ArgumentParser()
parser.add_argument("--inferences-dir", type=str, help="directory containing inferenced root files")
parser.add_argument(
    "--sample-files",
    nargs="+",
    type=str,
    help="root file names of samples",
    default=["qcd", "HHbbVV"],
)
parser.add_argument(
    "--sample-names", nargs="+", type=str, help="sample labels", default=["QCD", "HHbbVV"]
)
parser.add_argument("--plot-dir", required=True, help="output dir")
add_bool_arg(parser, "ph4q", default="True", help="plot old non-MD tagger as well")
args = parser.parse_args()


os.system(f"mkdir -p {args.plot_dir}")


# new branches
sum_scores = {
    "qcdall": ["isQCDb", "isQCDbb", "isQCDc", "isQCDcc", "isQCDothers"],
    "sigall": ["H_VV_4q_3q", "H_VV_4q_4q"],
}

T_scores = {
    "H_VV_4q_4q": ("H_VV_4q_4q", "qcdall"),
    "H_VV_4q_3q": ("H_VV_4q_3q", "qcdall"),
    "HVV4q": ("sigall", "qcdall"),
}

# cuts
pt_key = "pt"
msd_key = "msoftdrop"

cuts_list = [
    {pt_key: [300, 1500], msd_key: [20, 320]},
    {pt_key: [400, 600], msd_key: [60, 150]},
    # {pt_key: [300, 1500], msd_key: [110, 140]},
]

var_labels = {pt_key: "$p_T$", msd_key: "$m_{SD}$"}

# hist vars
# "file tag": {"title": plot title, "score_label": scores to plot}
hist_plot_vars = {
    # "th4q": {
    #     "title": "Non-MD Th4q",
    #     "score_label": "fj_PN_H4qvsQCD",
    # },
    "thvv4q": {
        "title": "MD THVV4q",
        "score_label": "score_fj_THVV4q",
    },
}

# roc vars
roc_plot_vars = {
    "th4q": {
        "title": r"$H\to 4q$ Non-MD GNN tagger",
        "score_label": "fj_PN_H4qvsQCD",
        "colour": "orange",
    },
    "thvv4q": {
        "title": r"$H\to VV\to 4q$ MD GNN tagger",
        "score_label": "score_fj_THVV4q",
        "colour": "green",
    },
    "pnet4q": {
        "title": "MD 4q only",
        "score_label": "score_fj_TH_VV_4q_4q",
        "colour": "red",
        "sig_selector": "fj_H_VV_4q_4q",
    },
    "pnet3q": {
        "title": "MD 3q only",
        "score_label": "score_fj_TH_VV_4q_3q",
        "colour": "blue",
        "sig_selector": "fj_H_VV_4q_3q",
    },
}

if not args.ph4q:
    del roc_plot_vars["th4q"]


# create sample dict
assert len(args.sample_files) == len(args.sample_names), "# sample files should = # sample names"
samples = {args.sample_files[i]: args.sample_names[i] for i in range(len(args.sample_files))}

# load samples
events_dict = {
    sample: uproot.open(f"{args.inferences_dir}/{sample}.root:Events").arrays()
    for sample in samples
}


print("Creating new branches")
create_branches(events_dict, sum_scores, T_scores)

print("Reweighting QCD")
reweight_qcd_pt(events_dict["qcd"])

print("Making cuts")
cuts_dict, cut_labels = make_cuts(events_dict, cuts_list, var_labels)

for cutstr in cut_labels:
    os.system(f"mkdir -p {args.plot_dir}/{cutstr}")

print("Making histograms")
plot_hists(hist_plot_vars, events_dict, cuts_dict, cut_labels, samples, args.plot_dir)


##############################################
# ROCs
##############################################

print("Making ROCs")


from sklearn.metrics import roc_curve, auc

rocs = {}
sig_key = "HHbbVV"
bg_key = "qcd"

for cutstr in cut_labels:
    rocs[cutstr] = {}

    for t, pvars in roc_plot_vars.items():
        sig_cut = cuts_dict[sig_key][cutstr]
        bg_cut = cuts_dict[bg_key][cutstr]

        if "sig_selector" in pvars:
            sig_cut = sig_cut * events_dict[sig_key][pvars["sig_selector"]]

        y_true = np.concatenate(
            [
                np.ones(np.sum(sig_cut)),
                np.zeros(np.sum(bg_cut)),
            ]
        )

        weights = np.concatenate(
            (events_dict[sig_key]["weight"][sig_cut], events_dict[bg_key]["weight"][bg_cut])
        )

        score_label = pvars["score_label"]
        scores = np.concatenate(
            (
                events_dict[sig_key][score_label][sig_cut],
                events_dict[bg_key][score_label][bg_cut],
            )
        )
        fpr, tpr, thresholds = roc_curve(y_true, scores, sample_weight=weights)
        rocs[cutstr][t] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc(fpr, tpr)}


xlim = [0, 0.6]
ylim = [1e-6, 1]

for cutstr in cut_labels:
    plt.figure(figsize=(12, 12))
    for t, pvars in roc_plot_vars.items():
        plt.plot(
            rocs[cutstr][t]["tpr"],
            rocs[cutstr][t]["fpr"],
            label=f"{pvars['title']} AUC: {rocs[cutstr][t]['auc']:.2f}",
            linewidth=2,
            color=pvars["colour"],
        )
        plt.vlines(
            x=rocs[cutstr][t]["tpr"][np.searchsorted(rocs[cutstr][t]["fpr"], 0.01)],
            ymin=0,
            ymax=0.01,
            colors=pvars["colour"],
            linestyles="dashed",
        )
    plt.hlines(y=0.01, xmin=0, xmax=1, colors="lightgrey", linestyles="dashed")
    plt.yscale("log")
    plt.xlabel("Signal Eff.")
    plt.ylabel("BG Eff.")
    plt.suptitle("HVV FatJet ROC Curves", y=0.95)
    plt.title(cut_labels[cutstr], fontsize=20)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend()
    plt.savefig(f"{args.plot_dir}/{cutstr}/roccurve_{cutstr}.pdf", bbox_inches="tight")
