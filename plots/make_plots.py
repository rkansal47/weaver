"""
Takes inferenced root files (output of weaver inference) and evaluates the tagger.

Author(s): Raghav Kansal
"""

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import uproot
import awkward as ak
import os
import argparse

plt.rcParams.update({"font.size": 16})
plt.style.use(hep.style.CMS)
hep.style.use("CMS")


# for local testing
args = type("test", (object,), {})()
args.inferences_dir = "../../inferences/04_18_ak8_qcd_oneweight"
args.plot_dir = "../../plots/05_23_testing"
args.sample_files = ["qcd", "HHbbVV"]
args.sample_names = ["QCD", "HHbbVV"]


parser = argparse.ArgumentParser()
parser.add_argument("--inferences-dir", type=str, help="directory containing inferenced root files")
parser.add_argument("--sample-files", nargs="+", type=str, help="root file names of samples")
parser.add_argument("--sample-names", nargs="+", type=str, help="sample labels")
parser.add_argument("--plot-dir", required=True, help="output dir")
args = parser.parse_args()


os.system(f"mkdir -p {args.plot_dir}")


##############################################
# Load samples, compute scores
##############################################

assert len(args.sample_files) == len(args.sample_names), "# sample files should = # sample names"
samples = {args.sample_files[i]: args.sample_names[i] for i in range(len(args.sample_files))}

events_dict = {
    sample: uproot.open(f"{args.inferences_dir}/{sample}.root:Events").arrays()
    for sample in samples
}

qcd_classes = ["QCDb", "QCDbb", "QCDc", "QCDcc", "QCDothers"]
sig_classes = ["H_VV_4q_3q", "H_VV_4q_4q"]

for sample, events in events_dict.items():
    events["score_fj_qcdall"] = np.sum(
        [events[f"score_fj_is{subclass}"] for subclass in qcd_classes], axis=0
    )

    for subclass in sig_classes:
        events[f"score_fj_T{subclass}"] = events[f"score_fj_{subclass}"] / (
            events[f"score_fj_{subclass}"] + events["score_fj_qcdall"]
        )

    events["score_fj_sigall"] = np.sum(
        [events[f"score_fj_{subclass}"] for subclass in sig_classes], axis=0
    )

    events["score_fj_THVV4q"] = events["score_fj_sigall"] / (
        events["score_fj_sigall"] + events["score_fj_qcdall"]
    )

    events["weight"] = np.ones(len(events))


num_3q = np.sum(events_dict["HHbbVV"]["fj_H_VV_4q_3q"])
num_4q = np.sum(events_dict["HHbbVV"]["fj_H_VV_4q_4q"])
print(f"# of 3q signal events: {num_3q}")
print(f"# of 4q signal events: {num_4q}")

##############################################
# Weight QCD to be exponentially falling by pT
##############################################

from coffea.lookup_tools.dense_lookup import dense_lookup
from hist import Hist

reweight_bins = [120, 300, 1500]
bin_centers = np.linspace(reweight_bins[1], reweight_bins[2], reweight_bins[0] + 1)[:-1]
bin_centers += (bin_centers[1] - bin_centers[0]) / 2
# qcd pt 1/e range is roughly ~100 GeV
exp_falling_weights = np.exp((reweight_bins[1] - bin_centers) / 100)

qcd_pt = (Hist.new.Reg(*reweight_bins, name="pt").Double()).fill(
    pt=events_dict["qcd"]["fj_pt"],
)

pt_weights = exp_falling_weights / qcd_pt.view(flow=False)
pt_weights_lookup = dense_lookup(pt_weights, qcd_pt.axes.edges)

events_dict["qcd"]["weight"] = pt_weights_lookup(events_dict["qcd"]["fj_pt"])

# Checking QCD distribution

# _ = plt.hist(
#     events_dict["qcd"]["fj_pt"],
#     np.linspace(300, 1000, 101),
#     weights=events_dict["qcd"]["weight"],
#     histtype="step",
# )

##############################################
# Cuts
##############################################

"""
``cuts_dict`` will be of format:
{
    sample1: {
        "cut1var1_min_max_cut1var2...": cut1,
        "cut2var2...": cut2,
        ...
    },
    sample2...
}
"""

pt_key = "pt"
msd_key = "msoftdrop"

all_cuts = [
    {pt_key: [300, 1500], msd_key: [20, 320]},
    {pt_key: [400, 600], msd_key: [60, 150]},
    # {pt_key: [300, 1500], msd_key: [110, 140]},
]

var_labels = {pt_key: "pT", msd_key: "mSD"}


cuts_dict = {}
cut_labels = {}  # labels for plot titles, formatted as "var1label: [min, max] var2label..."

for sample, events in events_dict.items():
    cuts_dict[sample] = {}
    for cutvars in all_cuts:
        cutstrs = []
        cutlabel = []
        cuts = []
        for cutvar, (cutmin, cutmax) in cutvars.items():
            cutstrs.append(f"{cutvar}_{cutmin}_{cutmax}")
            cutlabel.append(f"{var_labels[cutvar]}: [{cutmin}, {cutmax}]")
            cuts.append(events[f"fj_{cutvar}"] >= cutmin)
            cuts.append(events[f"fj_{cutvar}"] < cutmax)

        cutstr = "_".join(cutstrs)
        cut = np.prod(cuts, axis=0)
        cuts_dict[sample][cutstr] = cut.astype(bool)

        if cutstr not in cut_labels:
            cut_labels[cutstr] = " ".join(cutlabel)

for cutstr in cut_labels:
    os.system(f"mkdir -p {args.plot_dir}/{cutstr}")


##############################################
# Histograms
##############################################

hist_plot_vars = {
    "th4q": {
        "title": "Non-MD Th4q",
        "score_label": "fj_PN_H4qvsQCD",
    },
    "thvv4q": {
        "title": "MD THVV4q",
        "score_label": "score_fj_THVV4q",
    },
}

hists = {}

for t, pvars in hist_plot_vars.items():
    hists[t] = {}
    for cutstr in cut_labels:
        hists[t][cutstr] = {}
        plt.figure(figsize=(16, 12))
        plt.suptitle(f"HVV FatJet {pvars['title']} Scores", y=0.95)
        plt.title(cut_labels[cutstr], fontsize=20)

        for sample, colour in [("HHbbVV", "red"), ("qcd", "blue")]:
            _ = plt.hist(
                events_dict[sample][pvars["score_label"]][cuts_dict[sample][cutstr]],
                histtype="step",
                bins=np.linspace(0, 1, 101),
                label=f"{samples[sample]}",
                linewidth=2,
                color=colour,
                density=True,
                weights=events_dict[sample]["weight"][cuts_dict[sample][cutstr]],
            )

            hists[t][cutstr][sample] = np.histogram(
                events_dict[sample][pvars["score_label"]][cuts_dict[sample][cutstr]],
                bins=np.linspace(0, 1, 101),
                weights=events_dict[sample]["weight"][cuts_dict[sample][cutstr]],
                density=True,
            )

        plt.ylabel("Normalized # Jets")
        plt.xlabel(f"PNet {pvars['title']} score")
        plt.legend()
        plt.savefig(
            f"{args.plot_dir}/{cutstr}/{t}_hist_{cutstr}.pdf",
            bbox_inches="tight",
        )

# import pickle
#
# with open(f"{args.plot_dir}/hists.pkl", "wb") as f:
#     pickle.dump(hists, f)


##############################################
# ROCs
##############################################

roc_plot_vars = {
    "th4q": {
        "title": "Non-MD Th4q",
        "score_label": "fj_PN_H4qvsQCD",
        "colour": "orange",
    },
    "thvv4q": {
        "title": "MD THVV4q",
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
