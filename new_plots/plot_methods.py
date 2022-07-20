import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
import pickle

plt.rcParams.update({"font.size": 16})
plt.style.use(hep.style.CMS)
hep.style.use("CMS")

from typing import Dict, List, Tuple


def create_branches(
    events_dict: Dict[str, ak.Array], sum_scores: Dict[str, List], T_scores: Dict[str, Tuple]
):
    """
    Creates new branches in events in ``events_dict``.

    ``sum_scores`` defines new branches which sum scores of existing branches.
    ``T_scores`` defines new branches which = sig scores / (sig + bg scores).

    Args:
        events_dict (Dict[str, ak.Array]): {sample: awkward array of events}.
        sum_scores (Dict[str, List]): {new branch label: [score branches to sum over]}.
        T_scores (Dict[str, Tuple]): {new branch label: (sig, bg)}.

    """

    for sample, events in events_dict.items():
        for label, subclasses in sum_scores.items():
            events[f"score_fj_{label}"] = np.sum(
                [events[f"score_fj_{subclass}"] for subclass in subclasses], axis=0
            )

        for label, (sig, bg) in T_scores.items():
            events[f"score_fj_T{label}"] = events[f"score_fj_{sig}"] / (
                events[f"score_fj_{sig}"] + events[f"score_fj_{bg}"]
            )

        events["weight"] = np.ones(len(events))


def reweight_qcd_pt(events: ak.Array):
    """Reweights QCD pT to be exponentially falling"""
    from coffea.lookup_tools.dense_lookup import dense_lookup
    from hist import Hist

    reweight_bins = [120, 300, 1500]
    bin_centers = np.linspace(reweight_bins[1], reweight_bins[2], reweight_bins[0] + 1)[:-1]
    bin_centers += (bin_centers[1] - bin_centers[0]) / 2
    # qcd pt 1/e range is roughly ~100 GeV
    exp_falling_weights = np.exp((reweight_bins[1] - bin_centers) / 100)

    qcd_pt = (Hist.new.Reg(*reweight_bins, name="pt").Double()).fill(
        pt=events["fj_pt"],
    )

    pt_weights = exp_falling_weights / qcd_pt.view(flow=False)
    pt_weights_lookup = dense_lookup(pt_weights, qcd_pt.axes.edges)

    events["weight"] = pt_weights_lookup(events["fj_pt"])


def make_cuts(events_dict: Dict[str, ak.Array], cuts_list: List[Dict], var_labels: Dict[str, str]):
    """
    Makes cuts on events in ``events_dict``.

    ``cuts_list`` is a list of cut Dicts, each of format:
    {
        var1: [min, max],
        var2: ...,
    }

    ``var_labels`` is a Dict of formatted labels for each cut var, for plotting.
    {
        var1: label,
        var2: ...
    }

    Returns 1) a dictionary of cuts of format:
    {
        sample1: {
            "cut1var1_min_max_cut1var2...": cut1,
            "cut2var2...": cut2,
            ...
        },
        sample2...
    }

    and 2) a dictionary of labels for plots of format:
    {
        "cut1var1_min_max_cut1var2...": "var1label: [min, max] var2label...",
        "cut2var1...": ...
    }

    """
    cuts_dict = {}
    cut_labels = {}

    for sample, events in events_dict.items():
        cuts_dict[sample] = {}
        for cutvars in cuts_list:
            cutstrs = []
            cutlabel = []
            cuts = []
            for cutvar, (cutmin, cutmax) in cutvars.items():
                cutstrs.append(f"{cutvar}_{cutmin}_{cutmax}")
                cutlabel.append(f"{var_labels[cutvar]}: [{cutmin}, {cutmax}] GeV")
                cuts.append(events[f"fj_{cutvar}"] >= cutmin)
                cuts.append(events[f"fj_{cutvar}"] < cutmax)

            cutstr = "_".join(cutstrs)
            cut = np.prod(cuts, axis=0)
            cuts_dict[sample][cutstr] = cut.astype(bool)

            if cutstr not in cut_labels:
                cut_labels[cutstr] = " ".join(cutlabel)

    return cuts_dict, cut_labels


def plot_hists(
    hist_plot_vars: Dict[str, Dict],
    events_dict: Dict[str, ak.Array],
    cuts_dict: Dict[str, Dict],
    cut_labels: Dict[str, str],
    sample_labels: Dict[str, str],
    plot_dir: str,
    samples_colours: List[Tuple] = [("HHbbVV", "red"), ("qcd", "blue")],
    save_hists: bool = False,
):
    """
    Plots and (optionally) saves histograms.

    ``hist_plot_vars`` is a dictionary of histograms to make formatted as:
    {
        'tag': {
            'title': plot title,
            'score_label': scores to make histograms of,
        }
    }
    """

    hists = {}

    for t, pvars in hist_plot_vars.items():
        hists[t] = {}
        for cutstr in cut_labels:
            hists[t][cutstr] = {}
            plt.figure(figsize=(16, 12))
            plt.suptitle(f"FatJet {pvars['title']} Scores", y=0.95)
            plt.title(cut_labels[cutstr], fontsize=20)

            for sample, colour in samples_colours:
                _ = plt.hist(
                    events_dict[sample][pvars["score_label"]][cuts_dict[sample][cutstr]],
                    histtype="step",
                    bins=np.linspace(0, 1, 101),
                    label=f"{sample_labels[sample]}",
                    linewidth=2,
                    color=colour,
                    density=True,
                    weights=events_dict[sample]["weight"][cuts_dict[sample][cutstr]],
                )

                if save_hists:
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
                f"{plot_dir}/{cutstr}/{t}_hist_{cutstr}.pdf",
                bbox_inches="tight",
            )

    if save_hists:
        with open(f"{plot_dir}/hists.pkl", "wb") as f:
            pickle.dump(hists, f)
