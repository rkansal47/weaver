#!/usr/bin/env python

import os
import argparse
import numpy as np
import uproot
import awkward as ak

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.CMS)

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from PlotOutput import PlotOutput

from labels import label_dict
from plot_utils import *


def get_rocs(p, args, jsig=0, signame=""):
    fp_tp = {}
    score = p.score
    fp_tp[score + signame] = p.roc(score)
    fp_tp[score + signame + "-ratio"] = p.roc(score, "ratio")
    if args.oldpn and jsig == 0:
        fp_tp["PNnonMD"] = p.roc(args.oldpn)
        fp_tp["PNnonMD-ratio"] = p.roc(args.oldpn, "ratio")
        fp_tp["PNnonMD-nomasscut"] = p.roc(args.oldpn, "nomass")
    # add pt ranges
    for key, ptmask in p.ptmasks.items():
        fp_tp[key] = p.roc(score, "norocmask", ptmask)
    # add v/v* rocs
    fp_tp[score + signame + "_V_closer_Jet"] = p.roc(score, "sigmask", p.mask_vcloser)
    fp_tp[score + signame + "_V*_closer_Jet"] = p.roc(score, "sigmask", p.mask_vscloser)

    if args.mbranch:
        # for flat sample
        if ak.any(p.mask_flat):
            fp_tp[r"$m_H$ flat"] = p.roc(score, "sigmask", p.mask_flat)
            if ak.any(p.mask_proc_sigmh125):
                fp_tp[r"$m_H$ flat $\times p_T^SM$"] = p.roc(score, "ptweight", p.mask_flat)
            if args.oldpn:
                fp_tp[r"$m_H$ flat PNnonMD"] = p.roc(args.oldpn, "ptweight", p.mask_flat)
            # add mh ranges
            for key, mhmask in p.mhmasks.items():
                fp_tp[key] = p.roc(score, "norocmask", mhmask)
        if ak.any(p.mask_proc_mh120130) and ak.any(p.mask_proc_sigmh125):
            fp_tp[r"$m_H:[120,130]$ GeV $\times p_T^SM$"] = p.roc(
                score, "ptweight", p.mask_proc_mh120130
            )
        if args.nprongs:
            fp_tp[r"$m_H$ flat 3prongs"] = p.roc(score, "sigmask", (p.mask_flat & p.mask3p))
            fp_tp[r"$m_H$ flat 4prongs"] = p.roc(score, "sigmask", (p.mask_flat & p.mask4p))
        # for mh125
        if ak.any(p.mask_mh125):
            fp_tp[r"$m_H$:125"] = p.roc(score, "sigmask", p.mask_mh125)
        if args.oldpn:
            fp_tp[r"$m_H$:125 PNnonMD"] = p.roc(args.oldpn, "sigmask", p.mask_mh125)
        if args.nprongs:
            fp_tp[r"$m_H$:125 3prongs"] = p.roc(score, "sigmask", (p.mask_mh125 & p.mask3p))
            fp_tp[r"$m_H$:125 4prongs"] = p.roc(score, "sigmask", (p.mask_mh125 & p.mask4p))

    else:
        if args.nprongs:
            fp_tp["3prongs"] = p.roc(score, "sigmask", p.mask3p)
            fp_tp["4prongs"] = p.roc(score, "sigmask", p.mask4p)

    return fp_tp


def main(args):
    signals = args.signals.split(",")
    backgrounds = args.bkgs.split(",")
    if len(signals) != len(backgrounds):
        print("Number of signals should be the same as backgrounds!")
        exit

    fp_tp_all = {}
    for i, sig in enumerate(signals):
        bkg = backgrounds[i]
        bkglegend = label_dict[bkg]["legend"]
        odir = args.odir + sig
        os.system(f"mkdir -p {odir}")

        sigfiles = None
        if args.isig:
            sigfiles = args.isig.split(",")
        else:
            sigfiles = [None]

        fp_tp_sigfiles = {}
        plot_keys = {
            "sigfiles": [],
            "ratio": [],
            "pt": [],
            "mh": [],
            "closer": [],
        }
        fillmh = False

        for j, sigfile in enumerate(sigfiles):
            p = PlotOutput(
                args.ifile,
                args.name,
                odir,
                args.jet,
                sig,
                bkg,
                sigfile,
                args.dnn,
                args.verbose,
                args.oldpn,
                args.mbranch,
                args.nprongs,
            )

            # keys of ROCS to plot
            signame = ""
            if args.isig:
                signame = " " + args.isignames.split(",")[j]

            fp_tp = get_rocs(p, args, signame)
            if args.isig:
                for key,item in fp_tp.items():
                    fp_tp_sigfiles[key+signame] = item
                    if key==p.score:
                        if not p.mbranch:
                            plot_keys["sigfiles"].append(key+signame)
                        if j==0:
                            fp_tp_all[sig] = item
                    elif key == p.score + "-ratio":
                        plot_keys["ratio"].append(key + signame)
                    elif "closer" in key:
                        plot_keys["closer"].append(key + signame)
                    elif "pT:" in key:
                        if j==0:
                            plot_keys["pt"].append(key+signame)
                    elif "mH:" in key and p.mbranch:
                        if key not in plot_keys["mh"]:
                            plot_keys["mh"].append(key+signame)
                    elif "m_H" in key and p.mbranch and not fillmh:
                        # only do this for the first sample that has a range of mh
                        plot_keys["sigfiles"].append(key + signame)
                        fillmh = True
                    else:
                        print("not plotting ", key)
            else:
                fp_tp_sigfiles = fp_tp
                fp_tp_all[sig] = fp_tp[p.score]
                for key,item in fp_tp.items():
                    if key==p.score and not p.mbranch:
                        plot_keys["sigfiles"].append(key)     
                    if key == p.score + "-ratio":
                        plot_keys["ratio"].append(key)
                    elif "closer" in key:
                        plot_keys["closer"].append(key)
                    elif "-pt" in key:
                        plot_keys["pt"].append(key)
                    elif "mH:" in key and p.mbranch:
                        plot_keys["mh"].append(key)
                    elif p.mbranch:
                        plot_keys["sigfiles"].extend([r"$m_H$ flat", r"$m_H$:125"])
                    elif args.oldpn:
                        plot_keys["sigfiles"] += ["PNnonMD"]

        if args.verbose:
            print(fp_tp_sigfiles.keys())

        # plot ROCs
        p.plot(fp_tp_sigfiles, "summary", plot_keys["sigfiles"])
        p.plot(fp_tp_sigfiles, "ratio", plot_keys["ratio"])

        # plot for different mH and pT cuts
        p.plot(fp_tp_sigfiles, "pt", plot_keys["pt"], [300, 1500], [30, 320])
        if len(plot_keys["mh"]) > 0:
            p.plot(fp_tp_sigfiles, "mh", plot_keys["mh"], [400, 600], [30, 320])

        # plot ROCs for different W/W* distance
        p.plot(fp_tp_sigfiles, "closeVVstar", plot_keys["closer"])

        # mass decorrelation
        hists_to_plot = []
        labels_to_plot = []
        for i, cat in enumerate(p.percentiles):
            hists_to_plot.append(
                p.hists["features"][{"process": bkg, "cat": str(cat)}].project("msd")
            )
            if cat == 0:
                labels_to_plot.append("Inclusive")
            else:
                per = round((1 - cat) * 100, 2)
                tag = f"$\epsilon_B={per} \%$"
                labels_to_plot.append(r"%s" % tag)

        ptcut = r"%s $p_T$:[%s-%s] GeV, $|\eta|<2.4$" % (p.jet, p.ptrange[0], p.ptrange[1])
        plot_var_aftercut(odir, hists_to_plot, labels_to_plot, r"m$_{SD}$ [GeV]", f"msd_{bkg}", ptcut)

    if len(signals) > 1:
        # plot summary ROC for all signal classes and first isig file
        plot_roc(
            args.odir,
            "HWW",
            "Bkg",
            fp_tp_all,
            label="allsig_summary",
            title=f"HWW vs %s"%bkg,
            ptcut="%s $p_T$:[400-600] GeV, $|\eta|<2.4$"%(args.jet),
            msdcut="$m_{SD}$:[%s-%s] GeV"%(p.msdrange[0],p.msdrange[1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifile", help="input file with bkg and possibly signal")
    parser.add_argument("--isig", default=None, help="signal files - split by ,")
    parser.add_argument("--isignames", default=None, help="signal tags - split by ,")
    parser.add_argument("--odir", required=True, help="output dir")
    parser.add_argument("--name", required=True, help="name of the model(s)")
    parser.add_argument("--mbranch", default=None, help="mass branch name")
    parser.add_argument("--jet", default="AK15", help="jet type")
    parser.add_argument("--dnn", action="store_true", default=False, help="is dnn tuple?")
    parser.add_argument(
        "--oldpn", action="store_true", default=False, help="is oldpn branch saved?"
    )
    parser.add_argument("-v", action="store_true", default=False, help="verbose", dest="verbose")
    parser.add_argument(
        "--nprongs", action="store_true", default=False, help="is nprongs branch saved?"
    )
    parser.add_argument("--signals", default="hww_4q_merged", help="signals")
    parser.add_argument(
        "--bkgs",
        default="qcd",
        help="backgrounds (if qcd_label then assume that you only have one qcd label)",
    )
    args = parser.parse_args()

    main(args)
