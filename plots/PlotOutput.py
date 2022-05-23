#!/usr/bin/env python

import os
import numpy as np
import uproot
import hist as hist2
import awkward as ak

import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.CMS)

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from labels import label_dict
from plot_utils import *


class PlotOutput:
    def __init__(
        self,
        ifile,
        name,
        odir,
        jet,
        sig,
        bkg,
        isigfile=None,
        dnn=False,
        verbose=False,
        oldpn=False,
        mbranch=None,
        nprongs=None,
    ):
        self.ifile = ifile
        self.dnn = dnn
        self.odir = odir
        self.score = f"{name}_score"
        self.verbose = verbose

        self.pt = "fj_pt"
        self.eta = "fj_eta"
        self.msd = "fj_msoftdrop" if not self.dnn else "fj_sdmass"
        self.mbranch = mbranch
        self.nprongs = "fj_nProngs" if nprongs else None
        if oldpn:
            self.oldpn = (
                "pfParticleNetDiscriminatorsJetTags_H4qvsQCD" if self.dnn else "fj_PN_H4qvsQCD"
            )
        else:
            self.oldpn = None

        self.ptrange = [400, 600]
        if '4q' in label_dict[sig]["label"] or '3q' in label_dict[sig]["label"]:
            self.msdrange = [60, 150]
        else:
            self.msdrange = [30, 150]

        self.jet = jet

        self.sigfile = isigfile

        self.sig = sig
        self.bkg = bkg
        self.bkglabel = label_dict[bkg]["label"]
        self.siglabel = label_dict[sig]["label"]
        self.bkglegend = label_dict[bkg]["legend"]
        self.siglegend = label_dict[sig]["legend"]

        self.events,bkgmask = self.get_events()
        self.get_masks()
        self.percentiles, self.cuts = (
            computePercentiles(
                self.events[self.score][bkgmask].to_numpy(),
                [0.97, 0.99, 0.995],
            )
        )
        self.hists = self.fill_hists()

    def get_events(self):
        branches = [self.pt, self.msd, self.eta]
        if self.oldpn:
            branches.append(self.oldpn)
        if self.mbranch:
            branches.append(self.mbranch)
        if self.nprongs:
            branches.append(self.nprongs)
        branches.append("fj_dR_V")
        branches.append("fj_dR_Vstar")

        # score for signal
        branches.append(self.siglabel)
        branches.append(f"score_{self.siglabel}")

        # score for background
        if self.bkg == "qcd":
            bkglabels = [
                "fj_isQCDb",
                "fj_isQCDbb",
                "fj_isQCDc",
                "fj_isQCDcc",
                "fj_isQCDlep",
                "fj_isQCDothers",
            ]
        elif self.bkg == "qcdnolep":
            bkglabels = [
                "fj_isQCDb",
                "fj_isQCDbb",
                "fj_isQCDc",
                "fj_isQCDcc",
                "fj_isQCDothers",
            ]
        elif "asqcd" in self.bkg:
            bkglabels = [
                "fj_isQCDb",
                "fj_isQCDbb",
                "fj_isQCDc",
                "fj_isQCDcc",
                "fj_isQCDothers",
            ]
        elif self.bkg == "qcd1lep":
            bkglabels = [
                "fj_QCD_label",
            ]
        elif self.bkg == "ttbar":
            bkglabels = [
                "fj_ttbar_bsplit",
                "fj_ttbar_bmerged",
            ]
        elif self.bkg == "ttbarwjets":
            bkglabels = [
                "fj_ttbar_bsplit",
                "fj_ttbar_bmerged",
                "fj_wjets_label",
            ]
        elif self.bkg == "wjets":
            bkglabels = ["fj_wjets_label"]
        elif self.bkg == "qcd_dnn":
            bkglabels = [
                "label_QCD_b",
                "label_QCD_bb",
                "label_QCD_c",
                "label_QCD_cc",
                "label_QCD_others",
            ]
        else:
            bkglabels = []

        if len(bkglabels)>0:
            branches.extend(bkglabels)
            branches.extend([f"score_{label}" for label in bkglabels])
            if self.verbose:
                print("Background labels ",bkglabels)
        else:
            branches.append(self.bkglabel)
            branches.extend([f"score_{self.bkglabel}"])

        if self.verbose:
            print("List of branches read ", branches)

        # pt mask
        mask = f"({self.pt}>200) & ({self.pt}<1500)"
        if self.mbranch:
            if self.bkg == "qcdnolep":
                mask += (
                    f"& ( (((fj_isQCDb==1) | (fj_isQCDbb==1) | (fj_isQCDc==1) | (fj_isQCDcc==1) | (fj_isQCDothers==1)) & ({self.mbranch}<=0)) | "
                    f"(({self.siglabel}==1) & ({self.mbranch}>0)) )"
                )
            elif "asqcd" in self.bkg:
                mask += (
                    f"& ( (((fj_isQCDb==1) | (fj_isQCDbb==1) | (fj_isQCDc==1) | (fj_isQCDcc==1) | (fj_isQCDothers==1)) ) | "
                    f"(({self.siglabel}==1) & ({self.mbranch}>0)) )"
                )
            elif self.bkg == "qcd_old":
                mask += (
                    f"& ( ((fj_isQCD==1) & ({self.mbranch}<=0)) | "
                    f"(({self.siglabel}==1) & ({self.mbranch}>0)) )"
                )
            elif self.bkg == "qcd1lep":
                mask += (
                    f"& ( ((fj_QCD_label==1) & ({self.mbranch}<=0)) | "
                    f"(({self.siglabel}==1) & ({self.mbranch}>0)) )"
                )

        if self.verbose:
            print(f'mask applied {mask}')
                
        # open file
        if self.sigfile:
            events = uproot.concatenate(
                {self.ifile: "Events", self.sigfile: "Events"}, branches, cut=mask
            )
        else:
            events = uproot.open(self.ifile)["Events"].arrays(branches, mask)

        # make sure that the are flat samples
        if ~ak.any(
            (events[self.siglabel] == 1)
            & (events[self.mbranch] > 0)
            & (events[self.mbranch] != 125)
        ):
            print("No mbranch")
            self.mbranch = None

        # bkg label
        # and score
        """
        We expect all scores to sum up to 1, e.g. given two signals in the event (signal 1 and 2) and one background process (background 1)
        score_signal_1 + score_signal_2 + score_background_1 = 1
        Then nn_signal_1 = score_signal_1 / (score_signal_1 + score_background_1) = score_signal_1 / (1 - score_signal_2)
        """
        if len(bkglabels)>0:
            events["fj_bkg_label"] = (
                (np.sum([events[label] for label in bkglabels], axis=0).astype(bool).squeeze())
                if len(bkglabels) > 1
                else np.array(events[bkglabels[0]]).astype(bool)
            )
            if not ak.any(events["fj_bkg_label"] == 1):
                print(f"WARNING: NO BKG WITH bkg label formed by ",bkglabels)
                exit()
            print(f"Score defined as score_{self.siglabel}/(score_bkglabels) with bkglabels: ", bkglabels)
            score_branch = events[f"score_{self.siglabel}"] / (
                events[f"score_{self.siglabel}"]
	        + np.sum([events[f"score_{bkglabel}"] for bkglabel in bkglabels], axis=0).squeeze()
	    )
            # define bkg_mask in events for computing percentiles
            bkgmask = (events["fj_bkg_label"]==1)
        else:            
            if not ak.any(events[self.bkglabel] == 1):
                print(f"WARNING: NO BKG WITH {self.bkglabel}")
                exit()
            print(
                f"Score defined as score_{self.siglabel}/(score_{self.siglabel}+score_{self.bkglabel}"
            )
            score_branch = events[f"score_{self.siglabel}"] / (
                events[f"score_{self.siglabel}"] + events[f"score_{self.bkglabel}"]
            )
            bkgmask = (events[self.bkglabel]==1)

        events[self.score] = score_branch
        
        if self.verbose:
            print(self.score, events[self.score])

        return events,bkgmask

    def get_masks(self):
        def build_range(bins, var, mask, events, branch):
            rangemask = {}
            for i, ibin in enumerate(bins):
                if i == len(bins) - 1:
                    continue
                bin_low = bins[i]
                bin_high = bins[i + 1]
                bin_tag = f"{var}:{bin_low}-{bin_high}"
                bin_mask = mask & (events[branch] > bin_low) & (events[branch] < bin_high)
                if ak.any(bin_mask):
                    rangemask[bin_tag] = bin_mask
                else:
                    print(f"No events for mask {bin_tag}")
            return rangemask

        self.roc_mask_nomass = (
            (self.events[self.pt] > self.ptrange[0])
            & (self.events[self.pt] < self.ptrange[1])
            & (abs(self.events[self.eta]) < 2.4)
        )
        self.roc_mask = (
            self.roc_mask_nomass
            & (self.events[self.msd] > self.msdrange[0])
            & (self.events[self.msd] < self.msdrange[1])
        )
        if self.mbranch:
            self.mask_flat = self.events[self.mbranch] != 125
            self.mask_mh125 = self.events[self.mbranch] == 125
            self.mask_proc_mh120130 = (
                (self.events[self.mbranch] >= 120)
                & (self.events[self.mbranch] <= 130)
                & self.mask_flat
            )
            self.mask_proc_sigmh125 = (self.events[self.siglabel] == 1) & self.mask_mh125
            self.mask_proc_sig = (self.events[self.siglabel] == 1) & self.mask_flat
            self.mhmasks = build_range(
                list(range(20, 240, 20)), "mH", self.roc_mask_nomass, self.events, self.mbranch
            )
        else:
            print("No mbranch present")
            self.mask_flat = np.zeros(len(self.events), dtype="bool")
            self.mask_mh125 = np.zeros(len(self.events), dtype="bool")
            self.mask_proc_mh120130 = np.zeros(len(self.events), dtype="bool")
            self.mask_proc_sigmh125 = np.zeros(len(self.events), dtype="bool")
            self.mask_proc_sig = np.zeros(len(self.events), dtype="bool")

        self.mask_vcloser = self.events["fj_dR_V"] > self.events["fj_dR_Vstar"]
        self.mask_vscloser = ~self.mask_vcloser
        if not ak.any(self.mask_vcloser):
            print("Vstar is always closer than V")
        if not ak.any(self.mask_vscloser):
            print("V is always closer than Vstar")

        if self.nprongs:
            self.mask3p = self.events[self.nprongs] == 3
            self.mask4p = self.events[self.nprongs] == 4

        # should we add the msD mask here?
        self.ptmasks = build_range(
            list(range(200, 1200, 200)),
            "pT",
            (abs(self.events[self.eta]) < 2.4),
            self.events,
            self.pt,
        )

    def get_weights(self):
        # log bins as: np.round(np.exp(np.linspace(np.log(MIN), np.log(MAX), NUM_BINS))).astype('int').tolist()
        ptbins = [200, 251, 316, 398, 501, 630, 793, 997, 1255, 1579, 1987, 2500]
        pthistsigmh, bin_edges = np.histogram(
            self.events[self.pt][self.mask_proc_sigmh125].to_numpy(), bins=ptbins
        )
        ptflat = self.events[self.pt][self.mask_proc_sig]
        pthistsig, bin_edges = np.histogram(ptflat.to_numpy(), bins=ptbins)
        pthist = pthistsigmh / pthistsig
        weight_sig = pthist[
            np.digitize(self.events[self.pt][self.mask_proc_sig].to_numpy(), ptbins) - 1
        ]

        # plots
        for density in [True]:  # False]:
            fig, axs = plt.subplots(1, 1)
            axs.hist(ptflat, bins=ptbins, histtype="step", density=density)
            axs.hist(ptflat, bins=ptbins, weights=weight_sig, histtype="step", density=density)
            axs.set_xlabel("pT (GeV)")
            axs.legend(["Unweighted", "Weighted"])
            tag = self.siglabel + "_density" if density else self.siglabel
            fig.savefig(f"%s/ptweights_%s.pdf" % (self.odir, tag))
            fig.savefig(f"%s/ptweights_%s.png" % (self.odir, tag))

        return pthist, ptbins, weight_sig

    def roc(self, score, option=None, sigmask=None):
        if option == "ratio":
            roc = get_roc(
                self.events,
                score,
                self.siglabel,
                self.bkglabel,
                ratio=True,
                mask_num=self.roc_mask,
                mask_den=self.roc_mask_nomass,
            )
        elif option == "nomass":
            roc = get_roc(
                self.events, score, self.siglabel, self.bkglabel, roc_mask=self.roc_mask_nomass
            )
        elif option == "sigmask":
            roc = get_roc(
                self.events,
                score,
                self.siglabel,
                self.bkglabel,
                roc_mask=self.roc_mask,
                sig_mask=sigmask,
            )
        elif option == "norocmask":
            roc = get_roc(
                self.events, score, self.siglabel, self.bkglabel, roc_mask=None, sig_mask=sigmask
            )
        elif option == "ptweight":
            if self.mbranch and ak.any(self.mask_proc_sigmh125):
                pthist, ptbins, weight_sig = self.get_weights()
                roc = get_roc(
                    self.events,
                    score,
                    self.siglabel,
                    self.bkglabel,
                    weight_hist=pthist,
                    bins=ptbins,
                    roc_mask=self.roc_mask,
                    sig_mask=sigmask,
                )
        else:
            roc = get_roc(self.events, score, self.siglabel, self.bkglabel, roc_mask=self.roc_mask)
        return roc

    def plot(self, fp_tp, tag, keys, ptcuts=None, msdcuts=None):
        if not ptcuts:
            ptcuts = [self.ptrange[0], self.ptrange[1]]
        if not msdcuts:
            msdcuts = [self.msdrange[0], self.msdrange[1]]
        title = r"%s vs %s" % (self.siglegend, self.bkglegend)
        label = f"{self.siglabel}vs{self.bkglabel}"
        ptcut = r"%s $p_T$:[%s-%s] GeV, $|\eta|<2.4$" % (self.jet, ptcuts[0], ptcuts[1])
        msdcut = r"%s $m_{SD}$:[%s-%s] GeV" % (self.jet, msdcuts[0], msdcuts[1])
        plot_roc(
            self.odir,
            self.siglegend,
            self.bkglegend,
            fp_tp,
            label=label + "_" + tag,
            title=title,
            pkeys=keys,
            ptcut=ptcut,
            msdcut=msdcut,
        )

    def fill_hists(self):
        hists = {}
        hists["features"] = hist2.Hist(
            hist2.axis.StrCategory([], name="process", growth=True),
            hist2.axis.StrCategory([], name="cat", growth=True),
            hist2.axis.Regular(
                23, 30, 260, name="msd", label=r"fj msoftdrop [GeV]"
            ),  # bins of 10 gev
            hist2.axis.Regular(50, 200, 1200, name="pt", label=r"fj $p_T$ [GeV]"),
            hist2.axis.Regular(1000, 0, 1, name="score", label=r"Tagger score"),
        )
        if self.mbranch:
            hists["genmass"] = hist2.Hist(
                hist2.axis.StrCategory([], name="process", growth=True),
                hist2.axis.Regular(100, 0, 1, name="score", label=r"Tagger score"),
                hist2.axis.Regular(50, 200, 1200, name="pt", label=r"fj $p_T$ [GeV]"),
                hist2.axis.Regular(46, 30, 260, name="mh", label=r"gen Res mass [GeV]"),
            )

        processes = [self.bkg]
        if self.mbranch:
            processes.append(f"{self.sig}-mh125")
            processes.append(f"{self.sig}-mhflat")
        else:
            processes.append(self.sig)

        legends = {}
        for proc in processes:
            p = proc.split("-")[0]
            legend = label_dict[p]["legend"]
            if "mh125" in proc:
                mask_proc = (self.events[label_dict[p]["label"]] == 1) & (
                    self.events[self.mbranch] == 125
                )
                legend += " mh125"
            elif "mhflat" in proc:
                mask_proc = (self.events[label_dict[p]["label"]] == 1) & (
                    self.events[self.mbranch] != 125
                )
                legend += " mhflat"
            else:
                mask_proc = self.events[label_dict[p]["label"]] == 1
                legends[proc] = legend

            # check if events with that mask are not zero
            if len(self.events[self.msd][mask_proc]) == 0:
                processes.remove(proc)
                continue

            # fill histogram
            for i, cat in enumerate(self.percentiles):
                # print(cat,self.cuts[i])
                mask_cat = (
                    mask_proc & (self.events[self.score] >= self.cuts[i]) & self.roc_mask_nomass
                )
                hists["features"].fill(
                    process=proc,
                    cat=str(cat),
                    msd=self.events[self.msd][mask_cat],
                    pt=self.events[self.pt][mask_cat],
                    score=self.events[self.score][mask_cat],
                )

            if self.sig in proc and "genmass" in hists.keys() and self.mbranch:
                hists["genmass"].fill(
                    process=proc,
                    mh=self.events[self.mbranch][mask_proc],
                    score=self.events[self.score][mask_proc],
                    pt=self.events[self.pt][mask_proc],
                )
        return hists
