from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

from coffea import hist
    
def roc_input(events,var,label_sig,label_bkg,
              weight_hist=None,bins=None,
              sig_mask=None,bkg_mask=None,roc_mask=None):
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
    mask_sig = (events[label_sig] == 1)
    mask_bkg = (events[label_bkg] == 1)
    if sig_mask is not None:
        mask_sig = (mask_sig) & (sig_mask) 
    if bkg_mask is not None:
        mask_bkg = (mask_bkg) & (bkg_mask)
    if roc_mask is not None:
        mask_sig = (mask_sig) & (roc_mask)
        mask_bkg = (mask_bkg) & (roc_mask)
    scores_sig = events[var][mask_sig].to_numpy()
    scores_bkg = events[var][mask_bkg].to_numpy()
    predict = np.concatenate((scores_sig,scores_bkg),axis=None)
    siglabels = np.ones(scores_sig.shape)
    bkglabels = np.zeros(scores_bkg.shape)
    truth = np.concatenate((siglabels,bkglabels),axis=None)

    weight = None
    if weight_hist is not None:
        weight_sig = weight_hist[np.digitize(events['fj_pt'][mask_sig].to_numpy(), bins)-1]
        weight_bkg = np.ones(scores_bkg.shape)
        weight = np.concatenate((weight_sig,weight_bkg),axis=None)

    return truth, predict, weight
    
def get_roc(events, score_name, label_sig, label_bkg,
            weight_hist=None,bins=None,sig_mask=None,bkg_mask=None,roc_mask=None,
            ratio=False,mask_num=None,mask_den=None):
    """
    Get both the ROC input and the curve
    Arguments:
    - inputs for roc_input
    """
    if ratio:
        truth, predict, weight = roc_input(events,score_name,label_sig,label_bkg)
        fprs, tprs, threshold = roc_curve(truth, predict, sample_weight=weight)
        weight_sig = np.sum(mask_num & (events[label_sig] == 1))/np.sum(mask_den & (events[label_sig] == 1))
        weight_bkg = np.sum(mask_num & (events[label_bkg] == 1))/np.sum(mask_den & (events[label_bkg] == 1))
        fprs = fprs*weight_bkg
        tprs = tprs*weight_sig
    else:
        truth, predict, weight = roc_input(events,score_name,label_sig,label_bkg, weight_hist,bins,sig_mask,bkg_mask,roc_mask)
        fprs, tprs, threshold = roc_curve(truth, predict, sample_weight=weight)

    return (fprs, tprs)

def plot_roc(odir, label_sig, label_bkg,
             fp_tp, label,
             title,
             pkeys=None,
             ptcut=None, msdcut=None,
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

    fig, axs = plt.subplots(1,1,figsize=(16,16))

    def get_round(x_effs,y_effs,to_get=[0.01,0.02,0.03]):
        effs = []
        for eff in to_get:
            for i,f in enumerate(x_effs):
                if round(f,2) == eff:
                    effs.append(y_effs[i])
                    break
        return effs

    def get_intersections(x_effs, y_effs, to_get=0.01):
        x_eff = 0
        for i,f in enumerate(y_effs):
            if f >= to_get:
                x_eff = x_effs[i]
                break
        return x_eff

    # draw intersections at 1% mis-tag rate
    ik = 0
    markers = ['v','^','o','s','p','P','h']
    for k,it in fp_tp.items():
        if pkeys and k not in pkeys:
            continue
        leg = k.replace('_score','')
        fp = it[0]
        tp = it[1]
        axs.plot(tp, fp, lw=3, label=r"{}, AUC = {:.1f}%".format(leg,auc(fp,tp)*100))
        y_eff = 0.01
        x_eff = get_intersections(tp, fp, y_eff)
        axs.hlines(
            y=y_eff, xmin=0.00001, xmax=0.99999, linewidth=1.3, color='dimgrey',linestyle="dashed"
        )
        axs.vlines(
            x=x_eff, ymin=0.00001, ymax=y_eff, linewidth=1.3, color='dimgrey',linestyle="dashed"
        )
        ik+=1

    axs.legend(loc='lower right',fontsize=40)
    axs.grid(which='minor', alpha=0.2)
    axs.grid(which='major', alpha=0.5)
    axs.set_xlabel(r'Tagging efficiency %s'%label_sig, fontsize=40)
    axs.set_ylabel(r'Mistagging rate %s'%label_bkg, fontsize=40)
    axs.set_ylim(0.0001,1)
    axs.set_xlim(0.0001,1)
    axs.set_yscale('log')
    if ptcut: axs.text(0.05, 0.5, ptcut, fontsize=30)
    if msdcut: axs.text(0.05, 0.3, msdcut, fontsize=30)
    axs.set_title(title, fontsize=40)

    plt.tight_layout()

    fig.savefig("%s/roc_%s_ylog.pdf"%(odir,label))
    fig.savefig("%s/roc_%s_ylog.png"%(odir,label))

    # also save version down to 10^-5
    axs.set_ylim(0.00001,1)
    axs.set_xlim(0.0001,1)
    fig.savefig("%s/roc_%s_ylogm5.pdf"%(odir,label))
    fig.savefig("%s/roc_%s_ylogm5.png"%(odir,label))
    
    axs.set_yscale('linear')

def plot_roc_by_var(odir,vars_to_corr,
                    bin_ranges,bin_widths,
                    events,score_name,sig,bkg,
                    label,title,
                    sig_mask=None):
    """
    Plot ROC for different cuts on variable var
    Arguments:
    - odir: output directory
    - vars_to_corr: variables used to make a selection (usually pT and mSD)
    - bin_ranges: bin ranges that define upper and lower boundaries for cuts on that variable
    - bin_widths: widths that define cuts on that variable
    - events: table with vars_to_corr
    - score_name: name of discriminator
    - sig: dictionary with label and legend for signal
    - bkg: dictionary with label and legend for background
    - label: label of plot
    - sig_mask: extra mask on signal e.g. to select on mH=125
    """
    i = 0
    fig, axs = plt.subplots(1,len(vars_to_corr.keys()),figsize=(8*len(vars_to_corr.keys()),8))
    for var,varname in vars_to_corr.items():
        fprs = {}; tprs = {}
        legends = []
        for j,b in enumerate(bin_ranges[i]):
            bi = b
            bf = b+bin_widths[i]
            tag = "%i"%bi
            signal_mask = (events[varname]>=bi) & (events[varname]<=bf)
            bkgnd_mask = (events[varname]>=bi) & (events[varname]<=bf)
            if sig_mask is not None:
                signal_mask = signal_mask & sig_mask
            fprs[tag], tprs[tag] = get_roc(events, score_name, sig['label'], bkg['label'], sig_mask=signal_mask, bkg_mask=bkgnd_mask)
            legends.append('%s:%i-%i GeV'%(var,bi,bf))

        if(len(vars_to_corr.keys())==1):
            axs_1 = axs
        else:
            axs_1 = axs[i]

        ik=0
        for k,it in fprs.items():
            axs_1.plot(tprs[k], fprs[k], lw=2.5, label=r"{}, AUC = {:.1f}%".format(legends[ik],auc(fprs[k],tprs[k])*100))
            ik+=1
        axs_1.legend(loc='upper left',fontsize=20)
        axs_1.grid(which='minor', alpha=0.2)
        axs_1.grid(which='major', alpha=0.6)
        axs_1.set_xlabel(r'Tagging efficiency %s'%sig['legend'], fontsize=25)
        axs_1.set_ylabel(r'Mistagging rate %s'%bkg['legend'], fontsize=25)
        axs_1.set_yscale('log')
        axs_1.set_title(title, fontsize=20)
        axs_1.set_ylim(0.0001,1)
        i+=1
    plt.tight_layout()

    fig.savefig("%s/rocs_by_var_%s_ylog.pdf"%(odir,label))
    fig.savefig("%s/rocs_by_var_%s_ylog.png"%(odir,label))

def plot_1d(odir,hist_val,vars_to_plot,label):
    """
    Plot 1d histograms for validation
    Arguments:
    - odir: output directory
    - hist_val: histogram
    - vars_to_plot: variables to project
    - label: label of plot
    """
    for density in [True,False]:
        fig, axs = plt.subplots(1,len(vars_to_plot), figsize=(len(vars_to_plot)*8,8))
        for i,m in enumerate(vars_to_plot):
            if(len(vars_to_plot)==1):
                axs_1 = axs
            else:
                axs_1 = axs[i]
            x = hist_val.sum(*[ax for ax in hist_val.axes() if ax.name not in {'process',m}])
            hist.plot1d(x,ax=axs_1,overlay="process",density=density)
            axs_1.set_ylabel('Jets')
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
    mincut = 0.
    tmp = np.quantile(data, np.array(percentiles))
    tmpl = [mincut]
    for x in tmp:
        tmpl.append(x)
    perc = [0.]
    for x in percentiles:
        perc.append(x)
    return perc, tmpl

def plot_score_aftercut(odir,hist_val,vars_to_corr,bin_ranges,bin_widths,processes,label):
    density = True
    for proc in processes:
        fig, axs = plt.subplots(1,len(vars_to_corr), figsize=(len(vars_to_corr)*8,8))
        for i,m in enumerate(vars_to_corr):
            if(len(vars_to_corr)==1):
                axs_1 = axs
            else:
                axs_1 = axs[i]
            x = hist_val.sum(*[ax for ax in hist_val.axes() if ax.name not in {'process','score',m}]).integrate("process",proc)
            legends = []
            for j,b in enumerate(bin_ranges[i]):
                y = x.integrate(m, slice(b,b+bin_widths[i]))
                legends.append('%s %i-%i GeV'%(m,b,b+bin_widths[i]))
                if j==0:
                    hist.plot1d(y,ax=axs_1,density=True)
                else:
                    hist.plot1d(y,ax=axs_1,density=True,clear=False)
            axs_1.set_ylabel('Jets')
            axs_1.legend(legends,title=m)
        fig.tight_layout()
        fig.savefig("%s/%s_scores_%s_density.pdf"%(odir,proc,label))
        fig.savefig("%s/%s_scores_%s_density.png"%(odir,proc,label))

def plot_var_aftercut(odir,hist_val,vars_to_plot,processes,label,cuts,percentiles):
    for proc in processes:
        fig, axs = plt.subplots(nrows=2,ncols=len(vars_to_plot), figsize=(len(vars_to_plot)*8,8),  gridspec_kw={"height_ratios": (4, 1)}, sharex=True)
        fig.subplots_adjust(hspace=.07)
        for i,var in enumerate(vars_to_plot):
            if(len(vars_to_plot)==1):
                axs_1 = axs[0]
                raxs_1 = axs[1]
            else:
                axs_1 = axs[0,i]
                raxs_1 = raxs[1,i]
            x = hist_val.sum(*[ax for ax in hist_val.axes() if ax.name not in {'process',var,'score'}])
            x = x.integrate('process',proc)
            
            legends = []
            hists = []
            hscales = []
            for i,cut in enumerate(cuts):
                cut = round(cut,2)
                y = x.integrate('score',slice(cut,1))
                hists.append(y)
                if i==0:
                    legends.append(r'Inclusive')
                else:
                    legends.append(r'$\epsilon_B=$%.1f'%((1-percentiles[i])*100)+' %')
                hist.plot1d(y,ax=axs_1,density=True,clear=False)

                val,_ = y.values(sumw2=True, overflow="none")[()]
                s = np.sum(y.values(sumw2=True, overflow="none")[()])
                y.scale(1/s)
                hscales.append(y)
                
            #x = hist_val.sum(*[ax for ax in hist_val.axes() if ax.name not in {'process',var,'pn_score'}])
            #x = x.integrate('process',proc)
            #y = x.integrate('pn_score',slice(cut,0.98))
            #hist.plot1d(y,ax=axs_1,density=True,clear=False)
            #legends.append(r'Old PN $\epsilon_B=2%$')
            
            axs_1.grid()
            axs_1.set_ylabel('A.U.')

            error_opts = {
                'linestyle': 'none',
                'marker': '.',
                'markersize': 10.,
                'elinewidth': 1,
            }
            for i,h in enumerate(hscales):
                hist.plotratio(num=h,denom=hscales[0],error_opts=error_opts,ax=raxs_1,unc="num",clear=False)
            raxs_1.grid(which='minor', alpha=0.4)
            raxs_1.grid(which='major', alpha=0.6)
            raxs_1.set_ylabel('Ratio')
            raxs_1.set_ylim(0.5,1.5)

            axs_1.set_xlabel("")
            axs_1.legend(legends)
            axs_1.get_shared_x_axes().join(axs_1, raxs_1)
            axs_1.set_title("Mass of QCD jets after tagging", fontsize=25)
        fig.tight_layout()
        fig.savefig("%s/%s_scoresculpting.pdf"%(odir,label))
        fig.savefig("%s/%s_scoresculpting.png"%(odir,label))
        