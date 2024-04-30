import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy.random import binomial
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import os

def none(pvals,threshold):
    sorted_pvals = np.array(pvals)
    new_threshold = np.array([threshold]*len(pvals))
    plot_areas = pvals < new_threshold
    n_rejected = np.sum(sorted_pvals < new_threshold)
    return plot_areas, sorted_pvals, new_threshold, n_rejected

def Bonferroni(pvals, threshold):
    pvals = np.array(pvals)
    new_threshold = np.array([threshold/len(pvals)]*len(pvals))
    plot_areas = np.array(pvals) < new_threshold
    sorted_pvals = np.array(pvals)
    n_rejected = np.sum(pvals<new_threshold)
    return plot_areas, sorted_pvals, new_threshold, n_rejected

def BH(pvals, threshold):
    pvals = np.array(pvals)
    indx = np.argsort(pvals)
    sorted_pvals = pvals[indx]
    new_threshold = np.array([m*threshold/len(pvals) for m in range(1,len(pvals)+1)])
    condition = sorted_pvals <= new_threshold
    t = len(pvals)
    not_found = True
    m_ = -1
    while t>-1 and not_found:
            t = t-1
            if condition[t]:
                m_ = t
                not_found = False
    if m_ > -1:
        rejected_indx = indx[:m_+1]
        n_rejected = m_+1
        plot_areas = np.zeros(len(pvals))
        for p in rejected_indx:
            plot_areas[p] = True
    else:
        n_rejected = 0
        plot_areas = np.zeros(len(pvals))
    return plot_areas, sorted_pvals, new_threshold, n_rejected

def BY(pvals, threshold):
    pvals = np.array(pvals)
    indx = np.argsort(pvals)
    sorted_pvals = pvals[indx]
    c = np.log(len(pvals)) + 0.57721 + (1/(2*len(pvals)))
    new_threshold = np.array([m*threshold/(len(pvals)*c) for m in range(1,len(pvals)+1)])
    condition = sorted_pvals <= new_threshold
    t = len(pvals)
    not_found = True
    m_ = -1
    while t>-1 and not_found:
            t = t-1
            if condition[t]:
                m_ = t
                not_found = False
    if m_ > -1:
        rejected_indx = indx[:m_+1]
        n_rejected = m_+1
        plot_areas = np.zeros(len(pvals))
        for p in rejected_indx:
            plot_areas[p] = True
    else:
        n_rejected = 0
        plot_areas = np.zeros(len(pvals))
    return plot_areas, sorted_pvals, new_threshold, n_rejected

def paired_permutation_test(X1,X2,maxiter=10000, plot_histo = False):

    diff = X1-X2
    T0 = np.square(np.median(diff))
    T_stat = np.zeros(maxiter)
    for it in range(maxiter):
        signs_perm = binomial(n=1,p=0.5,size=len(diff))*2 - 1
        diff_perm = diff*signs_perm
        T_stat[it] = np.square(np.median(diff_perm))
    if plot_histo:
        plt.figure()
        sns.histplot(T_stat)
        plt.axvline(T0,color='r')
    pval=np.sum(T_stat>T0)/maxiter  
    return pval

def perm_per_band(source_data,plot_folder,correction,threshold=0.1,maxiter=10000):
    pyreg = pd.read_excel(source_data+'-pyreg.xlsx')
    simpitk = pd.read_excel(source_data+'-simpitk.xlsx')

    pyreg = pyreg.iloc[::2,:]
    simpitk = simpitk.iloc[::2,:]

    pvals = []
    for i in range(pyreg.shape[1]):
        pvals.append(paired_permutation_test(np.array(pyreg.iloc[:,i],float),np.array(simpitk.iloc[:,i],float),maxiter))
    plot_areas, sorted_pvals, new_threshold, n_rejected = correction(pvals,threshold)
    
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(range(len(pvals)),sorted_pvals,'c')
    ax[0].plot(range(len(pvals)),new_threshold,color='orange')
    ax[0].set_xlabel('Band (#) sorted by p-value')
    ax[0].set_ylabel('p-value')
    ax[0].set_xlim((-1,51))
    ax[0].set_ylim(-0.05,1)
    ax[0].legend(['Sorted p-values','BY correction'])
    ax[0].grid()
    ax[1].plot(range(pyreg.shape[1]),np.median(pyreg,axis=0),'r')
    ax[1].plot(range(simpitk.shape[1]),np.median(simpitk,axis=0),'b')
    ax[1].set_ylabel('Mutual Information')
    ax[1].set_xlabel('Band (#)')
    ax[1].grid()
    y_min = 1.03
    y_max = 1.27
    ax[1].set_ylim((y_min,y_max))
    ax[1].set_xlim((-1,51))
    ax[1].fill_between(range(pyreg.shape[1]), y1=y_min, y2=y_max, where=plot_areas, facecolor='c', alpha=0.3)
    ax[1].legend(['pyStackReg','ITK'])
    fig.tight_layout()
    fig.savefig(plot_folder+'/mi.png')
    plt.figure()
    plt.plot(range(pyreg.shape[1]),np.median(pyreg-simpitk,axis=0),color='orange')
    plt.axhline(0)
    plt.ylabel('Mutual Information')
    plt.xlabel('Band (#)')
    plt.grid()
    y_min = -0.02
    y_max = 0.08
    plt.ylim((y_min,y_max))
    plt.xlim((-1,51))
    plt.fill_between(range(pyreg.shape[1]), y1=y_min, y2=y_max, where=plot_areas, facecolor='c', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_folder+'/differences.png')

    return pvals, n_rejected

def paired_permutation_multivariate(source_data,plot_folder,maxiter=10000):

    pyreg = pd.read_excel(source_data+'-pyreg.xlsx')
    simpitk = pd.read_excel(source_data+'-simpitk.xlsx')
    temp1 = np.array(pyreg.iloc[::2,:])
    temp2 = np.array(simpitk.iloc[::2,:])

    diff = temp1-temp2
    mean = np.mean(diff,axis=0)
    T0 = np.dot(mean,mean)
    T_stat = np.zeros(maxiter)
    for it in range(maxiter):
        signs_perm = binomial(n=1,p=0.5,size=len(diff))*2 - 1
        diff_perm = np.zeros(diff.shape)
        for i in range(len(signs_perm)):
            diff_perm[i,:] = diff[i,:]*signs_perm[i]
        mean_perm = np.mean(diff_perm,axis=0)
        T_stat[it] = np.dot(mean_perm,mean_perm)
    plt.figure()
    sns.histplot(T_stat)
    plt.axvline(T0,color='r')
    pval=np.sum(T_stat>T0)/maxiter
    plt.title('p-value: '+str(round(pval,3)))
    plt.savefig(plot_folder+'/global_test.png')
    return pval

@hydra.main(version_base=None, config_path="/workspace/conf", config_name="config")
def permutation_tests(cfg: DictConfig):

    date = str(cfg.db.date)
    target_folder = '/home/user/data/interim/registered/permutation_tests/'+date
    source_folder = '/home/user/data/interim/registered/mutual_information'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    source_data = source_folder+'/'+date

    paired_permutation_multivariate(source_data,target_folder)
    perm_per_band(source_data,target_folder,BY)




if __name__ == '__main__':
    permutation_tests()