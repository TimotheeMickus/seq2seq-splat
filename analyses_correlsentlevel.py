import sys, os, itertools, random, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtw import dtw,accelerated_dtw
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import scipy.stats as stats

from utils import read_datasets



def main(data, multilingual=False, oh_decomp=False, qualitymetric='comet',nsentsexp=2):
#    metadata, datadict = read_datasets(data, multilingual,oh_decomp=oh_decomp)
    #infer = ['gen','no-gen'] if metadata['generative']=='both' else [metadata['generative']]
    #qscores = pd.read_csv(f'results/{qualitymetric}-scores2.csv').sort_values('checkpoint')
    #qscores = qscores.set_index('checkpoint').sort_index()

    #rng = np.random.default_rng()
    samplesize = 5#1000
    nexperims=2#10
    spearmans = None
    components = ['I','S','T','F','C'] if not(oh_decomp) else ['S','T','C']
    name=''.join(components)+'_L1norm'
    fsuffix=f".oh-schuler" if oh_decomp else f""


    for _ in range(nexperims):
        for mod in ['rus _s0','rus _s1','rus _s2','sla','ine','mul']:
            diffDF=pd.DataFrame(columns=['ckpt', 'ckpt2','layer_idx','func',qualitymetric+'_score',name]+components)

            m1, seed = mod.split() if mod.find('rus')>=0 else (mod,'')
            df = pd.read_csv(f'results/sentence-level/res-sentence-level.{m1}-eng{seed}{fsuffix}.csv').set_index('ckpt')

            for sent in random.sample(list( itertools.combinations(df.hyp.unique(),nsentsexp)  ), samplesize):
                # exhaustive: for every two ckpts
                if nsentsexp==2:
                    ckpts = zip(*[ df[df.hyp==s].index.unique() for s in sent] )
                elif nsentsexp==1:
                    ckpts = itertools.combinations(df[df.hyp==sent[0]].index.unique(),2)
                for ck1,ck2 in ckpts:
                    for func in df.func.unique():
                        if ck1 == ck2:
                            diffs = df[(df.func==func) & ( df.hyp.isin(sent) ) ].sort_values('layer_idx')[['layer_idx','comet_score']+components]
                        else:
                            diffs = df[(df.func==func) & ( df.hyp.isin(sent) ) ].loc[[ck1,ck2]].sort_values('layer_idx')[['layer_idx','comet_score']+components]
                        # won't deal with layer 0
                        diffs.loc[diffs.layer_idx==0,components] = np.nan
                        diffs = diffs.diff().dropna()
                        diffs = diffs[diffs.layer_idx == 0]
                        diffs.layer_idx = [x for x in range(1,7)]
                        diffs['func'] = func
                        diffs = diffs.reset_index()
                        diffs['ckpt2'] = ck1
                        # compute L1norm(sum of abs.values)
                        diffs[name]=diffs[components].abs().sum(axis=1)
                        diffDF = pd.concat([diffDF,diffs])
            import ipdb; ipdb.set_trace()

            diffDF = diffDF.reset_index()
            print(f'MODEL {mod} experim{_}: sentence-level spearman correlations between Δ({qualitymetric}) and {[name]+components} for a sample of {samplesize} pairs of sentences')
            for column in [name]+components:
                aux = diffDF.groupby(['layer_idx','func'])[[qualitymetric+'_score',column]]
                colname = column+' '+''.join(mod)+' '+str(_)
                if spearmans is None:
                    spearmans = aux.corr('spearman').iloc[0::2,-1].sort_index(level=1).droplevel(level=2).reset_index()
                    spearmans = spearmans.rename(columns={name:colname})
                else:
                    spearmans[colname] = aux.corr('spearman').iloc[0::2,-1].sort_index(level=1).droplevel(level=2).reset_index()[column]
            #print(spearmans)

    
    spearmansSTATS = spearmans[['layer_idx','func']]
    for m1, column in itertools.product(datadict.keys(),[name]+components):
        cols = [col for col in spearmans if col.startswith(column+' '+m1)]
        spearmansSTATS[column+' '+m1+ ' mean'] = spearmans[cols].mean(axis=1) #spearmans.apply(lambda row: np.mean([row[col] for col in cols]), axis=1)
        spearmansSTATS[column+' '+m1+ ' stdv'] = spearmans[cols].std(axis=1) #spearmans.apply(lambda row: np.std([row[col] for col in cols]), axis=1)
    #print(spearmansSTATS)

    cols = [col for col in spearmans if re.search(r" s[0-2] ", col)]
    for column in [name]+components:
        cc = [col for col in cols if col.startswith(column)]
        spearmansSTATS[f'{column} rus_allseeds mean'] = spearmans[cc].mean(axis=1) 
        spearmansSTATS[f'{column} rus_allseeds stdv'] = spearmans[cc].std(axis=1) 
    print(spearmansSTATS)
    fsuffix=f"{qualitymetric}_oh-schuler" if oh_decomp else f"{qualitymetric}_mickus-etal"
    spearmansSTATS.to_csv(f'results/sentence-level/Spearmancorrelations-{fsuffix}.csv')


sys.argv = sys.argv if len(sys.argv)>3 else [sys.argv[0],'gen','notmultilingual','mickus']
if __name__ == '__main__':
    data = sys.argv[1]
    multilingual = True if sys.argv[2]=='multilingual' else False
    oh_decomp = True if sys.argv[3].lower().find('oh') > -1 else False

    #main(data, multilingual,oh_decomp)
    #main(data, multilingual,not(oh_decomp))


    main(data, multilingual,not(oh_decomp),qualitymetric='comet')
    main(data, multilingual,oh_decomp,qualitymetric='comet')

    main(data, multilingual,oh_decomp,qualitymetric='bleu')
    main(data, multilingual,oh_decomp,qualitymetric='chrf')
    main(data, multilingual,not(oh_decomp),qualitymetric='bleu')
    main(data, multilingual,not(oh_decomp),qualitymetric='chrf')
