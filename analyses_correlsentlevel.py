import sys, os, itertools, random, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from utils import read_datasets



def main(data, multilingual=False, oh_decomp=False, qualitymetric='comet',nsentsexp=2):
#    metadata, datadict = read_datasets(data, multilingual,oh_decomp=oh_decomp)
    #infer = ['gen','no-gen'] if metadata['generative']=='both' else [metadata['generative']]
    #qscores = pd.read_csv(f'results/{qualitymetric}-scores2.csv').sort_values('checkpoint')
    #qscores = qscores.set_index('checkpoint').sort_index()

    #rng = np.random.default_rng()
    samplesize = 1000 if nsentsexp >= 2 else 350
    nexperims=10
    spearmans = None
    components = ['I','S','T','F','C'] if not(oh_decomp) else ['S','T','C']
    name=''.join(components)+'_L1norm'
    fsuffix=f".oh-schuler" if oh_decomp else f""
    modelnames = ['rus _s0','rus _s1','rus _s2','sla','ine','mul']

    for _ in range(nexperims):
        for mod in modelnames:
            diffDF=pd.DataFrame(columns=['ckpt', 'ckpt2','layer_idx','func',qualitymetric+'_score',name]+components)

            m1, seed = mod.split() if mod.find('rus')>=0 else (mod,'')
            df = pd.read_csv(f'results/sentence-level/res-sentence-level.{m1}-eng{seed}{fsuffix}.csv').set_index('ckpt')
            actualN = 0
            for sent in random.sample(list( itertools.combinations(df.src.unique(),nsentsexp)  ), samplesize):
                # exhaustive: for every two ckpts
                if nsentsexp==2:
                    ckpts = zip(*[ df[df.src==s].index.unique() for s in sent] )
                elif nsentsexp==1:
                    ckpts = itertools.combinations(df[df.src==sent[0]].index.unique(),2)
                for ck1,ck2 in ckpts:
                    actualN += 1
                    for func in df.func.unique():
                        if ck1 == ck2:
                            diffs = df[(df.func==func) & ( df.src.isin(sent) ) ].loc[[ck1]].sort_values('layer_idx')[['layer_idx','comet_score']+components]
                        else:
                            s1c1 = df[(df.func==func) & ( df.src == sent[0] ) ].loc[[ck1]]
                            s2c2 = df[(df.func==func) & ( df.src == sent[-1])  ].loc[[ck2]]
                            diffs = pd.concat([s1c1,s2c2]).sort_values('layer_idx')[['layer_idx','comet_score']+components]
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

            diffDF = diffDF.reset_index()

            print(f'MODEL {mod} experim{_}: sentence-level spearman correlations between Δ({qualitymetric}) and {[name]+components} for a sample of {samplesize} pairs of sentences')
            print(f'actual N: {actualN}')
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
    for m1, column in itertools.product(modelnames,[name]+components):
        cols = [col for col in spearmans if col.startswith(column+' '+m1)]
        spearmansSTATS[column+' '+m1+ ' mean'] = spearmans[cols].mean(axis=1) #spearmans.apply(lambda row: np.mean([row[col] for col in cols]), axis=1)
        spearmansSTATS[column+' '+m1+ ' stdv'] = spearmans[cols].std(axis=1) #spearmans.apply(lambda row: np.std([row[col] for col in cols]), axis=1)
    #print(spearmansSTATS)

    cols = [col for col in spearmans if re.search(r" _s[0-2] ", col)]
    for column in [name]+components:
        cc = [col for col in cols if col.startswith(column)]
        spearmansSTATS[f'{column} rus_allseeds mean'] = spearmans[cc].mean(axis=1) 
        spearmansSTATS[f'{column} rus_allseeds stdv'] = spearmans[cc].std(axis=1) 
    print(spearmansSTATS)
    fsuffix=f"{qualitymetric}_oh-schuler" if oh_decomp else f"{qualitymetric}_mickus-etal"
    spearmansSTATS.to_csv(f'results/sentence-level/Spearmancorrelations-{fsuffix}-exp{str(nsentsexp)}.csv')


#sys.argv = sys.argv if len(sys.argv)>3 else [sys.argv[0],'gen','notmultilingual','mickus'] #< not needed, we explore all options
if __name__ == '__main__':
    data = 'gen' # sys.argv[1]
    multilingual = True # if sys.argv[2]=='multilingual' else False
    oh_decomp = True # if sys.argv[3].lower().find('oh') > -1 else False

    #main(data, multilingual,not(oh_decomp),qualitymetric='comet',nsentsexp=2)

    results = Parallel(n_jobs=4)(delayed(main)(*args, **kwargs)
                                    for *args, kwargs in (
                                        [ data, multilingual,oh_decomp, {'qualitymetric':'comet', 'nsentsexp':2} ],
                                        [ data, multilingual,not(oh_decomp), {'qualitymetric':'comet', 'nsentsexp':2} ],
                                        [ data, multilingual,oh_decomp, {'qualitymetric':'comet', 'nsentsexp':1} ],
                                        [ data, multilingual,not(oh_decomp), {'qualitymetric':'comet', 'nsentsexp':1} ],
                                    )
                                )

