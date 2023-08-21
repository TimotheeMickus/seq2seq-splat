import sys, os, itertools, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtw import dtw,accelerated_dtw
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import scipy.stats as stats

from utils import read_datasets



def main(data,multilingual=False):
    generative, datadict = read_datasets(data, multilingual)
    generativeOH, datadictOH = read_datasets(data, multilingual,oh_decomp=True)
    #infer = ['gen','no-gen'] if generative=='both' else [generative]
    bleu = pd.read_csv(f'results/bleu-scores2.csv').sort_values('checkpoint')
    bleu = bleu.set_index('checkpoint').sort_index()

    rng = np.random.default_rng()
    samplesize = 1000

    spearmans = None
    for _ in range(10):
        for m1,pd1 in datadict.items():
            diffDF=pd.DataFrame(columns=['checkpoint', 'checkpoint2','layer_idx','func','I','S','T','F','C','bleu','ISFTCmeanvec'])
            m2 = f'{m1}-eng' if len(m1)>2 else f'rus-eng_{m1}'
            for infer in bleu.infer_style.unique():
                bleudiff,funcdiff = [],[]
                bleupart = bleu[bleu.model==m2]
                ckpts = rng.integers(1,bleupart.index.max()//1000, size=2)*1000
                #for layer,func,comp in itertools.product(range(1,7),pd1[infer].func.unique(),['I','S','T','F','C']):
                for func in pd1[infer].func.unique():
                    #ts  =pd1[infer][(pd1[infer].func==func)& (pd1[infer].layer_idx==layer) ].set_index('checkpoint').drop('func',axis=1) #[comp]
                    #ts  =pd1[infer][(pd1[infer].layer_idx==layer) ].set_index('checkpoint')#[comp]
                    ts  =pd1[infer][(pd1[infer].func==func) ].set_index('checkpoint').drop('func',axis=1) #[comp]
                    maxckpt = np.min((bleupart.index.max(),ts.index.max()))//1000
                    for ck1,ck2 in random.sample(list(itertools.combinations(range(1,1+maxckpt),2)),samplesize):
                        ckpts = [ck1*1000,ck2*1000] 
                        coso = bleupart.bleu.loc[ckpts].diff().dropna().abs()
                        diffs = ts.loc[ckpts].sort_values(by='layer_idx').diff().dropna().abs()
                        diffs = diffs[diffs.layer_idx == 0]
                        diffs.layer_idx = [1,2,3,4,5,6]
                        diffs['bleu'] = coso
                        diffs['func'] = func
                        diffs = diffs.reset_index()
                        diffs['checkpoint2'] = ckpts[0]
                        # TODO: instead of the mean, use L1(sum of abs.values)
                        # TODO: compute them also individually
                        diffs['ISFTCmeanvec']=diffs[['I','S','F','T','C']].abs().sum(axis=1)
                        diffDF = pd.concat([diffDF,diffs])
                        #bleudiff.append(coso.loc[ckpts[1]])
                        #funcdiff.append(diffs.ISFTCmeanvec.values)
            diffDF = diffDF.reset_index()
            for column in ['ISFTCmeanvec','I','S','F','T','C']:
                aux = diffDF.groupby(['layer_idx','func'])[['bleu',column]]
                colname = column+' '+m1+' '+str(_)
                print(f'MODEL {m1}: Spearman correlations between Δ(BLEU) and {column} for a sample of {samplesize} pairs of ckpts')
                if spearmans is None:
                    spearmans = aux.corr('spearman').iloc[0::2,-1].sort_index(level=1).droplevel(level=2).reset_index()
                    spearmans = spearmans.rename(columns={'ISFTCmeanvec':colname})
                else:
                    spearmans[colname] = aux.corr('spearman').iloc[0::2,-1].sort_index(level=1).droplevel(level=2).reset_index()[column]
            #print(spearmans)
    

    spearmansSTATS = spearmans[['layer_idx','func']]
    for m1, column in zip(datadict.keys(),['ISFTCmeanvec','I','S','F','T','C']):
        cols = [col for col in spearmans if col.startswith(column+' '+m1)]
        spearmansSTATS[column+' '+m1+ ' mean'] = spearmans[cols].mean(axis=1) #spearmans.apply(lambda row: np.mean([row[col] for col in cols]), axis=1)
        spearmansSTATS[column+' '+m1+ ' stdv'] = spearmans[cols].std(axis=1) #spearmans.apply(lambda row: np.std([row[col] for col in cols]), axis=1)
    #print(spearmansSTATS)

    cols = [col for col in spearmans if re.search(r" s[0-2] ", col)]
    for column in ['ISFTCmeanvec','I','S','F','T','C']:
        spearmansSTATS[f'{column} rus_allseeds mean'] = spearmans[cols].mean(axis=1) #spearmans.apply(lambda row: np.mean([row[col] for col in cols]), axis=1)
        spearmansSTATS[f'{column} rus_allseeds stdv'] = spearmans[cols].std(axis=1) #spearmans.apply(lambda row: np.std([row[col] for col in cols]), axis=1)
    print(spearmansSTATS)
    spearmansSTATS.to_csv('Spearmancorrelations-mickus-etal.csv')
    import ipdb; ipdb.set_trace()


sys.argv = sys.argv if len(sys.argv)>=2 else [sys.argv[0],'gen','']
if __name__ == '__main__':
    data = sys.argv[1]
    multilingual = True if sys.argv[2]=='multilingual' else False
    main(data, multilingual)
