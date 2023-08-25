import sys, os, itertools, random, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from utils import read_datasets



def main(data, multilingual=False, oh_decomp=False, qualitymetric='bleu'):
    fsuffix=f"{qualitymetric}_oh-schuler" if oh_decomp else f"{qualitymetric}_mickus-etal"
    print(f'CORRELATIONS FOR {fsuffix}, multilingual={multilingual}')
    metadata, datadict = read_datasets(data, multilingual,oh_decomp=oh_decomp)
    components = ['I','S','T','F','C'] if not(oh_decomp) else ['S','T','C']
    name=''.join(components)+'_L1norm'
    #infer = ['gen','no-gen'] if metadata['generative']=='both' else [metadata['generative']]
    qscores = pd.read_csv(f'results/{qualitymetric}-scores2.csv').sort_values('checkpoint')
    qscores = qscores.set_index('checkpoint').sort_index()

    rng = np.random.default_rng()
    samplesize = 1000
    nexperims=10
    spearmans = None
    for _ in range(nexperims):
        for m1,pd1 in datadict.items():
            diffDF=pd.DataFrame(columns=['checkpoint', 'checkpoint2','layer_idx','func',qualitymetric,name]+components)
            m2 = f'{m1}-eng' if len(m1)>2 else f'rus-eng_{m1}'
            for infer in qscores.infer_style.unique():
                qscoresdiff,funcdiff = [],[]
                qscorespart = qscores[qscores.model==m2]
                ckpts = rng.integers(1,qscorespart.index.max()//1000, size=2)*1000
                #for layer,func,comp in itertools.product(range(1,7),pd1[infer].func.unique(),['I','S','T','F','C']):
                for func in pd1[infer].func.unique():
                    #ts  =pd1[infer][(pd1[infer].func==func)& (pd1[infer].layer_idx==layer) ].set_index('checkpoint').drop('func',axis=1) #[comp]
                    #ts  =pd1[infer][(pd1[infer].layer_idx==layer) ].set_index('checkpoint')#[comp]
                    ts  =pd1[infer][(pd1[infer].func==func) ].set_index('checkpoint').drop('func',axis=1) #[comp]
                    maxckpt = np.min((qscorespart.index.max(),ts.index.max()))//1000
                    for ck1,ck2 in random.sample(list(itertools.combinations(range(1,1+maxckpt),2)),samplesize):
                        ckpts = [ck1*1000,ck2*1000] 
                        coso = qscorespart.score.loc[ckpts].diff().dropna()
                        diffs = ts.loc[ckpts].sort_values(by='layer_idx').diff().dropna()
                        diffs = diffs[diffs.layer_idx == 0]
                        diffs.layer_idx = [1,2,3,4,5,6]
                        diffs[qualitymetric] = coso
                        diffs['func'] = func
                        diffs = diffs.reset_index()
                        diffs['checkpoint2'] = ckpts[0]
                        # compute L1(sum of abs.values)
                        diffs[name]=diffs[components].abs().sum(axis=1)
                        diffDF = pd.concat([diffDF,diffs])
                        #qscoresdiff.append(coso.loc[ckpts[1]])
                        #funcdiff.append(diffs.ISFTCmeanvec.values)
            diffDF = diffDF.reset_index()
            print(f'MODEL {m1} experim{_}: Spearman correlations between Δ({qualitymetric}) and {[name]+components} for a sample of {samplesize} pairs of ckpts')
            for column in [name]+components:
                aux = diffDF.groupby(['layer_idx','func'])[[qualitymetric,column]]
                colname = column+' '+m1+' '+str(_)
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
    spearmansSTATS.to_csv(f'results/Spearmancorrelations-{fsuffix}.csv')


sys.argv = sys.argv if len(sys.argv)>3 else [sys.argv[0],'gen','notmultilingual','mickus']
if __name__ == '__main__':
    data = sys.argv[1]
    multilingual = True if sys.argv[2]=='multilingual' else False
    oh_decomp = True if sys.argv[3].lower().find('oh') > -1 else False

    results = Parallel(n_jobs=6)(delayed(main)(*args, **kwargs)
                                    for *args, kwargs in (
                                        [ data, multilingual,oh_decomp, {'qualitymetric':'bleu'} ],
                                        [ data, multilingual,oh_decomp, {'qualitymetric':'comet'} ],
                                        [ data, multilingual,oh_decomp, {'qualitymetric':'chrf'} ],
                                        [ data, multilingual,not(oh_decomp), {'qualitymetric':'bleu'} ],
                                        [ data, multilingual,not(oh_decomp), {'qualitymetric':'comet'} ],
                                        [ data, multilingual,not(oh_decomp), {'qualitymetric':'chrf'} ],
                                    )
                                )

 
    """ LEGACY: not parallelized
    main(data, multilingual,oh_decomp,qualitymetric='bleu')
    main(data, multilingual,oh_decomp,qualitymetric='comet')
    main(data, multilingual,oh_decomp,qualitymetric='chrf')
    main(data, multilingual,not(oh_decomp),qualitymetric='bleu')
    main(data, multilingual,not(oh_decomp),qualitymetric='comet')
    main(data, multilingual,not(oh_decomp),qualitymetric='chrf')
    """