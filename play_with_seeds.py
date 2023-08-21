import sys, os, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtw import dtw,accelerated_dtw
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import scipy.stats as stats

def read_datasets():
    """ Returns a dictionary with the read data structures. """
    datadict={x:{'gen':[],'non-gen':[]} for x in ['s0','s1','s2','sla','ine','mul']}
    if sys.argv[1]=='gen':
        generative=True
        # compare the seeds of the gen decompositions
        s0=pd.read_csv('results/decomps/gen/res-gen-s0.csv')
        s1=pd.read_csv('results/decomps/gen/res-gen-s1.csv')
        s2=pd.read_csv('results/decomps/gen/res-gen-s2.csv')
        s0means = s0.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        s1means = s1.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        s2means = s2.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        if sys.argv[2]=='multilingual':
            sla=pd.read_csv('results/decomps/gen/res-gen.sla-eng.csv')
            ine=pd.read_csv('results/decomps/gen/res-gen.ine-eng.csv')
            mul=pd.read_csv('results/decomps/gen/res-gen.mul-eng.csv')
            slameans = sla.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
            inemeans = ine.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
            mulmeans = mul.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
    elif ((sys.argv[1]=='no-gen') or (sys.argv[1]=='no gen') or (sys.argv[1]=='nogen')):
        generative=False
        # gompare the seeds of no-gen decompositions
        s0=pd.read_csv('results/decomps/no-gen/res-no-gen-s0.csv')
        #s0 = s0[s0.checkpoint <= 585000] # drop ckpts that are not form this model
        s1=pd.read_csv('results/decomps/no-gen/res-no-gen-s1.csv')
        s2=pd.read_csv('results/decomps/no-gen/res-no-gen-s2.csv')
    elif ((sys.argv[1]=='both') or (sys.argv[1]=='gen-nogen')):
        generative='both'
        # compare gen vs no-gen series
        s0=pd.read_csv('results/decomps/gen/res-gen-s0.csv')
        s1=pd.read_csv('results/decomps/gen/res-gen-s1.csv')
        s2=pd.read_csv('results/decomps/gen/res-gen-s2.csv')
        ng0=pd.read_csv('results/decomps/no-gen/res-no-gen-s0.csv')
        ng1=pd.read_csv('results/decomps/no-gen/res-no-gen-s1.csv')
        ng2=pd.read_csv('results/decomps/no-gen/res-no-gen-s2.csv')
        s0means = ng0.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        s1means = ng1.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        s2means = ng2.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        ngs0means = ng0.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        ngs1means = ng1.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        ngs2means = ng2.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        if sys.argv[2]=='multilingual':
            sla = pd.read_csv('results/decomps/gen/res-gen.sla-eng.csv')
            ine = pd.read_csv('results/decomps/gen/res-gen.ine-eng.csv')
            mul = pd.read_csv('results/decomps/gen/res-gen.mul-eng.csv')
            ngsla=pd.read_csv('results/decomps/no-gen/res-no-gen.sla-eng.csv')
            ngine=pd.read_csv('results/decomps/no-gen/res-no-gen.ine-eng.csv')
            ngmul=pd.read_csv('results/decomps/no-gen/res-no-gen.mul-eng.csv')
            slameans = sla.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
            inemeans = ine.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
            mulmeans = mul.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
            ngslameans = ngsla.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
            nginemeans = ngine.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
            ngmulmeans = ngmul.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})

    normalize=True
    ZN=""
    if normalize:
        ZN=" normalized"
        s0means = Znormalize(s0means)
        s1means = Znormalize(s1means)
        s2means = Znormalize(s2means)
        if generative =='both':
            ngs0means = Znormalize(ngs0means)
            ngs1means = Znormalize(ngs1means)
            ngs2means = Znormalize(ngs2means)
        if multilingual:
            slameans = Znormalize(slameans)
            inemeans = Znormalize(inemeans)
            mulmeans = Znormalize(mulmeans)
            if generative =='both':
                ngslameans = Znormalize(ngslameans)
                nginemeans = Znormalize(nginemeans)
                ngmulmeans = Znormalize(ngmulmeans)

    for name in ['s0','s1','s2','sla','ine','mul']:
        datadict[name]['gen'] = eval(name+'means')
        if generative=='both':
            datadict[name]['non-gen'] = eval('ng'+name+'means')

    return (generative,normalize),datadict

def plot_series_by_ind_and_layer(s, seed, modname:str()='rus-eng', functions:list()=None):
    '''
    Plot a time series of each component at a layer level
    '''
    components= ['I', 'S', 'T', 'F', 'C']
    func = functions if functions else s.func.unique()
    fig, ax = plt.subplots(ncols=len(components), nrows=len(func), sharex=True, num=f'TS by layer - {modname} {seed}')
    fig.suptitle(f'MODEL: {modname}   {seed}')
    for j, comp in enumerate(components):
        for i,f in enumerate(func): 
            currax= ax[i][j]
            df =  pd.DataFrame(s[s['func']==f].filter(['layer_idx',f'{comp}','checkpoint'])).reset_index(drop=True)
            df = df.pivot(index='checkpoint', columns='layer_idx',values=f'{comp}')
            df.plot(ax=currax)
            currax.get_legend().remove()
            if j==0:
                currax.set_ylabel(f'{f}')
            if i==0:
                currax.set_title(f'{comp}')

    currax.legend(bbox_to_anchor=(1.64, 4),title='layer')

def scatter_2seeds(s1, seed1:str(), s2, seed2:str(), f:str(), overlapfig:tuple()=(None,None)):
    '''
    make scatter plots matrix comparing two seeds
    IN:
        s1: dataframe sImeans like
        seed1: string indicating the seed of s1
        s2: another dataframe
        seed2: string indicating the seed of s2
        f: either 'cosine', 'norm_ratio', 'l2' or 'spim'
    '''
    components= ['I', 'S', 'T', 'F', 'C']
    nlayers=7
    if overlapfig[0]:
        fig, ax = overlapfig
        fig.suptitle(f'Scatter plots of {f} for rus-eng models with all three seeds')

    else:
        fig, ax = plt.subplots(nrows=len(components), ncols=(nlayers-1), num=f'scatter{auxIDX} metric:{f}')
        fig.suptitle(f'Scatter of {f} for rus-eng models with seeds: {seed1} & {seed2}')

    for i,comp in enumerate(components):
        df =  pd.DataFrame(s1[s1['func']==f].filter(['layer_idx',f'{comp}','checkpoint'])).reset_index(drop=True)
        df = df.pivot(index='checkpoint', columns='layer_idx',values=f'{comp}')

        df2 = pd.DataFrame(s2[s2['func']==f].filter(['layer_idx',f'{comp}','checkpoint'])).reset_index(drop=True)
        df2 = df2.pivot(index='checkpoint', columns='layer_idx',values=f'{comp}') 

        for j in range(1,nlayers):
            n = min(len(df[j]),len( df2[j]))

            currax= ax[i][j-1]
            #dots = currax.scatter(df[j][:n],df2[j][:n], alpha=0.5, label=f'seed {seed1} vs seed {seed2}')
            dots = currax.plot(df[j][:n].to_numpy(),df2[j][:n].to_numpy(), alpha=0.5, label=f'seed {seed1} vs seed {seed2}')
            #currax.get_legend().remove()

            if j==0:
                currax.set_ylabel(f'{comp}', weight='bold')
            if i==0:
                currax.set_title(f'Layer {j}')
        
    currax.legend(bbox_to_anchor=(2, 3.5))
    
    return fig,ax

def plot_dtw_allseeds(models:dict(), distances:pd.DataFrame(), metric:str(), comp:str(), multilingual=False):
    nlayers=7
    #fig, ax = plt.subplots(nrows=len(models), ncols=nlayers, num=f'DTW{auxIDX} metric{metric} component{comp}')
    #fig.suptitle(f'DTW Minimum Path comparison across seeds \n metric:{metric} and component:{comp}')

    comparisons = ['m1-m2',   'm2-m3',   'm3-m1'] if not multilingual else [
                   'm1-m2',   'm2-m3',   'm3-m1',  'ine-m1', 'ine-m2','ine-m3', 
                   'sla-m1',  'sla-m2',  'sla-m3', 'mul-m1', 'mul-m2','mul-m3',
                   'ine-sla', 'ine-mul', 'sla-mul',
                   ]
    for i, m in enumerate(comparisons):
        m1,m2 = m.split('-')
        mod1, mname1 = models[m1]['df'], models[m1]['modname']
        mod2, mname2 = models[m2]['df'], models[m2]['modname']
        df =  pd.DataFrame(mod1[mod1['func']==metric].filter(['layer_idx',f'{comp}','checkpoint'])).reset_index(drop=True)
        df = df.pivot(index='checkpoint', columns='layer_idx',values=f'{comp}')
        df2 = pd.DataFrame(mod2[mod2['func']==metric].filter(['layer_idx',f'{comp}','checkpoint'])).reset_index(drop=True)
        df2 = df2.pivot(index='checkpoint', columns='layer_idx',values=f'{comp}') 
        dlist=[mname1,mname2,metric,comp]
        for j in range(nlayers):
            d1 = df[j].values
            d2 = df2[j].values
            d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(d1,d2, dist='euclidean')
            dlist.append(d)
            if False:
                currax = ax[i][j]
                currax.imshow(acc_cost_matrix.T, origin='lower', interpolation='nearest')
                currax.plot(path[0], path[1], 'w')
                if j == 0:
                    currax.set_xlabel(f'{mname1}', weight='bold')
                    currax.set_ylabel(f'{mname2}', weight='bold')
                if i == 0:
                    currax.set_title(f'layer {j}')
                    currax.text(50, 500, f'distance: {np.round(d,3)}', 
                     bbox=dict(fill=True, color='gray'))
        distances = pd.concat([distances, pd.DataFrame([dlist], columns=distances.columns)])
    return distances

def Znormalize(df:pd.DataFrame()):
    # z-normalize a given dataframe
    normalized = pd.DataFrame(columns=df.columns)
    for metric in df.func.unique():
        for layer in df.layer_idx.unique():
            coso = df[((df.func==metric)&(df.layer_idx==layer))]
            cosonorm = {'layer_idx': [layer]*len(coso), 
                      'func':[metric]*len(coso), 
                      'checkpoint':coso.checkpoint }
            for comp in ['I', 'S', 'T', 'F', 'C']:
                cosonorm[comp]=(coso[comp].mean() - coso[comp])/coso[comp].std()
            
            normalized = pd.concat([normalized, pd.DataFrame(cosonorm)], ignore_index=True)
    return normalized

def plotDTWheatmaps(df):
    fig, ax = plt.subplots(nrows=len(df.metric.unique()), ncols=3)
    for i,func in enumerate(df.metric.unique()):
        for j,seed in enumerate(['s1', 's2', 's3']):
            currax = ax[i][j]
            coso = df[((df.metric==func)  & ((df.model2==f'rus-eng {seed}') | (df.model1==f'rus-eng {seed}') ))]# & (df.component!='C')]
            currax.imshow(coso.iloc[:,-6:].T)
            #xlabels =  [f"{i} \t{j.split(' ')[-1].split('-')[0]}-{k.split(' ')[-1].split('-')[0]}".expandtabs() for i,j,k in zip(coso.component,coso.model1,coso.model2)]
            #xlabels = [f"{k} {l.split(' ')[-1].split('-')[0]}-{m.split(' ')[-1].split('-')[0]}" for k,l,m in zip(coso.component,coso.model1,coso.model2)]
            xlabels = [f"{k} {l.split(' ')[-1].split('-')[0]}" for k,l in zip(coso.component,coso.model1)]
            for idx,x in enumerate(xlabels):
                if x.find(seed) > 0 :
                    xlabels[idx] = x.split(' ')[0]+' s'+str((int(seed.lstrip('s'))+1)%4)
            currax.set_xticks([i for i in range(len(coso))])
            currax.set_xticklabels(['']*len(coso))
            #currax.set_yticklabels(["","2","","4","","6"])
            if j==0:
                currax.set_ylabel(func)
            if i==0:
                currax.set_title(f'DTW dist. to rus-eng seed {seed}')
            if i==(len(df.metric.unique())-1):
                #currax.set_yticks([1,2,3,4,5,6])
                #currax.set_yticklabels(["","2","","4","","6"])
                currax.set_xlabel('COMPONENT  distance-to-model')
                currax.set_xticklabels(xlabels)
                currax.xaxis.set_tick_params(rotation=90)

def do_tests(g1,g2,func,comp,seed=""):
    uH0reject, uH0notreject=[],[]
    permH0reject, permH0notreject=[],[]
    for layer in range(1,7):
        pop1, pop2 = g1[f'layer{layer}'], g2[f'layer{layer}']
        pop = pd.concat((pop1,pop2))
        appendable = (func,comp,layer,seed) if seed else (func,comp,layer)
        # U-test
        ustat = stats.mannwhitneyu(pop1, pop2, alternative='two-sided')
        meandiff = pop1.mean() - pop2.mean()
        if ustat.pvalue < 0.05:
            uH0reject.append(appendable)
        else:
            uH0notreject.append(appendable)
        # permutation test:
        mudiffs = []
        for part1 in itertools.combinations(pop.index, len(pop1)): 
            part2 = [x for x in pop.index if x not in part1]
            mudiffs.append(pop.loc[np.array(part1)].mean() - pop.loc[np.array(part2)].mean())
        ge = [1 for x in mudiffs if x > meandiff]
        pval = sum(ge)/len(mudiffs)
        if pval < 0.05:
            permH0reject.append(appendable)
        else:
            permH0notreject.append(appendable)
    return uH0reject, uH0notreject, permH0reject, permH0notreject

def permutation_and_U_tests(df):
    # PERFORM MannWhitney U-test to see if distance distributions are equivalent:
    uH0reject, uH0notreject=[],[]
    permH0reject, permH0notreject=[],[]
    for func in df.metric.unique():
        for comp in ['I', 'S', 'T', 'F', 'C']:
            # CASE1: fix k, then g1={DTWdist(sk,sj) \forall seeds j } and g2={DTWdist(multilingual,sj) \forall seeds j}
            for seed in ['s1', 's2', 's3']:
                # there are 360cases here: 6layer*3seeds*5components*4metrics
                coso = df[((df.metric==func)  & ((df.model2==f'rus-eng {seed}') | (df.model1==f'rus-eng {seed}') )) & (df.component==comp)]
                ing1=coso.model1.str.contains(" s")
                g1 = coso[ing1]
                g2 = coso[ing1==0]
                tests = do_tests(g1,g2,func,comp,seed)
                uH0reject = uH0reject + tests[0]
                uH0notreject = uH0notreject + tests[1]
                permH0reject= permH0reject + tests[2]
                permH0notreject = permH0notreject + tests[3]

    #H0reject is EMPTY in this case -> there is no sufficient evidence to reject the H0: "the two samples come from the same distribution"
    print(f"CASE 1 -   Utest: the null hypothesis is rejected on {len(uH0reject)} our of the {len(uH0reject)+len(uH0notreject)} times. They are:")
    print(f"   - Permutation: the null hypothesis is rejected on {len(permH0reject)} our of the {len(permH0reject)+len(permH0notreject)} times. They are:")
    for rej in permH0reject:
        print(rej)
    uH0reject, uH0notreject=[],[]
    permH0reject, permH0notreject=[],[]
    for func in df.metric.unique():
        for comp in ['I', 'S', 'T', 'F', 'C']:
            # CASE2: g1={DTWdist(si,sj) \forall seeds i,j } and g2={DTWdist(multilingual,si) \forall seed i}U{DTWdist(mult1,mult2) for multN in {mul,sla,ine}}
            g1 = df[(df.model1.str.contains(" s")) & (df.metric==func) & (df.component==comp)]
            g2 = df[(df.model1.str.contains('sla|ine|mul')) & (df.metric==func) & (df.component==comp)]
            tests = do_tests(g1,g2,func,comp)
            uH0reject = uH0reject + tests[0]
            uH0notreject = uH0notreject + tests[1]
            permH0reject= permH0reject + tests[2]
            permH0notreject = permH0notreject + tests[3]

    #H0reject is EMPTY in this case -> there is no sufficient evidence to reject the H0: "the two samples come from the same distribution"
    print(f"CASE 2 -   Utest: the null hypothesis is rejected on {len(uH0reject)} our of the {len(uH0reject)+len(uH0notreject)} times. They are:")
    print(f"   - Permutation: the null hypothesis is rejected on {len(permH0reject)} our of the {len(permH0reject)+len(permH0notreject)} times. They are:")
    for rej in permH0reject:
        print(rej)

    uH0reject, uH0notreject=[],[]
    permH0reject, permH0notreject=[],[]
    for func in df.metric.unique():
        for comp in ['I', 'S', 'T', 'F', 'C']:
            # CASE3: g1={DTWdist(si,sj) \forall seeds i,j } and g2={DTWdist(multilingual,si) \forall seed i}
            g1 = df[(df.model1.str.contains(" s")) & (df.metric==func) & (df.component==comp)]
            g2 = df[(df.model1.str.contains('sla|ine|mul')) & (df.metric==func) & (df.component==comp)]
            g2 = g2[g2.model2.str.contains(" s")]
            tests = do_tests(g1,g2,func,comp)
            uH0reject = uH0reject + tests[0]
            uH0notreject = uH0notreject + tests[1]
            permH0reject= permH0reject + tests[2]
            permH0notreject = permH0notreject + tests[3]

    #H0reject is EMPTY in this case -> there is no sufficient evidence to reject the H0: "the two samples come from the same distribution"
    print(f"CASE 3 -   Utest: the null hypothesis is rejected on {len(uH0reject)} our of the {len(uH0reject)+len(uH0notreject)} times. They are:")
    print(f"   - Permutation: the null hypothesis is rejected on {len(permH0reject)} our of the {len(permH0reject)+len(permH0notreject)} times. They are:")
    for rej in permH0reject:
        print(rej)

    uH0reject, uH0notreject=[],[]
    permH0reject, permH0notreject=[],[]
    for func in df.metric.unique():
        for comp in ['I', 'S', 'T', 'F', 'C']:
            # CASE4: g1={DTWdist(si,sj) \forall seeds i,j } and g2={DTWdist(mult1,mult2) for multN in {mul,sla,ine}}
            g1 = df[(df.model1.str.contains(" s")) & (df.metric==func) & (df.component==comp)]
            g2 = df[(df.model1.str.contains('sla|ine|mul')) & (df.metric==func) & (df.component==comp)]
            g2 = g2[g2.model2.str.contains('sla|ine|mul')]
            tests = do_tests(g1,g2,func,comp)
            uH0reject = uH0reject + tests[0]
            uH0notreject = uH0notreject + tests[1]
            permH0reject= permH0reject + tests[2]
            permH0notreject = permH0notreject + tests[3]

    #H0reject is EMPTY in this case -> there is no sufficient evidence to reject the H0: "the two samples come from the same distribution"
    print(f"CASE 4 -   Utest: the null hypothesis is rejected on {len(uH0reject)} our of the {len(uH0reject)+len(uH0notreject)} times. They are:")
    print(f"   - Permutation: the null hypothesis is rejected on {len(permH0reject)} our of the {len(permH0reject)+len(permH0notreject)} times. They are:")
    for rej in permH0reject:
        print(rej)

    uH0reject, uH0notreject=[],[]
    permH0reject, permH0notreject=[],[]
    for func in df.metric.unique():
        for comp in ['I', 'S', 'T', 'F', 'C']:
            # CASE5: g1={DTWdist(mi,mj) \forall mh \in SET={si \forall seed i}U{sla}} and g2={DTWdist(mult1,mult2) for multN in {mul,sla,ine}}
            g1 = df[(df.model1.str.contains(" s|sla")) & (df.metric==func) & (df.component==comp)]
            g1 = g1[g1.model2.str.contains(' s')]
            g2 = df[(df.model1.str.contains('sla|ine|mul')) & (df.metric==func) & (df.component==comp)]
            g2 = g2[g2.model2.str.contains('sla|ine|mul')]
            tests = do_tests(g1,g2,func,comp)
            uH0reject = uH0reject + tests[0]
            uH0notreject = uH0notreject + tests[1]
            permH0reject= permH0reject + tests[2]
            permH0notreject = permH0notreject + tests[3]

    #H0reject is EMPTY in this case -> there is no sufficient evidence to reject the H0: "the two samples come from the same distribution"
    print(f"CASE 5 -   Utest: the null hypothesis is rejected on {len(uH0reject)} our of the {len(uH0reject)+len(uH0notreject)} times. They are:")
    print(f"   - Permutation: the null hypothesis is rejected on {len(permH0reject)} our of the {len(permH0reject)+len(permH0notreject)} times. They are:")
    for rej in permH0reject:
        print(rej)



sys.argv = sys.argv if len(sys.argv)>=2 else [sys.argv[0],'gen','']
def main(multilingual=False):
    (generative,normalize), datadict = read_datasets()


    global auxIDX # aux in naming the Figure windows
    auxIDX = 1
    if not (generative=='both'):
        if False:
            plot_series_by_ind_and_layer(datadict['s0']['gen'],seed=str(1111)+ZN)
            auxIDX += 1
            plot_series_by_ind_and_layer(datadict['s1']['gen'],seed=str(1989)+ZN)
            auxIDX += 1
            plot_series_by_ind_and_layer(datadict['s2']['gen'],seed=str(20232)+ZN)
            if multilingual:
                auxIDX += 1
                plot_series_by_ind_and_layer(datadict['sla']['gen'],seed=ZN, modname='sla-eng')
                auxIDX += 1
                plot_series_by_ind_and_layer(datadict['ine']['gen'],seed=ZN, modname='ine-eng')
                auxIDX += 1
                plot_series_by_ind_and_layer(datadict['mul']['gen'],seed=ZN, modname='mul-eng')

            plt.show()
        auxIDX = 1
        for func in datadict['s0']['gen'].func.unique():
            figure  = scatter_2seeds(datadict['s0']['gen'], '1111', datadict['s1']['gen'], '1988', func)
            scatter_2seeds(datadict['s0']['gen'], '1111', datadict['s2']['gen'], '20232', func, overlapfig=figure)
            scatter_2seeds(datadict['s1']['gen'], '1989', datadict['s2']['gen'], '20232', func, overlapfig=figure)
            if multilingual:    
                scatter_2seeds(datadict['s0']['gen'], '1111', datadict['sla']['gen'], 'sla', func, overlapfig=figure)
                scatter_2seeds(datadict['s0']['gen'], '1111', datadict['ine']['gen'], 'ine', func, overlapfig=figure)
                scatter_2seeds(datadict['s0']['gen'], '1111', datadict['mul']['gen'], 'mul', func, overlapfig=figure)
                auxIDX+=1
            #savefig... (f'RUS-ENG_SCATTERseeds_{func}.pdf') <- need the right size

        # ATTEMPT: Granger causality tests
        bleu = pd.read_csv(f'results/bleu-scores2.csv').sort_values('checkpoint')
        bleu = bleu.set_index('checkpoint').sort_index()
        # HUOM! This one is not being rendered for some reason...
        bleu.groupby(['model'])['bleu'].plot(kind='line', legend=True, alpha=0.7)
        #plt.show()  
        #import ipdb; ipdb.set_trace()

        grangertests=pd.DataFrame(columns=['model','layer','component','function','pval'])
        for m1,pd1 in datadict.items():
            m2 = f'{m1}-eng' if len(m1)>2 else f'rus-eng_{m1}'
            for layer,func,comp in itertools.product(range(1,7),pd1['gen'].func.unique(),['I','S','T','F','C']):
                s1  =pd1['gen'][(pd1['gen'].func==func)& (pd1['gen'].layer_idx==layer) ].set_index('checkpoint')[comp]
                s2 = bleu[bleu.model==m2].bleu
                # we use diff'd series becaudse TSs must be stationary to compute the Granger test. 
                s1 = s1.diff().dropna()
                s2 = s2[1:]
                #s2 = s2.diff().dropna()

                # ADFtest: Stationarity - H0:TS is stationary(if failed to rejec => TS not stationary.)
                warn = (adfuller(s1)[1]>=0.05, adfuller(s2)[1]>=0.05)
                if any(warn):
                    if sum(warn)==1:
                        print(f'WARN: {["TS","bleu"][warn.index(True)]} mod={m1},comp={comp},layer={layer},fn.={func},  is NOT stationary: Granger test may not be valid.')
                    else:
                        print(f'WARN: TS and bleu BOTH are not stationary: Granger test may not be valid for:mod.={m1},comp={comp},fn.={func}, layer={layer}')
                else:
                    coso = pd.DataFrame((s1,s2)).T
                    ## HUOM: 
                    ##      There was a single missing BLEU: model=rus-eng_s0 step=489000 ... filled it by hand, comment followng lines:
                    #if coso['bleu'].isna().sum() > 0:
                    #    step=coso.bleu[coso.bleu.isna()==True].index.to_numpy()
                    #    print(f"filling NAs with last observed value for mod {m1}(ckpt,comp,fn,layer={step},{comp},{layer},{func}):{coso.isna().sum()} ")
                    #    coso = coso.fillna(method='ffill')
                    testres= grangercausalitytests(coso[[comp,'bleu']], maxlag=15, verbose=False)
                    test='ssr_chi2test'
                    pvals=[round(testres[i+1][0][test][1],4) for i in range(15)]
                    # (H0): Time series x(model,layer,func)=comp DOEN NOT Granger-cause time series y(model)=blue
                    grangertests = grangertests.append(   pd.DataFrame([[m1,layer,comp,func,np.min(pvals)]], columns=grangertests.columns)   )   
        print(f"""
            Done {len(grangertests)} Granger-causality tests out of all {4*6*6*5}, using 
                H0: Time series X(model,layer,func)=comp DOES NOT Granger-cause time series Y(model)=blue 
                    forall model in {list(datadict)}, layer=1,...,6 ,func in {list(pd1['gen'].func.unique())} comp in [I,S,T,F,C]""")
        print("Reject H0 in all cases BUT:",grangertests[grangertests.pval>0.05])
        import ipdb; ipdb.set_trace()
        #   ATTEMPT: Dynamic Time Warp
        if sys.argv[2]=='multilingual':
            maxckpt = min(datadict['s0']['gen'].checkpoint.max(),
                          datadict['s1']['gen'].checkpoint.max(),
                          datadict['s2']['gen'].checkpoint.max(),
                          datadict['sla']['gen'].checkpoint.max(),
                          datadict['ine']['gen'].checkpoint.max(),
                          datadict['mul']['gen'].checkpoint.max(),)
            models  ={'m1': {'df':datadict['s0']['gen'][datadict['s0']['gen'].checkpoint <= maxckpt].copy(), 'modname':'rus-eng s1'},
                    'm2'  : {'df':datadict['s1']['gen'][datadict['s1']['gen'].checkpoint <= maxckpt].copy(), 'modname':'rus-eng s2'},
                    'm3'  : {'df':datadict['s2']['gen'][datadict['s2']['gen'].checkpoint <= maxckpt].copy(), 'modname':'rus-eng s3'},
                    'sla': {'df':datadict['sla']['gen'][datadict['sla']['gen'].checkpoint <= maxckpt].copy(), 'modname':'sla-eng'},
                    'ine': {'df':datadict['ine']['gen'][datadict['ine']['gen'].checkpoint <= maxckpt].copy(), 'modname':'ine-eng'},
                    'mul': {'df':datadict['mul']['gen'][datadict['mul']['gen'].checkpoint <= maxckpt].copy(), 'modname':'mul-eng'},}
        else:
            maxckpt = min(datadict['s0']['gen'].checkpoint.max(),datadict['s1']['gen'].checkpoint.max(),datadict['s2']['gen'].checkpoint.max())
            models  ={'m1': {'df':datadict['s0']['gen'][datadict['s0']['gen'].checkpoint <= maxckpt].copy(), 'modname':'rus-eng s1'},
                    'm2'  : {'df':datadict['s1']['gen'][datadict['s1']['gen'].checkpoint <= maxckpt].copy(), 'modname':'rus-eng s2'},
                    'm3'  : {'df':datadict['s2']['gen'][datadict['s2']['gen'].checkpoint <= maxckpt].copy(), 'modname':'rus-eng s3'},}



        dfdist = pd.DataFrame(columns=['model1','model2', 'metric','component', 'layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6'])
        nom = "_multilingual" if sys.argv[2]=='multilingual' else "_bilingual"
        if os.path.exists(f'results/DTWdistances-res-gen{nom}.csv'):
            dfdist = pd.read_csv(f'results/DTWdistances-res-gen{nom}.csv')
        else:
            for func in datadict['s0']['gen'].func.unique():
                for comp in ['I', 'S', 'T', 'F', 'C']:
                    print(func,comp)
                    dfdist = plot_dtw_allseeds(models, dfdist, func, comp, sys.argv[2]=='multilingual')
                    auxIDX+=1
            dfdist.to_csv(f'results/DTWdistances-res-gen{nom}.csv', index=False)


        # dfdist = pd.read
        df = dfdist.copy().reset_index(drop=True)
        plotDTWheatmaps(df)

        #plt.show()
        # PERFORM MannWhitney U-test to see if distance distributions are equivalent:
        permutation_and_U_tests(df)

        #import ipdb; ipdb.set_trace()


    else:
        if False:
            plot_series_by_ind_and_layer(datadict['s0']['gen'],seed=str(1111)+ZN)
            auxIDX += 1
            plot_series_by_ind_and_layer(datadict['s1']['gen'],seed=str(1989)+ZN)
            auxIDX += 1
            plot_series_by_ind_and_layer(datadict['s2']['gen'],seed=str(20232)+ZN)
            if multilingual:
                auxIDX += 1
                plot_series_by_ind_and_layer(datadict['sla']['gen'],seed=ZN, modname='sla-eng')
                auxIDX += 1
                plot_series_by_ind_and_layer(datadict['ine']['gen'],seed=ZN, modname='ine-eng')
                auxIDX += 1
                plot_series_by_ind_and_layer(datadict['mul']['gen'],seed=ZN, modname='mul-eng')

            plt.show()
        auxIDX = 1
        for func in datadict['s0']['gen'].func.unique():
            figure  = scatter_2seeds(datadict['s0']['gen'], 's1', datadict['s0']['non-gen'], 'ng-s1', func)
            scatter_2seeds(datadict['s1']['gen'], 's2', datadict['s1']['non-gen'], 'ng-s2', func, overlapfig=figure)
            scatter_2seeds(datadict['s2']['gen'], 's3', datadict['s2']['non-gen'], 'ng-s3', func, overlapfig=figure)
            scatter_2seeds(datadict['sla']['gen'], 'sla', datadict['sla']['non-gen'], 'ng-sla', func, overlapfig=figure)
            scatter_2seeds(datadict['ine']['gen'], 'ine', datadict['ine']['non-gen'], 'ng-ine', func, overlapfig=figure)
            scatter_2seeds(datadict['mul']['gen'], 'mul', datadict['mul']['non-gen'], 'ng-mul', func, overlapfig=figure)
            auxIDX+=1
            #savefig... (f'RUS-ENG_SCATTERseeds_{func}.pdf') <- need the right size
        plt.show()
        # ATTEMPT: Granger causality tests

if __name__ == '__main__':
    multilingual = True if sys.argv[2]=='multilingual' else False
    main(multilingual)
"""
FAILED ATTEMPT: Pearson correlation & rolling-window version of it... 
                failed bc the homoscedasticity assumption does not hold

Pearson correlation tests to measure how two continuous signals co-vary over time and 
indicate the linear relationship as a number between -1 (negatively correlated) to 0 (not correlated) to 1 (perfectly correlated).
ASSUMPTION:
     homoscedasticity of the data (i.e., equal variance throughout observations)

df = df =  pd.DataFrame(s[s['func']==f].filter(['layer_idx',f'{ind}','checkpoint'])).reset_index(drop=True)
df = df.pivot(index='checkpoint', columns='layer_idx',values=f'{ind}')


# PART 1: is the homoscedasticity assumprion holds?
from sklearn.linear_model import LinearRegression

x,y = df.index.array.reshape(-1,1),df[6].array.reshape(-1,1)
regr = LinearRegression()
regr.fit(x,y)
y_hat= regr.predict(y)
residuals= y-y_hat
plt.scatter(y_hat,residuals)
# Breusch-Pagan test
residuals2 = residuals ** 2 # square the residuals
regr2 = LinearRegression()  # new regression model using the squared residuals as the response values
regr2.fit(x,residuals2)
# Calculate the Chi-Square test statistic X2 as num_observs * R2 (R2 is coef of determination of regr2)
r2 = regr2.score(x,residuals2) 
"""

